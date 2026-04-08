from __future__ import annotations

import os
import resource
import subprocess
import sys
import time
import urllib.request


def _wait_for_server(base_url: str, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None

    url = f"{base_url}/state"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                # 200 means env already reset; fine.
                if resp.status in (200,):
                    return
        except Exception as e:
            # 409 before /reset is also fine; it means server is up.
            if hasattr(e, "code") and getattr(e, "code") == 409:
                return
            last_err = e
            time.sleep(0.2)

    raise RuntimeError(f"Server did not become ready within {timeout_s}s. Last error: {last_err}")


def main() -> int:
    base_url = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")
    timeout_s = float(os.environ.get("VALIDATION_TIMEOUT_S", "1200"))

    # Start server
    server_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "7860",
        "--log-level",
        "warning",
    ]

    start_wall = time.perf_counter()
    server = subprocess.Popen(
        server_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        _wait_for_server(base_url)

        # Run inference against local server, but disable LLM calls so performance is sim-bound.
        env = {
            **os.environ,
            "ENV_BASE_URL": base_url,
            "DISABLE_LLM": "1",
        }

        subprocess.run(
            [sys.executable, "inference.py"],
            env=env,
            check=True,
            timeout=timeout_s,
        )

        elapsed_s = time.perf_counter() - start_wall

        # ru_maxrss units:
        # - macOS: bytes
        # - Linux: kilobytes
        ru_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ru_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

        print("\n=== Validation Summary ===")
        print(f"Wall time (s): {elapsed_s:.2f}")
        print(f"Max RSS self: {ru_self}")
        print(f"Max RSS children: {ru_children}")
        print("Note: ru_maxrss units differ by OS (macOS=bytes, Linux=KB).")

        # We can't hard-enforce CPU=2 and RAM=8GB without cgroups (Docker/HF).
        # This script is a smoke test + timing guardrail.
        if elapsed_s > 20 * 60:
            print("FAIL: exceeded 20-minute window")
            return 2

        print("PASS: completed within 20-minute window")
        return 0

    except subprocess.TimeoutExpired:
        print("FAIL: timed out (20-minute window)")
        return 2
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
