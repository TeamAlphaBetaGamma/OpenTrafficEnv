from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _wait_for_server(base_url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None

    url = f"{base_url}/state"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except Exception as e:
            # 409 before /reset is also fine; it means server is up.
            if hasattr(e, "code") and getattr(e, "code") == 409:
                return
            last_err = e
            time.sleep(0.2)

    raise RuntimeError(f"Server did not become ready within {timeout_s}s. Last error: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local server + inference and write a combined log.")
    parser.add_argument("--disable-llm", choices=["0", "1"], required=True)
    parser.add_argument("--log", required=True, help="Path to log file (will be overwritten).")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--server-timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["ENV_BASE_URL"] = args.base_url
    env["DISABLE_LLM"] = args.disable_llm

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

    t0 = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as logf:
        def w(line: str = "") -> None:
            logf.write(line + "\n")
            logf.flush()

        w("=== OpenTrafficEnv Run Log ===")
        w(f"started_at_utc={_utc_now_iso()}")
        w(f"ENV_BASE_URL={args.base_url}")
        w(f"DISABLE_LLM={args.disable_llm}")
        w(f"python={sys.executable}")
        w("")
        w("=== Starting server (uvicorn) ===")

        server = subprocess.Popen(
            server_cmd,
            stdout=logf,
            stderr=logf,
            env={**env, "PYTHONUNBUFFERED": "1"},
        )

        try:
            time.sleep(0.2)
            if server.poll() is not None:
                w(f"Server exited early with code {server.returncode}")
                return 2

            _wait_for_server(args.base_url, timeout_s=args.server_timeout_s)
            w("=== Server ready ===")
            w("")
            w("=== Running inference.py ===")

            r = subprocess.run(
                [sys.executable, "inference.py"],
                env=env,
                stdout=logf,
                stderr=logf,
            )

            elapsed = time.perf_counter() - t0
            w("")
            w("=== Run summary ===")
            w(f"exit_code={r.returncode}")
            w(f"wall_seconds={elapsed:.3f}")
            return r.returncode

        finally:
            w("")
            w("=== Stopping server ===")
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
