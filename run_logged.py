from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _parse_dotenv_file(path: Path) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE per line; ignores comments and blanks)."""
    if not path.exists():
        return {}

    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Strip surrounding quotes if present
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        out[key] = value
    return out


def _merged_env_with_dotenv(base_dir: Path) -> dict[str, str]:
    """Merge current process env with values from .env (dotenv fills only missing keys)."""
    env = dict(os.environ)
    dotenv = _parse_dotenv_file(base_dir / ".env")
    for k, v in dotenv.items():
        env.setdefault(k, v)
    return env


def _should_start_local_server(base_url: str) -> bool:
    """Start a local uvicorn server only for localhost/127.0.0.1 targets."""
    u = urlparse(base_url)
    host = (u.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        return False
    # If port is unspecified, assume default 80/443 which is not our local uvicorn setup.
    port = u.port
    if port is None:
        return False
    return port == 7860


def _wait_for_server(base_url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None

    # Prefer endpoints that should always be available.
    urls = [f"{base_url}/ping", f"{base_url}/health", f"{base_url}/state"]
    while time.time() < deadline:

        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    if resp.status == 200:
                        return
            except Exception as e:
                # 409 before /reset is also fine; it means server is up.
                if url.endswith("/state") and hasattr(e, "code") and getattr(e, "code") == 409:
                    return
                last_err = e

        time.sleep(0.2)

    raise RuntimeError(f"Server did not become ready within {timeout_s}s. Last error: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local server + inference and write a combined log.")
    parser.add_argument("--disable-llm", choices=["0", "1"], required=True)
    parser.add_argument("--log", required=True, help="Path to log file (will be overwritten).")
    parser.add_argument("--base-url", default=None, help="Overrides ENV_BASE_URL; if omitted, loads from .env/env.")
    parser.add_argument("--server-timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = _merged_env_with_dotenv(base_dir)

    base_url = args.base_url or env.get("ENV_BASE_URL") or "http://127.0.0.1:7860"
    env["ENV_BASE_URL"] = base_url
    env["DISABLE_LLM"] = args.disable_llm

    start_server = _should_start_local_server(base_url)

    server_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.app:app",
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
        w(f"ENV_BASE_URL={base_url}")
        w(f"DISABLE_LLM={args.disable_llm}")
        w(f"python={sys.executable}")
        w("")
        if start_server:
            w("=== Starting server (uvicorn) ===")

            server = subprocess.Popen(
                server_cmd,
                stdout=logf,
                stderr=logf,
                env={**env, "PYTHONUNBUFFERED": "1"},
            )

            time.sleep(0.2)
            if server.poll() is not None:
                w(f"Server exited early with code {server.returncode}")
                return 2

        else:
            w("=== Using external server (no local uvicorn) ===")

        try:
            _wait_for_server(base_url, timeout_s=args.server_timeout_s)
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
            if start_server:
                w("=== Stopping server ===")
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
