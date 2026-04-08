---
title: AlphaBetaGamma
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Adaptive urban traffic signal control
---

# AlphaBetaGamma (Docker Space)

This repository packages a lightweight **urban traffic signal control** environment as an HTTP API (FastAPI) plus a reference **hybrid policy** (rules → LLM → greedy fallback) and an **inference runner** that drives the environment via its API.

The Hugging Face Space runs the API server from the Dockerfile.

## What’s included

- **Environment server**: `server.py` (FastAPI + Pydantic models)
- **Traffic simulator**: `simulator/` (grid of intersections, vehicle spawning, movement, reward signals)
- **Agent policy**: `agent/policy.py` (min-green, emergency priority, fairness override, optional LLM, greedy fallback)
- **Reward utilities**: `agent/reward.py` (step reward + episode score)
- **Client/inference**: `inference.py` (runs tasks 1–3 by calling the server)
- **Runtime smoke test**: `validate_runtime.py` (starts server locally and runs `inference.py` with LLM disabled)
- **Logged runner**: `run_logged.py` (runs `inference.py` and writes a combined log + wall-clock time)
- **Task grader**: `agent/grader.py` (deterministic episode scoring + success threshold)

## HTTP API

The Space exposes a JSON API on port `7860`.

### `GET /` and `GET /health` and `GET /ping`

- `GET /` returns a small JSON payload listing the available endpoints.
- `GET /health` returns `{"status":"ok"}` for liveness checks.
- `GET /ping` returns `{"status":"ok"}` for OpenEnv-style ping checks.

### `POST /reset`

Initializes a new episode.

Body:

```json
{"task_id": 1, "seed": 42}
```

- `task_id`: `1` (easy), `2` (medium), `3` (hard)
- `seed`: optional for reproducibility

Returns: `ResetResponse` with `observation` (global state snapshot).

### `POST /step`

Advances the simulator by one step.

Body:

```json
{
	"actions": [
		{"intersection_id": 0, "phase": 0}
	]
}
```

- `phase`: `0` = North/South green, `1` = East/West green

Returns: `StepResult` containing `observation`, `reward` (normalized to `[0,1]`), `done`, and `info`.

### `GET /state`

Returns the current `GlobalState`. If you haven’t called `/reset` yet, the server returns `409`.

### `GET /docs`

FastAPI Swagger UI.

## Run locally (Python)

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

In a second terminal:

```bash
curl -sS -X POST http://127.0.0.1:7860/reset \
	-H "Content-Type: application/json" \
	-d '{"task_id":1,"seed":42}'
```

Run the reference inference loop:

```bash
export ENV_BASE_URL=http://127.0.0.1:7860
python3 inference.py
```

### Generate a combined run log (recommended)

This repo includes a small helper that runs the environment end-to-end and writes a single log file containing:

- server readiness check
- full `inference.py` output
- `exit_code` and `wall_seconds` summary

It also loads variables from a local `.env` file if present.

Run with LLM disabled (deterministic + faster):

```bash
python3 run_logged.py --disable-llm 1 --log runs/disable_llm_1.log
```

Run with LLM enabled:

```bash
python3 run_logged.py --disable-llm 0 --log runs/disable_llm_0.log
```

By default, `run_logged.py` uses `ENV_BASE_URL` from `.env`/environment. To force local:

```bash
python3 run_logged.py --disable-llm 1 --base-url http://127.0.0.1:7860 --log runs/local_disable_llm_1.log
```

## Run locally (Docker)

```bash
docker build -t alphabetagamma .
docker run --rm -p 7860:7860 alphabetagamma
```

### Validate under 2 vCPU / 8GB

Run the server with hard limits:

```bash
docker run --rm -p 7860:7860 --cpus=2 --memory=8g --name opentrafficenv_local alphabetagamma
```

Then drive it with the client (LLM disabled for deterministic runtime):

```bash
ENV_BASE_URL=http://127.0.0.1:7860 DISABLE_LLM=1 python3 inference.py
```

## Configuration (env vars)

You can set env vars in your shell, or put them in a local `.env` file.

Important: `.env` commonly contains secrets (e.g., API keys). Do not commit it.

- `ENV_BASE_URL`: where `inference.py` calls the environment (default `http://localhost:7860`)
- `DISABLE_LLM`: set to `1` to force the policy to avoid LLM calls
- `HF_TOKEN`: OpenAI API key (checklist-required; used when LLM is enabled)
- `OPENAI_API_KEY`: supported alias for the same key
- `API_BASE_URL`: OpenAI-compatible base URL (default `https://api.openai.com/v1`)
- `MODEL_NAME`: model name string used by the policy client
- `VALIDATION_TIMEOUT_S`: timeout for `validate_runtime.py` (default `1200`)

## Notes

- The environment uses a **minimum green time** constraint (prevents flickering).
- The simulator returns a normalized `reward` per step in `[0,1]`.
	- Reward is designed to stay informative under congestion (it avoids large negative sums that would collapse to `0.0`).
	- `agent/reward.py` can compute a reward from `info` as a fallback if needed.

## Repository layout

```
.
├── Dockerfile
├── server.py
├── models.py
├── inference.py
├── validate_runtime.py
├── requirements.txt
├── agent/
└── simulator/
```
