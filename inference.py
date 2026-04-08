import os
import sys
import json
import logging
import requests
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    GlobalState, TrafficAction, StepResult, ResetRequest, StepInfo
)
from agent.policy import decide_all_phases, _get_openai_client
from agent.reward import compute_reward, compute_episode_score
from agent.grader import grade_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASK_CONFIG = {
    1: {
        "description": "1x1 grid, low density (P=0.2), no emergencies",
        "max_steps": 100,
        "seed": 42,
    },
    2: {
        "description": "2x2 grid, moderate density (P=0.4), 1 emergency vehicle",
        "max_steps": 150,
        "seed": 42,
    },
    3: {
        "description": "3x3 grid, high density (P=0.7), burst traffic, multiple emergencies",
        "max_steps": 200,
        "seed": 42,
    },
}


# ── stdout log helpers ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """Emit the [START] marker. Must be called once per episode."""
    payload = {"task": task, "env": env, "model": model}
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(step: int, action: int, reward: float, done: bool, error: str = None) -> None:
    """Emit a [STEP] marker. Must be called once per environment step."""
    payload = {
        "step": step,
        "action": action,
        "reward": round(reward, 4),
        "done": done,
        "error": error
    }
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """Emit the [END] marker. Must be called once per episode."""
    payload = {
        "success": success,
        "steps": steps,
        "score": round(score, 4),
        "rewards": [round(r, 4) for r in rewards]
    }
    print(f"[END] {json.dumps(payload)}", flush=True)


# ── Environment API calls ─────────────────────────────────────────────────────

def env_reset(task_id: int, seed: int) -> GlobalState:
    """Call the environment's /reset endpoint and return the initial GlobalState."""
    url = f"{ENV_BASE_URL}/reset"
    payload = {"task_id": task_id, "seed": seed}
    logger.info(f"Calling reset: task_id={task_id}, seed={seed}")
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return GlobalState(**data["observation"])


def env_step(actions: list[TrafficAction]) -> StepResult:
    """Call the environment's /step endpoint with a list of actions."""
    url = f"{ENV_BASE_URL}/step"
    payload = {"actions": [a.model_dump() for a in actions]}
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return StepResult(**data)


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task_id: int, client: OpenAI) -> dict:
    """
    Run a complete episode for one task.
    """
    config = TASK_CONFIG[task_id]
    max_steps = config["max_steps"]
    seed = config["seed"]
    description = config["description"]

    # ── START log ─────────────────────────────────────────────────────────────
    log_start(task=f"Task {task_id}", env="OpenTrafficEnv", model=os.environ.get("MODEL_NAME", "gpt-4o-mini"))

    # ── Reset the environment ─────────────────────────────────────────────────
    obs: GlobalState = env_reset(task_id, seed)
    logger.info(f"Task {task_id} reset. Intersections: {len(obs.intersections)}")

    rewards: list[float] = []
    done = False
    step = 0

    while not done and step < max_steps:
        step += 1

        # ── Get actions from hybrid policy ────────────────────────────────────
        actions = decide_all_phases(obs.intersections, client)

        # ── Step the environment ──────────────────────────────────────────────
        result: StepResult = env_step(actions)

        # ── Compute reward (use environment's reward if available) ─
        reward = result.reward if result.reward > 0.0 else compute_reward(result.info)
        rewards.append(reward)

        done = result.done
        obs = result.observation

        # Representative action for logging: intersection 0's chosen phase
        primary_action = actions[0].phase if actions else 0

        # ── STEP log ──────────────────────────────────────────────────────────
        log_step(step=step, action=primary_action, reward=reward, done=done, error=None)

    # ── END log ───────────────────────────────────────────────────────────────
    total_reward = sum(rewards)
    grade = grade_task(task_id, rewards)
    log_end(success=grade.success, steps=step, score=grade.score, rewards=rewards)

    logger.info(f"Task {task_id} complete. Score: {grade.score:.4f}")
    return {"task_id": task_id, "total_reward": total_reward, "steps": step, "score": grade.score, "success": grade.success}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== OpenTrafficEnv Inference Script Starting ===")
    logger.info(f"ENV_BASE_URL: {ENV_BASE_URL}")
    logger.info(f"MODEL_NAME: {os.environ.get('MODEL_NAME', 'not set')}")

    # Build shared OpenAI client
    client = _get_openai_client()

    # Run all 3 tasks
    results = []
    for task_id in [1, 2, 3]:
        try:
            result = run_episode(task_id, client)
            results.append(result)
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            # Still emit END so the parser doesn't hang
            log_end(success=False, steps=0, score=0.0, rewards=[])

    logger.info("=== All tasks complete ===")
    for r in results:
        logger.info(f"  Task {r['task_id']}: score={r['score']:.4f}")


if __name__ == "__main__":
    main()