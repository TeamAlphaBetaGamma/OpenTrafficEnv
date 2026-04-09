from __future__ import annotations

from dataclasses import dataclass

from agent.reward import compute_episode_score


@dataclass(frozen=True)
class GradeResult:
    task_id: int
    score: float
    success: bool


# Deterministic success thresholds. Score itself is always in [0.0, 1.0].
SUCCESS_SCORE_THRESHOLD_BY_TASK: dict[int, float] = {
    1: 0.5,
    2: 0.5,
    3: 0.5,
}


def grade_task(task_id: int, rewards: list[float]) -> GradeResult:
    score = float(compute_episode_score(rewards))
    # The submission validator requires scores strictly within (0, 1).
    # Avoid edge cases where the underlying scorer returns 0.0 or 1.0.
    eps = 0.1
    if score <= 0.0:
        score = eps
    elif score >= 1.0:
        score = 1.0 - eps
    threshold = SUCCESS_SCORE_THRESHOLD_BY_TASK.get(task_id, 0.5)
    return GradeResult(task_id=task_id, score=score, success=bool(score >= threshold))
