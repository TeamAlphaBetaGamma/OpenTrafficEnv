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
    # CRITICAL: eps must be >= 0.005 so that when formatted as :.2f it shows "0.01" not "0.00".
    # 0.001 formatted as :.2f rounds DOWN to "0.00" which fails the (0, 1) check.
    eps = 0.01
    score = max(eps, min(1.0 - eps, score))
    if score <= 0.0:
        score = eps
    if score >= 1.0:
        score = 1.0 - eps
    threshold = SUCCESS_SCORE_THRESHOLD_BY_TASK.get(task_id, 0.5)
    return GradeResult(task_id=task_id, score=score, success=bool(score >= threshold))
