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
    score = compute_episode_score(rewards)
    threshold = SUCCESS_SCORE_THRESHOLD_BY_TASK.get(task_id, 0.5)
    return GradeResult(task_id=task_id, score=float(score), success=bool(score >= threshold))
