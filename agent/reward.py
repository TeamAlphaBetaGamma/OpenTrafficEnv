from models import StepInfo

FAIRNESS_THRESHOLD: float = 20.0   # steps. Match this with policy.py constant
EMERGENCY_PENALTY_PER_STEP: float = 0.05  # penalty per step an emergency car is blocked


def compute_reward(info: StepInfo) -> float:
    """
    Compute normalized reward from step info.
    """
    total_cars_safe = info.total_cars + 1

    throughput_bonus   = info.cars_passed / total_cars_safe
    wait_penalty       = info.total_wait / total_cars_safe
    fuel_penalty       = info.total_fuel / total_cars_safe
    fairness_penalty   = info.max_wait / FAIRNESS_THRESHOLD
    emergency_penalty  = info.emergency_delay * EMERGENCY_PENALTY_PER_STEP

    raw = throughput_bonus - wait_penalty - fuel_penalty - fairness_penalty - emergency_penalty

    # valid range
    return float(max(0.0, min(1.0, raw)))


def compute_episode_score(rewards: list[float]) -> float:
    """
    Aggregate per-step rewards into an episode score.
    """
    if not rewards:
        return 0.0
    return float(sum(rewards) / len(rewards))