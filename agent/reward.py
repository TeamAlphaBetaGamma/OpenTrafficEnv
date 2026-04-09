from models import StepInfo

FAIRNESS_THRESHOLD: float = 20.0   # steps
EMERGENCY_PENALTY_PER_STEP: float = 0.05  # penalty per step
_EPS: float = 0.1  # keeps rewards strictly inside (0.0, 1.0)

def compute_reward(info: StepInfo) -> float:
    """
    Compute normalized reward from step info with capped penalties.
    """
    # Use a safety denominator
    total_cars_safe = info.total_cars + 1

    # 1. Throughput is the primary goal (Positive)
    throughput_bonus = info.cars_passed / total_cars_safe
    
    # 2. Penalties (Negative) - Normalized to 0.0 - 1.0 range
    wait_penalty      = min(1.0, info.total_wait / (total_cars_safe)) # Scale by 10 for sensitivity
    fuel_penalty      = min(1.0, info.total_fuel / (total_cars_safe))  # Fuel is already 0.1 per step
    
    # FIX: Cap the fairness penalty so it doesn't overwhelm the score
    fairness_penalty   = min(1.0, info.max_wait / FAIRNESS_THRESHOLD)
    
    # Emergency vehicles are high priority
    emergency_penalty  = min(1.0, info.emergency_delay * EMERGENCY_PENALTY_PER_STEP)

    # 3. Combine with an offset to keep the agent motivated
    raw = throughput_bonus - (0.1 * wait_penalty) - (0.1 * fuel_penalty) - (0.2 * fairness_penalty) - (0.3 * emergency_penalty)

    # Return strictly in (0.0, 1.0) — both endpoints excluded
    return float(max(_EPS, min(1.0 - _EPS, raw)))

def compute_episode_score(rewards: list[float]) -> float:
    """Average reward across all steps."""
    if not rewards:
        return _EPS
    return float(max(_EPS, min(1.0 - _EPS, sum(rewards) / len(rewards))))