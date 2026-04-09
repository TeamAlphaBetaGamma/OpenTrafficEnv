from simulator.env import TrafficSimulator
from models import GlobalState, IntersectionState, LaneState
from agent.policy import decide_all_phases

def build_global_state(env):
    """Dynamically parses the local simulator data into the strictly typed GlobalState Pydantic object so the AI can read it."""
    intersections = []
    idx = 0
    for i in range(env.size):
        for j in range(env.size):
            inter = env.grid[i][j]
            
            def make_lane(d):
                lane_cars = inter.lanes[d]
                cum_wait = sum(v.wait_time for v in lane_cars)
                cum_fuel = sum(v.fuel_consumed for v in lane_cars)
                has_emergency = any(v.is_emergency for v in lane_cars)
                return LaneState(
                    queue_length=len(lane_cars),
                    cumulative_wait=float(cum_wait),
                    emergency_flag=has_emergency,
                    fuel_consumed=float(cum_fuel)
                )

            max_w = 0.0
            for l in inter.lanes.values():
                for v in l:
                    max_w = max(max_w, v.wait_time)

            inter_state = IntersectionState(
                intersection_id=idx,
                north=make_lane("N"),
                south=make_lane("S"),
                east=make_lane("E"),
                west=make_lane("W"),
                current_phase=inter.current_phase,
                phase_duration=inter.phase_duration,
                max_wait=float(max_w),
                upstream_hint=None
            )
            intersections.append(inter_state)
            idx += 1
            
    task_map = {"easy": 1, "medium": 2, "hard": 3}
    return GlobalState(
        intersections=intersections,
        step_number=env.step_count,
        task_id=task_map.get(env.task, 1),
        done=env.step_count >= env.max_steps
    )

# Initialize the environment
env = TrafficSimulator(task="hard") 
state = env.reset()

total_episode_reward = 0

print(f"--- Starting Simulation: Task {env.task.upper()} ---")
print("NOTE: Ensure API_KEY, API_BASE_URL, and MODEL_NAME are configured in your console to use the AI!")

# Run for 10 steps (or use env.max_steps for a full episode)
for i in range(10):
    # 1. Manually build the required GlobalState from physics grid for the AI to read
    obs = build_global_state(env)
    
    # 2. Ask the AI for its action list
    ai_actions_list = decide_all_phases(obs.intersections)
    
    # 3. Convert the Pydantic list into a dictionary for the environment
    actions = {}
    for act in ai_actions_list:
        row = act.intersection_id // env.size
        col = act.intersection_id % env.size
        actions[f"{row},{col}"] = act.phase
        
    # 4. Take a step
    state, reward, done, _ = env.step(actions)
    total_episode_reward += reward
    
    print(f"Step {i:02d} | Step Reward: {reward:7.2f} | Completed Trips: {env.completed_trips}")

print("-" * 40)

# --- THE NORMALIZATION STEP ---
final_score = env.get_score(total_episode_reward)

print(f"Raw Cumulative Reward: {total_episode_reward:.2f}")
print(f"FINAL NORMALIZED SCORE: {final_score:.4f}") 
print("-" * 40)