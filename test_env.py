from simulator.env import TrafficSimulator

# Initialize the environment
env = TrafficSimulator(task="hard") 
state = env.reset()

total_episode_reward = 0

print(f"--- Starting Simulation: Task {env.task.upper()} ---")

# Run for 10 steps (or use env.max_steps for a full episode)
for i in range(10):
    # Generate actions: alternate 0 and 1
    actions = {k: i % 2 for k in state} 
    
    # Take a step
    state, reward, done, _ = env.step(actions)
    total_episode_reward += reward
    
    print(f"Step {i:02d} | Step Reward: {reward:7.2f} | Completed Trips: {env.completed_trips}")

print("-" * 40)

# --- THE NORMALIZATION STEP ---
final_score = env.get_score(total_episode_reward)

print(f"Raw Cumulative Reward: {total_episode_reward:.2f}")
print(f"FINAL NORMALIZED SCORE: {final_score:.4f}") 
print("-" * 40)