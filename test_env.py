from simulator.env import TrafficEnv

env = TrafficEnv()
env.reset()

for i in range(10):
    state, reward, done, _ = env.step({"0,0": i % 2})
    print(f"Step {i} | Reward: {reward}")