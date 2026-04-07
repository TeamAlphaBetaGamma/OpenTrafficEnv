from simulator.env import TrafficEnv

env = TrafficEnv()
env.reset()

for i in range(10):
    actions = {
    "0,0": i % 2,
    "0,1": (i+1) % 2,
    "1,0": i % 2,
    "1,1": (i+1) % 2
    }   
    state, reward, done, _ = env.step(actions)
    print(f"Step {i} | Reward: {reward}")