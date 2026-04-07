import random
from simulator.intersection import Intersection
from simulator.vehicle import Vehicle

class TrafficEnv:
    def __init__(self):
        self.grid = [[Intersection()]]  # single intersection
        self.step_count = 0

    def reset(self):
        self.grid = [[Intersection()]]
        self.step_count = 0
        return {}

    def step(self, actions):
        intersection = self.grid[0][0]

        # 1. spawn vehicle
        if random.random() < 0.5:
            intersection.add_vehicle("N", Vehicle())

        # 2. apply action
        intersection.current_phase = actions["0,0"]

        # 3. process signal
        moved = intersection.process_signal()

        # 4. update waiting
        intersection.update_waiting()

        # 5. reward
        reward = len(moved)

        self.step_count += 1

        return {}, reward, False, {}