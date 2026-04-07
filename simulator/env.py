import random
from simulator.intersection import Intersection
from simulator.vehicle import Vehicle
class TrafficEnv:
    def __init__(self):
        self.grid = [[Intersection() for _ in range(2)] for _ in range(2)]
        self.step_count = 0

    def reset(self):
        self.grid = [[Intersection() for _ in range(2)] for _ in range(2)]
        self.step_count = 0
        return {}
    def step(self, actions):
        total_reward = 0
        outgoing = []
        # Process all intersections
        for i in range(2):
            for j in range(2):
                intersection = self.grid[i][j]
                # 1. add new vehicles
                if random.random() < 0.5:
                    intersection.add_vehicle("N", Vehicle())
                # 2. apply action
                intersection.current_phase = actions.get(f"{i},{j}", 0)
                # 3. process signal
                moved = intersection.process_signal()
                # store outgoing vehicles
                for direction, vehicle in moved:
                    outgoing.append((i, j, direction, vehicle))
                # 4. update waiting
                intersection.update_waiting()
                # 5. reward
                total_reward += len(moved)
        for i, j, direction, vehicle in outgoing:
            ni, nj = i, j

            if direction == "N":
                ni -= 1
            elif direction == "S":
                ni += 1
            elif direction == "E":
                nj += 1
            elif direction == "W":
                nj -= 1
            if 0 <= ni < 2 and 0 <= nj < 2:
                self.grid[ni][nj].add_vehicle(direction, vehicle)

        self.step_count += 1

        return {}, total_reward, False, {}