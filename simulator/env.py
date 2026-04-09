import random
import math
from collections import deque
from .intersection import Intersection
from .vehicle import Vehicle


class TrafficSimulator:
    def __init__(self, task="easy", seed_val=42):
        self.task = task
        self.base_seed = seed_val
        self.set_seed(seed_val)
        
        # Difficulty Config
        configs = {"easy": (1, 0.2), "medium": (2, 0.5), "hard": (3, 0.8)}
        self.size, self.arrival_rate = configs.get(task, (1, 0.2))

        self.grid = [[Intersection() for _ in range(self.size)] for _ in range(self.size)]
        self.step_count = 0
        self.max_steps = 50
        self.completed_trips = 0

    def set_seed(self, seed):
        self.base_seed = seed
        random.seed(seed)

    def reset(self, max_steps=50):
        self.set_seed(self.base_seed)
        self.grid = [[Intersection() for _ in range(self.size)] for _ in range(self.size)]
        self.step_count = 0
        self.max_steps = max_steps
        self.completed_trips = 0
        return self.get_state()

    def step(self, actions):
        outgoing = []
        step_completed = 0
        moved_total = 0

        # Aggregate info for StepInfo
        info_wait = 0.0
        info_fuel = 0.0
        info_cars = 0
        info_max_wait = 0.0
        info_emergency_delay = 0.0  # max emergency wait (not a summed penalty)

        # Reward is computed as an average across intersections then mapped to [0,1]
        raw_sum = 0.0
        n_inters = self.size * self.size

        for i in range(self.size):
            for j in range(self.size):
                intersection = self.grid[i][j]
                
                # 1. Update Phase (with Min Green constraint)
                if intersection.phase_duration >= 3:
                    # Fix: Default to current phase instead of 0
                    new_phase = actions.get(f"{i},{j}", intersection.current_phase)
                    if new_phase != intersection.current_phase:
                        intersection.current_phase = new_phase
                        intersection.phase_duration = 0
                
                intersection.phase_duration += 1

                # 2. Spawn Vehicles (with Rush Hour Logic)
                rate = 0.95 if (self.task == "hard" and 20 <= self.step_count <= 35) else self.arrival_rate
                if random.random() < rate:
                    is_e = random.random() < 0.1
                    intersection.add_vehicle(random.choice(["N","S","E","W"]), Vehicle(is_e))

                # 3. Process & Movement
                moved = intersection.process_signal()
                moved_total += len(moved)
                for d, v in moved:
                    outgoing.append((i, j, d, v))

                # 4. Metrics & Reward Calculation
                intersection.update_waiting()

                t_cars = 0
                sum_wait = 0.0
                sum_fuel = 0.0
                local_max_wait = 0.0
                local_emergency_max_wait = 0.0

                for lane in intersection.lanes.values():
                    for v in lane:
                        t_cars += 1
                        sum_wait += float(v.wait_time)
                        sum_fuel += float(v.fuel_consumed)
                        if v.wait_time > local_max_wait:
                            local_max_wait = float(v.wait_time)
                        if v.is_emergency and v.wait_time > local_emergency_max_wait:
                            local_emergency_max_wait = float(v.wait_time)

                # Aggregate StepInfo-style metrics
                info_cars += t_cars
                info_wait += sum_wait
                info_fuel += sum_fuel
                info_max_wait = max(info_max_wait, local_max_wait)
                info_emergency_delay = max(info_emergency_delay, local_emergency_max_wait)

                # Reward shaping (avoid huge negative sums that clip to 0)
                # - Throughput: fraction of vehicles that moved this step (bounded)
                # - Penalties: normalized averages, capped to [0,1]
                denom = t_cars + len(moved) + 1
                moved_ratio = len(moved) / denom

                if t_cars > 0:
                    avg_wait = sum_wait / t_cars
                    avg_fuel = sum_fuel / t_cars
                else:
                    avg_wait = 0.0
                    avg_fuel = 0.0

                # Typical ranges: avg_wait tens of steps; fuel grows ~0.1/step
                wait_norm = min(1.0, avg_wait / 30.0)
                fuel_norm = min(1.0, avg_fuel / 3.0)
                emergency_norm = min(1.0, local_emergency_max_wait / 30.0)

                raw_sum += (
                    moved_ratio
                    - 0.35 * wait_norm
                    - 0.10 * fuel_norm
                    - 0.25 * emergency_norm
                )

        # 5. Global Movement Logic
        for i, j, d, v in outgoing:
            ni, nj = i + (1 if d=="S" else -1 if d=="N" else 0), j + (1 if d=="E" else -1 if d=="W" else 0)
            if 0 <= ni < self.size and 0 <= nj < self.size:
                opp = {"N":"S","S":"N","E":"W","W":"E"}
                self.grid[ni][nj].add_vehicle(opp[d], v)
            else:
                step_completed += 1
                self.completed_trips += 1

        # Add a small bounded bonus for trips that exit the grid (helps sparse cases)
        raw_avg = (raw_sum / max(1, n_inters)) + (0.10 * (step_completed / max(1, n_inters)))

        # Map to (0,1) smoothly so values don't collapse to exactly 0
        step_reward = 1.0 / (1.0 + math.exp(-3.0 * raw_avg))
        
        # Clamp step reward strictly within (0, 1) to satisfy validator requirements
        eps = 1e-6
        step_reward = max(eps, min(1.0 - eps, step_reward))

        info_dict = {
            # Use moved vehicles as "cars_passed" so throughput isn't zero when nothing exits the grid.
            "cars_passed": int(moved_total),
            "total_cars": int(info_cars),
            "total_wait": float(info_wait),
            "total_fuel": float(info_fuel),
            "max_wait": float(info_max_wait),
            "emergency_delay": float(info_emergency_delay),
        }
        
        self.step_count += 1
        return self.get_state(), float(step_reward), (self.step_count >= self.max_steps), info_dict

    def get_state(self):
        state = {}
        for i in range(self.size):
            for j in range(self.size):
                inter = self.grid[i][j]
                has_e = any(v.is_emergency for l in inter.lanes.values() for v in l)
                state[f"{i},{j}"] = {
                    "N": len(inter.lanes["N"]), "S": len(inter.lanes["S"]),
                    "E": len(inter.lanes["E"]), "W": len(inter.lanes["W"]),
                    "emergency_alert": has_e,
                    "phase": inter.current_phase,
                    "phase_time": inter.phase_duration
                }
        return state

    def get_score(self, total_reward):
        # Professional 0-1 Normalization based on sum of 0-1 step rewards
        eps = 1e-6
        if self.step_count == 0:
            return eps
        return max(eps, min(1.0 - eps, total_reward / self.step_count))