import random
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
        self.max_steps = 100
        self.completed_trips = 0

    def set_seed(self, seed):
        self.base_seed = seed
        random.seed(seed)

    def reset(self, max_steps=100):
        self.set_seed(self.base_seed)
        self.grid = [[Intersection() for _ in range(self.size)] for _ in range(self.size)]
        self.step_count = 0
        self.max_steps = max_steps
        self.completed_trips = 0
        return self.get_state()

    def step(self, actions):
        total_reward = 0
        outgoing = []
        step_completed = 0
        
        info_wait, info_fuel, info_cars, info_max_wait, info_e_delay = 0, 0, 0, 0, 0

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
                for d, v in moved:
                    outgoing.append((i, j, d, v))

                # 4. Metrics & Reward Calculation
                intersection.update_waiting()
                
                t_wait, t_fuel, t_cars, m_wait, e_penalty = 0, 0, 0, 0, 0
                for lane in intersection.lanes.values():
                    for v in lane:
                        if v.wait_time > 2:
                            t_wait += v.wait_time
                            t_fuel += v.fuel_consumed
                        t_cars += 1
                        m_wait = max(m_wait, v.wait_time)
                        if v.is_emergency: e_penalty += v.wait_time * 0.5

                den = t_cars + 1
                total_reward += (len(moved) / den) - (t_wait / den) - (t_fuel / den) - (e_penalty / den)
                
                info_wait += t_wait
                info_fuel += t_fuel
                info_cars += t_cars
                info_max_wait = max(info_max_wait, m_wait)
                info_e_delay += e_penalty

        # 5. Global Movement Logic
        for i, j, d, v in outgoing:
            ni, nj = i + (1 if d=="S" else -1 if d=="N" else 0), j + (1 if d=="E" else -1 if d=="W" else 0)
            if 0 <= ni < self.size and 0 <= nj < self.size:
                opp = {"N":"S","S":"N","E":"W","W":"E"}
                self.grid[ni][nj].add_vehicle(opp[d], v)
            else:
                step_completed += 1
                self.completed_trips += 1

        total_reward += step_completed * 0.5 # Throughput bonus
        
        # Normalize step reward to fit OpenEnv specification [0.0, 1.0]
        step_reward = max(0.0, min(1.0, 0.5 + (total_reward / 20.0)))
        
        info_dict = {
            "cars_passed": step_completed,
            "total_cars": info_cars,
            "total_wait": float(info_wait),
            "total_fuel": float(info_fuel),
            "max_wait": float(info_max_wait),
            "emergency_delay": float(info_e_delay)
        }
        
        self.step_count += 1
        return self.get_state(), step_reward, (self.step_count >= self.max_steps), info_dict

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
        if self.step_count == 0:
            return 0.0
        return max(0.0, min(1.0, total_reward / self.step_count))