from collections import deque

class Intersection:
    def __init__(self):
        self.lanes = {
            "N": deque(), "S": deque(), "E": deque(), "W": deque()
        }
        self.current_phase = 0  # 0: NS, 1: EW
        self.phase_duration = 0

    def add_vehicle(self, direction, vehicle):
        self.lanes[direction].append(vehicle)

    def process_signal(self, k=2):
        moved = []
        def has_emergency(dirs):
            return any(v.is_emergency for d in dirs for v in self.lanes[d])

        ns_e = has_emergency(["N", "S"])
        ew_e = has_emergency(["E", "W"])

        if ns_e and not ew_e: directions = ["N", "S"]
        elif ew_e and not ns_e: directions = ["E", "W"]
        else: directions = ["N", "S"] if self.current_phase == 0 else ["E", "W"]

        for d in directions:
            for _ in range(min(k, len(self.lanes[d]))):
                moved.append((d, self.lanes[d].popleft()))
        return moved

    # THIS IS THE MISSING METHOD
    def update_waiting(self):
        for direction in ["N", "S", "E", "W"]:
            for vehicle in self.lanes[direction]:
                vehicle.wait_time += 1
                vehicle.fuel_consumed += 0.1