from collections import deque

class Intersection:
    def __init__(self):
        self.lanes = {
            "N": deque(),
            "S": deque(),
            "E": deque(),
            "W": deque()
        }
        self.current_phase = 0  # 0: NS, 1: EW
        self.phase_duration = 0

    def add_vehicle(self, direction, vehicle):
        self.lanes[direction].append(vehicle)

    def process_signal(self, k=2):
        moved = []

        if self.current_phase == 0:
            directions = ["N", "S"]
        else:
            directions = ["E", "W"]

        for d in directions:
            for _ in range(min(k, len(self.lanes[d]))):
                moved.append(self.lanes[d].popleft())

        return moved

    def update_waiting(self):
        for lane in self.lanes.values():
            for v in lane:
                v.wait_time += 1
                v.fuel_consumed += 0.1