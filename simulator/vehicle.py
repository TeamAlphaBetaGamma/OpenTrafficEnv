class Vehicle:
    def __init__(self, is_emergency=False):
        self.wait_time = 0
        self.is_emergency = is_emergency
        self.fuel_consumed = 0.0