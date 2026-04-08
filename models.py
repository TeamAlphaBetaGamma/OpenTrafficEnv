from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ── Lane-level state ──────────────────────────────────────────────────────────

class LaneState(BaseModel):
    """State of a single lane (North, South, East, or West) at one intersection."""
    queue_length: int = Field(0, ge=0, description="Number of vehicles waiting")
    cumulative_wait: float = Field(0.0, ge=0.0, description="Total wait time across all vehicles in lane (steps)")
    emergency_flag: bool = Field(False, description="True if an emergency vehicle is present in this lane")
    fuel_consumed: float = Field(0.0, ge=0.0, description="Total fuel consumed by vehicles in this lane")


# ── Intersection-level state ──────────────────────────────────────────────────

class IntersectionState(BaseModel):
    """Full state of one intersection (one agent in the multi-agent system)."""
    intersection_id: int = Field(..., description="Unique ID of this intersection in the grid")
    north: LaneState = Field(default_factory=LaneState)
    south: LaneState = Field(default_factory=LaneState)
    east: LaneState = Field(default_factory=LaneState)
    west: LaneState = Field(default_factory=LaneState)
    current_phase: int = Field(0, ge=0, le=1, description="0=NS green, 1=EW green")
    phase_duration: int = Field(0, ge=0, description="How many steps the current phase has been active")
    max_wait: float = Field(0.0, ge=0.0, description="Maximum wait time of any single vehicle across all lanes")
    upstream_hint: Optional[int] = Field(None, description="Suggested phase from an upstream neighbor (None if no hint)")

    def ns_queue(self) -> int:
        """Total queue length for North-South direction."""
        return self.north.queue_length + self.south.queue_length

    def ew_queue(self) -> int:
        """Total queue length for East-West direction."""
        return self.east.queue_length + self.west.queue_length

    def has_emergency(self, phase: int) -> bool:
        """Check if the given phase direction has an emergency vehicle."""
        if phase == 0:  # NS
            return self.north.emergency_flag or self.south.emergency_flag
        else:  # EW
            return self.east.emergency_flag or self.west.emergency_flag


# ── Global state (entire grid) ────────────────────────────────────────────────

class GlobalState(BaseModel):
    """State of the entire traffic grid at one time step."""
    intersections: List[IntersectionState] = Field(default_factory=list)
    step_number: int = Field(0, ge=0)
    task_id: int = Field(1, ge=1, le=3, description="1=Easy, 2=Medium, 3=Hard")
    done: bool = Field(False, description="True if the episode is over")


# ── Action ────────────────────────────────────────────────────────────────────

class TrafficAction(BaseModel):
    """Action taken by one agent (one intersection)."""
    intersection_id: int = Field(..., description="Which intersection this action is for")
    phase: int = Field(..., ge=0, le=1, description="0=NS green, 1=EW green")


# ── Step result (what the environment returns after each step) ─────────────────

class StepInfo(BaseModel):
    """Supplementary info returned alongside each step result."""
    cars_passed: int = Field(0, ge=0)
    total_cars: int = Field(0, ge=0)
    total_wait: float = Field(0.0, ge=0.0)
    total_fuel: float = Field(0.0, ge=0.0)
    max_wait: float = Field(0.0, ge=0.0)
    emergency_delay: float = Field(0.0, ge=0.0, description="Steps that emergency vehicles were blocked")


class StepResult(BaseModel):
    """Complete result returned by environment.step()."""
    observation: GlobalState
    reward: float = Field(0.0, ge=0.0, le=1.0, description="Normalized reward in [0.0, 1.0]")
    done: bool = False
    info: StepInfo = Field(default_factory=StepInfo)


# ── Reset request/response ────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = Field(1, ge=1, le=3)
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")


class ResetResponse(BaseModel):
    observation: GlobalState
    task_id: int
    message: str = "Environment reset successfully"