from __future__ import annotations

import os
from typing import Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from models import (
    GlobalState,
    IntersectionState,
    LaneState,
    ResetRequest,
    ResetResponse,
    StepInfo,
    StepResult,
    TrafficAction,
)
from simulator.env import TrafficSimulator

app = FastAPI(title="OpenTrafficEnv", version="0.1.0")


class StepRequest(BaseModel):
    actions: list[TrafficAction] = Field(default_factory=list)


@app.get("/")
def root() -> dict:
    return {
        "name": "OpenTrafficEnv",
        "status": "ok",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ping")
def ping() -> dict:
    return {"status": "ok"}


_env: Optional[TrafficSimulator] = None
_task_id: int = 1


def _task_id_to_name(task_id: int) -> str:
    return {1: "easy", 2: "medium", 3: "hard"}.get(task_id, "easy")


def _max_steps_for_task(task_id: int) -> int:
    # Keep in sync with inference.py TASK_CONFIG
    return {1: 25, 2: 35, 3: 45}.get(task_id, 25)


def _require_env() -> TrafficSimulator:
    if _env is None:
        raise HTTPException(status_code=409, detail="Environment not initialized. Call POST /reset first.")
    return _env


def _build_global_state(env: TrafficSimulator, task_id: int) -> GlobalState:
    intersections: list[IntersectionState] = []
    intersection_index = 0

    for row in range(env.size):
        for col in range(env.size):
            inter = env.grid[row][col]

            def make_lane(direction: str) -> LaneState:
                lane_cars = inter.lanes[direction]
                cum_wait = sum(v.wait_time for v in lane_cars)
                cum_fuel = sum(v.fuel_consumed for v in lane_cars)
                has_emergency = any(v.is_emergency for v in lane_cars)
                return LaneState(
                    queue_length=len(lane_cars),
                    cumulative_wait=float(cum_wait),
                    emergency_flag=has_emergency,
                    fuel_consumed=float(cum_fuel),
                )

            max_wait = 0.0
            for lane in inter.lanes.values():
                for vehicle in lane:
                    max_wait = max(max_wait, float(vehicle.wait_time))

            intersections.append(
                IntersectionState(
                    intersection_id=intersection_index,
                    north=make_lane("N"),
                    south=make_lane("S"),
                    east=make_lane("E"),
                    west=make_lane("W"),
                    current_phase=inter.current_phase,
                    phase_duration=inter.phase_duration,
                    max_wait=float(max_wait),
                    upstream_hint=None,
                )
            )
            intersection_index += 1

    return GlobalState(
        intersections=intersections,
        step_number=env.step_count,
        task_id=task_id,
        done=env.step_count >= env.max_steps,
    )


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = Body(default_factory=ResetRequest)) -> ResetResponse:
    global _env, _task_id

    _task_id = req.task_id
    task_name = _task_id_to_name(req.task_id)
    seed = req.seed if req.seed is not None else 42

    _env = TrafficSimulator(task=task_name, seed_val=seed)
    _env.reset(max_steps=_max_steps_for_task(req.task_id))

    obs = _build_global_state(_env, _task_id)
    return ResetResponse(observation=obs, task_id=_task_id)


@app.post("/step", response_model=StepResult)
def step(req: StepRequest = Body(default_factory=StepRequest)) -> StepResult:
    env = _require_env()

    # Convert actions list -> simulator dict ("row,col" -> phase)
    actions_map: dict[str, int] = {}
    for action in req.actions:
        row = action.intersection_id // env.size
        col = action.intersection_id % env.size
        actions_map[f"{row},{col}"] = action.phase

    _, reward, done, info_dict = env.step(actions_map)
    obs = _build_global_state(env, _task_id)
    info = StepInfo(**info_dict)
    return StepResult(observation=obs, reward=float(reward), done=bool(done), info=info)


@app.get("/state", response_model=GlobalState)
def state() -> GlobalState:
    env = _require_env()
    return _build_global_state(env, _task_id)


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)
