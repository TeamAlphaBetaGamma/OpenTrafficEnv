import os
import json
import logging
from openai import OpenAI
from models import IntersectionState, TrafficAction

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_GREEN_TIME: int = 3          # Minimum steps before a phase change is allowed
FAIRNESS_THRESHOLD: float = 20.0 # If any vehicle waits this many steps, force phase change

logger = logging.getLogger(__name__)


def _env_flag_true(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


# ── OpenAI client (initialized from env vars) ─────────────────────────────────
def _get_openai_client() -> OpenAI:
    """Build the OpenAI client using env variables."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")
    return OpenAI(
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
    )

MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# ── LLM reasoning layer ───────────────────────────────────────────────────────
def _llm_decide(state: IntersectionState, client: OpenAI) -> int | None:
    """
    Ask the LLM to recommend a traffic phase.
    """
    state_summary = {
        "intersection_id": state.intersection_id,
        "north_queue": state.north.queue_length,
        "south_queue": state.south.queue_length,
        "east_queue": state.east.queue_length,
        "west_queue": state.west.queue_length,
        "north_wait": state.north.cumulative_wait,
        "south_wait": state.south.cumulative_wait,
        "east_wait": state.east.cumulative_wait,
        "west_wait": state.west.cumulative_wait,
        "current_phase": state.current_phase,
        "phase_duration": state.phase_duration,
        "max_wait": state.max_wait,
        "upstream_hint": state.upstream_hint,
    }

    prompt = f"""You are a traffic signal controller AI.

Current intersection state (JSON):
{json.dumps(state_summary, indent=2)}

Phase 0 = North-South GREEN (vehicles from north and south can move).
Phase 1 = East-West GREEN (vehicles from east and west can move).

Based on queue lengths and wait times, which phase should be active next?
Respond with ONLY a JSON object: {{"phase": 0}} or {{"phase": 1}}.
Do not include any other text."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        text = response.choices[0].message.content.strip()
        parsed = json.loads(text)
        phase = int(parsed["phase"])
        if phase not in (0, 1):
            return None
        return phase
    except Exception as e:
        logger.warning(f"LLM call failed for intersection {state.intersection_id}: {e}")
        return None


# ── Greedy fallback ───────────────────────────────────────────────────────────
def _greedy_decide(state: IntersectionState) -> int:
    """Pick the phase with the larger total queue length."""
    ns_total = state.ns_queue()
    ew_total = state.ew_queue()
    return 0 if ns_total >= ew_total else 1


# ── Main policy function ──────────────────────────────────────────────────────
def decide_phase(state: IntersectionState, client: OpenAI | None = None) -> TrafficAction:
    """
    Apply the hybrid decision waterfall to decide a traffic phase.
    """
    llm_disabled = _env_flag_true("DISABLE_LLM")
    if client is None and not llm_disabled:
        client = _get_openai_client()

    current = state.current_phase
    other = 1 - current

    # ── Rule 1: MIN_GREEN_TIME lock ──────────────────────────────────────────
    # If the current phase has been active for fewer than MIN_GREEN_TIME steps,
    # we MUST keep it. This prevents signal flickering.

    if state.phase_duration < MIN_GREEN_TIME:
        return TrafficAction(intersection_id=state.intersection_id, phase=current)

    # ── Rule 2: Emergency override ───────────────────────────────────────────
    # If the OTHER direction has an emergency vehicle, switch to unblock it.

    if state.has_emergency(other):
        return TrafficAction(intersection_id=state.intersection_id, phase=other)

    # ── Rule 3: Fairness override ─────────────────────────────────────────────
    # If any vehicle has been waiting too long, force a switch.

    if state.max_wait > FAIRNESS_THRESHOLD:
        # Find which phase direction has the highest wait time
        ns_max = max(state.north.cumulative_wait, state.south.cumulative_wait)
        ew_max = max(state.east.cumulative_wait, state.west.cumulative_wait)
        forced_phase = 0 if ns_max >= ew_max else 1
        return TrafficAction(intersection_id=state.intersection_id, phase=forced_phase)

    # ── Rule 4: LLM reasoning ─────────────────────────────────────────────────
    if not llm_disabled:
        llm_phase = _llm_decide(state, client)
        if llm_phase is not None:
            return TrafficAction(intersection_id=state.intersection_id, phase=llm_phase)

    # ── Rule 5: Greedy fallback ───────────────────────────────────────────────
    greedy_phase = _greedy_decide(state)
    return TrafficAction(intersection_id=state.intersection_id, phase=greedy_phase)


def decide_all_phases(
    intersections: list[IntersectionState],
    client: OpenAI | None = None
) -> list[TrafficAction]:
    """
    Decide phases for all intersections in the grid.
    """
    if client is None:
        client = _get_openai_client()
    return [decide_phase(state, client) for state in intersections]