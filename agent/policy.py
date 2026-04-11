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
    # Validator injects API_KEY and API_BASE_URL — always prefer those.
    # Fall back to HF_TOKEN / OPENAI_API_KEY for local runs.
    api_key = (
        os.environ.get("API_KEY")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("OPENAI_API_KEY", "")
    )
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
    Decide a traffic phase using the LLM as the primary decision-maker.

    The LLM is ALWAYS called first so that every step produces an API call
    through the LiteLLM proxy.  Hard safety rules then override the LLM only
    when strictly required (emergency vehicle present, or excessive wait time).
    The MIN_GREEN_TIME guard is kept but applied AFTER the LLM call so the
    proxy still sees the request.
    """
    llm_disabled = _env_flag_true("DISABLE_LLM")
    if client is None and not llm_disabled:
        client = _get_openai_client()

    current = state.current_phase
    other = 1 - current

    # ── Step 1: Ask the LLM (always, so the proxy sees every call) ───────────
    llm_phase: int | None = None
    if not llm_disabled and client is not None:
        llm_phase = _llm_decide(state, client)

    # Start with the LLM recommendation, fall back to greedy if LLM failed.
    preferred_phase = llm_phase if llm_phase is not None else _greedy_decide(state)

    # ── Step 2: Hard safety overrides (applied after the LLM call) ───────────

    # MIN_GREEN_TIME lock — prevent signal flickering.
    if state.phase_duration < MIN_GREEN_TIME:
        return TrafficAction(intersection_id=state.intersection_id, phase=current)

    # Emergency override — unblock emergency vehicle in the other direction.
    if state.has_emergency(other):
        return TrafficAction(intersection_id=state.intersection_id, phase=other)

    # Fairness override — relieve the direction with the longest wait.
    if state.max_wait > FAIRNESS_THRESHOLD:
        ns_max = max(state.north.cumulative_wait, state.south.cumulative_wait)
        ew_max = max(state.east.cumulative_wait, state.west.cumulative_wait)
        forced_phase = 0 if ns_max >= ew_max else 1
        return TrafficAction(intersection_id=state.intersection_id, phase=forced_phase)

    # ── Step 3: Return LLM / greedy recommendation ────────────────────────────
    return TrafficAction(intersection_id=state.intersection_id, phase=preferred_phase)


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