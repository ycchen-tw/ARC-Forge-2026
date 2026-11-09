"""Budgets shared by the RL orchestrator and prepare subprocess (one definition for both sides)."""

import math

DEFAULT_ROUTING_GB = 60.0
DEFAULT_RESERVE_GB = 50.0
#: SPEC §5: the largest fraction of episodes one optimizer step may drop as routing D_cross (positions forwarded twice
#: under two weight versions; only episodes whose server ran --session-exact-resume). prepare_rl's
#: --max-d-cross-fraction default; the orchestrator's group_ready gate uses the same value when the config has no
#: prepare.max_d_cross_fraction
MAX_D_CROSS_FRACTION = 0.02


def required_free_bytes(max_gb: float = DEFAULT_ROUTING_GB,
                        reserve_gb: float = DEFAULT_RESERVE_GB) -> int:
    """Routing reservation plus workspace/export safety margin, in decimal GB."""
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) and v > 0 for v in (max_gb, reserve_gb)):
        raise ValueError("max_gb and reserve_gb must be positive finite decimal GB")
    return math.ceil((max_gb + reserve_gb) * 1e9)
