"""ARC3 single-level RL reward on the official level score (prepare_rl ``--arc3-reward official`` and the orchestrator's
group_live / retire rule share this one definition).

The official level score is ``S = min(100 (baseline / actions)^2, 115)`` (``arcforge.arc_code.score.level_score``,
verbatim from the official scorecard). The reward of an episode is ``R = c * f(S)``, c = cleared, with ``f`` piecewise
linear through three points and flat outside them::

    S >= top_score            -> 1
    S = knee_score            -> knee_reward
    S <= zero_score           -> 0

v3.3 (10-22, user): top 115 (the cap: actions <= 0.933 x baseline), knee 100 -> 0.7 (actions = baseline), zero 75
(actions = 1.155 x baseline): more than ~15% over the human baseline is a failure. No token term: running out of the
output budget already fails the episode.
"""

from __future__ import annotations

import math

from arcforge.arc_code.score import level_score

#: the official cap of the level score (scorecard.py; level_score)
LEVEL_SCORE_CAP = 115.0
#: prepare_rl ``--arc3-<name>`` / orchestrator config ``arc3_<name>``: all four are required, no defaults
PARAMS = ("top_score", "knee_score", "knee_reward", "zero_score")


def check_params(*, top_score: float, knee_score: float, knee_reward: float, zero_score: float) -> None:
    """``0 <= zero_score < knee_score < top_score <= 115`` and ``0 < knee_reward < 1``, all finite numbers; raise otherwise."""
    vals = {"top_score": top_score, "knee_score": knee_score, "knee_reward": knee_reward, "zero_score": zero_score}
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals.values()):
        raise ValueError(f"arc3 official: parameters must be finite numbers: {vals}")
    if not (0 <= zero_score < knee_score < top_score <= LEVEL_SCORE_CAP and 0 < knee_reward < 1):
        raise ValueError(f"arc3 official: want 0 <= zero_score < knee_score < top_score <= {LEVEL_SCORE_CAP:g} and "
                         f"0 < knee_reward < 1: {vals}")


def score_reward(score: float, *, top_score: float, knee_score: float, knee_reward: float, zero_score: float) -> float:
    """``f(S)``: the reward of a cleared episode whose official level score is ``score``."""
    if not (isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= LEVEL_SCORE_CAP):
        raise ValueError(f"arc3 official: level score {score!r} outside [0, {LEVEL_SCORE_CAP:g}]")
    if score >= top_score:
        return 1.0
    if score >= knee_score:
        return knee_reward + (1.0 - knee_reward) * (score - knee_score) / (top_score - knee_score)
    if score > zero_score:
        return knee_reward * (score - zero_score) / (knee_score - zero_score)
    return 0.0


def episode_reward(cleared: bool, actions: int, baseline_actions: int, **params: float) -> float:
    """``R = c * f(level_score(actions, baseline))``. A cleared episode has >= 1 action and a baseline >= 1; an uncleared
    one is 0 whatever its actions."""
    if not isinstance(cleared, bool):
        raise ValueError(f"arc3 official: cleared {cleared!r} is not a bool")
    for name, v in (("actions", actions), ("baseline_actions", baseline_actions)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(f"arc3 official: {name} {v!r} is not an int >= 0")
    if baseline_actions < 1:
        raise ValueError(f"arc3 official: baseline_actions {baseline_actions} < 1")
    if not cleared:
        return 0.0
    if actions < 1:
        raise ValueError(f"arc3 official: a cleared episode with {actions} actions")
    return score_reward(level_score(actions, baseline_actions), **params)
