"""ARC3 RL v8 reward: the first context window of a level episode (docs/plans/arc3_rl_v8_window_note_judge_20261029.md §3).

An episode runs from a level seed to the first window change (harness ``stop_after_first_window``, stop ``window``).

- A clear is scored by ``clear_reward``: the official level score ``L = min(115, 100 (b / a)^2)`` (b = baseline actions,
  a = actions spent, RESETs included) mapped to ``0.9 + 0.1 (L - 100) / 15`` while ``a <= b``, and an exponential penalty
  ``0.9 exp(-5 (a / b - 1))`` past the baseline.
- An uncleared episode that reached the window change at the context threshold (stop ``window``, ``window_cause``
  ``threshold``) hands its note to the judge (``work/qwen38fn_rl/arc3_note_judge.py``). The judge grades the note and
  estimates the further actions a successor needs to clear from the handoff state (``remaining_actions``); the grade's
  value ``TIERS`` is capped by a clear at the estimated total ``actions + remaining_actions`` (``note_reward``). The judge
  writes one ``judge.json`` per group; ``judge_rewards`` reads it back and re-derives every reward from the grade.
- A window change the model asked for before the threshold (``new_context`` before the hook's ``.context_full``;
  ``window_cause`` ``early``) is not a valid handoff: R = 0, not judged (user 10-29).
- Every other stop (action cap, budget, died, ...) is 0.

prepare_rl ``--arc3-reward note`` and the orchestrator's group_live / judge step share this one definition. Standard
library only: the orchestrator runs on the system python3 and the judge in its own venv.
"""

from __future__ import annotations

import math

#: grade -> reward before the cap (§4)
TIERS = {'S': 0.9, 'A': 0.8, 'B': 0.55, 'C': 0.25, 'D': 0.1, 'F': 0.0}
ORDER = 'SABCDF'                      # best first
DECAY = 5.0                           # §3: R = 0.9 exp(-DECAY (r - 1)) past baseline
#: the harness stop of an episode that ended at its first window change (run_arc3_level, level_episode.WINDOW)
WINDOW_STOP = 'window'
#: the judge's call structure (work/qwen38fn_rl/arc3_note_judge.py): at most this many passes per group (two shuffled
#: passes, a third when they disagree by more than one grade), each pass retried this many times when its reply leaves out
#: or repeats a note; one api call is the SDK request plus API_MAX_RETRIES retries, one codex call is one ``codex exec``
JUDGE_MAX_PASSES = 3
PASS_ATTEMPTS = 3
API_MAX_RETRIES = 2
#: api: a call refused as rate-limited / overloaded (HTTP 429 / 529, after the SDK's own retries) is retried with backoff
#: outside the attempt arithmetic, up to this many seconds per judge process (the refused calls' time and the sleeps)
API_OVERLOAD_WAIT_SECONDS = 1800.0
#: judge_attempt_seconds' margin for the SDK's retry sleeps, the process start-up and the kill grace: x(1 + fraction) + s
ATTEMPT_MARGIN_FRACTION = 0.1
ATTEMPT_MARGIN_SECONDS = 600.0


def judge_attempt_seconds(backend, call_timeout):
    """The longest one judge process can take when every call runs into ``call_timeout`` (seconds per backend request):
    passes x pass retries x one call (api: the request plus API_MAX_RETRIES retries; codex: one ``codex exec``); api adds
    the overload budget API_OVERLOAD_WAIT_SECONDS plus one call that may start just inside it. Then the margin:
    x (1 + ATTEMPT_MARGIN_FRACTION) + ATTEMPT_MARGIN_SECONDS. The orchestrator's per-attempt limit must cover it."""
    if backend not in ('api', 'codex', 'ensemble', 'claude-code'):
        raise ValueError(f'judge backend {backend!r} is not api, codex, ensemble or claude-code')
    if not (isinstance(call_timeout, (int, float)) and not isinstance(call_timeout, bool) and math.isfinite(call_timeout)
            and call_timeout > 0):
        raise ValueError(f'judge call timeout {call_timeout!r} must be a finite number > 0')
    if backend == 'ensemble':          # v8b: the judges of a round run in parallel, one pass each; at most V8B_MAX_ROUNDS
        worst = V8B_MAX_ROUNDS * PASS_ATTEMPTS * call_timeout
        return round(worst * (1 + ATTEMPT_MARGIN_FRACTION) + ATTEMPT_MARGIN_SECONDS, 3)
    if backend == 'claude-code':       # v8b, Claude Code alone: one pass, one round
        return round(PASS_ATTEMPTS * call_timeout * (1 + ATTEMPT_MARGIN_FRACTION) + ATTEMPT_MARGIN_SECONDS, 3)
    call = (API_MAX_RETRIES + 1 if backend == 'api' else 1) * call_timeout
    worst = JUDGE_MAX_PASSES * PASS_ATTEMPTS * call
    if backend == 'api':
        worst += API_OVERLOAD_WAIT_SECONDS + call
    return round(worst * (1 + ATTEMPT_MARGIN_FRACTION) + ATTEMPT_MARGIN_SECONDS, 3)


def combine_grades(grades):
    """§4, a note's grade from its passes in pass order: two passes within one grade -> the lower; two further apart ->
    None (a third pass is needed). Three passes (the third runs for the whole group when any note needs it): the first two
    combined when they are within one grade, else the median of the three."""
    if len(grades) not in (2, 3):
        raise ValueError(f'combine_grades takes 2 or 3 passes, not {len(grades)}')
    first = sorted(ORDER.index(g) for g in grades[:2])
    if first[1] - first[0] <= 1:
        return ORDER[first[1]]
    if len(grades) == 2:
        return None
    return ORDER[sorted(ORDER.index(g) for g in grades)[1]]


def median_remaining(values):
    """The note's remaining_actions from its passes: the median (two passes: the larger)."""
    if not values or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
        raise ValueError(f'remaining_actions of the passes {values!r}: want ints >= 0')
    return sorted(values)[len(values) // 2]


#: why the window changed (run_arc3_level.window_cause): the context had reached the hook's change point (the model called
#: new_context after the reminder, or the change was forced), or the model called new_context before it
WINDOW_CAUSES = ('threshold', 'early')


def level_score(actions, baseline):
    """Official level score min(115, 100 (b/a)^2)."""
    if actions <= 0 or baseline <= 0:
        raise ValueError(f'actions {actions}, baseline {baseline}: both must be positive')
    return min(115.0, 100.0 * (baseline / actions) ** 2)


def clear_reward(actions, baseline):
    """§3: r <= 1 maps L in [100, 115] to [0.9, 1.0]; r > 1 decays 0.9 exp(-5 (r - 1))."""
    r = actions / baseline
    if r <= 1:
        return 0.9 + 0.1 * (level_score(actions, baseline) - 100.0) / 15.0
    return 0.9 * math.exp(-DECAY * (r - 1.0))


def _count(name, v):
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise ValueError(f'arc3 note: {name} {v!r} is not an int >= 0')


def note_reward(tier, actions, remaining_actions, baseline):
    """§3 (10-29 revision): the tier's value, capped by a clear at the estimated total ``actions + remaining_actions``
    (the actions this window spent plus the judge's estimate of the further actions to clear from the handoff state).
    A total of 0 has the full cap 1."""
    _count('actions', actions)
    _count('remaining_actions', remaining_actions)
    total = actions + remaining_actions
    cap = 1.0 if total == 0 else clear_reward(total, baseline)
    return min(TIERS[tier], cap)


def judged_episode(stop, window_cause):
    """Does this episode's note go to the judge: stop ``window`` with ``window_cause`` threshold. ``window_cause`` is given
    (one of WINDOW_CAUSES) exactly for the window episodes."""
    if (window_cause is not None) != (stop == WINDOW_STOP):
        raise ValueError(f'arc3 note: stop {stop!r} with window_cause {window_cause!r} (a window episode has exactly one)')
    if window_cause is not None and window_cause not in WINDOW_CAUSES:
        raise ValueError(f'arc3 note: window_cause {window_cause!r} is not one of {WINDOW_CAUSES}')
    return window_cause == 'threshold'


def episode_reward(stop, correct, actions, baseline, window_cause, judged):
    """One episode's R: a clear -> ``clear_reward(actions, baseline)``; stop ``window`` at the threshold -> ``judged`` (the
    judge's reward for its note, which must be given); an early window change and every other stop -> 0. ``judged`` is
    given exactly for the judged episodes (``judged_episode``)."""
    if not isinstance(correct, bool) or correct != (stop == 'cleared'):
        raise ValueError(f'arc3 note: correct {correct!r} with stop {stop!r} (a clear is exactly stop cleared)')
    _count('actions', actions)
    _count('baseline', baseline)
    if baseline < 1:
        raise ValueError(f'arc3 note: baseline {baseline} < 1')
    judge = judged_episode(stop, window_cause)
    if (judged is not None) != judge:
        raise ValueError(f'arc3 note: stop {stop!r} window_cause {window_cause!r} with judged reward {judged!r} (exactly the '
                         'window episodes changed at the threshold are judged)')
    if correct:
        if actions < 1:
            raise ValueError(f'arc3 note: a clear with {actions} actions')
        return clear_reward(actions, baseline)
    if judge:
        if not (isinstance(judged, (int, float)) and not isinstance(judged, bool) and 0.0 <= judged <= max(TIERS.values())):
            raise ValueError(f'arc3 note: judged reward {judged!r} outside [0, {max(TIERS.values())}]')
        return float(judged)
    return 0.0


# ---------------------------------------------------------------------------------------------------------------------
# v8b (docs/plans/arc3_rl_v8b_judge_ensemble_20261101.md): every judge pass gives each note an absolute grade and puts it
# in a bucket of the group ranking (buckets best first, one grade per bucket, grades never better further down; a
# flexible quota of about n/3..n/2 buckets). A pass's score of a note, by ``mapping``:
#   tier       TIERS[grade] (v8's values, the ranking unused);
#   tier+rank  TIERS[grade] - p * RANK_SPREAD * (TIERS[grade] - TIER_LOW[grade]), p = the bucket's place among the
#              buckets of that grade (0 best .. 1 worst; a grade with one bucket keeps its value): the ranking orders
#              notes inside a grade's band, never across grades, and no note goes above its grade's value;
#   band       TIER_HIGH[grade] - (0.05 + 0.9 p) * (TIER_HIGH[grade] - TIER_LOW[grade]): the ranking spread over the
#              grade's whole band, F included (a grade with one bucket keeps its value); 11-01 dev: AUC as rank, above
#              tier+rank, and it keeps every grade inside its band;
#   rank       0.9 * (1 - q), q = the bucket's mid-rank percentile in the whole ranking (0 best .. 1 worst).
# A note's score is the mean over its passes (every judge, every round), its remaining_actions the median
# (median_remaining), and its reward min(score, the clear reward at actions + remaining_actions) as note_reward.

V8B_VERSION = 'v8b'
V8B_MAPPINGS = ('tier', 'tier+rank', 'band', 'rank')
#: a grade's band runs down to the midpoint to the next lower grade's value (F's is 0)
TIER_LOW = {'S': 0.85, 'A': 0.675, 'B': 0.4, 'C': 0.175, 'D': 0.05, 'F': 0.0}
#: ... and up to the midpoint to the next upper grade's value (S's is its value: no note above a clear at the baseline)
TIER_HIGH = {'S': 0.9, 'A': 0.85, 'B': 0.675, 'C': 0.4, 'D': 0.175, 'F': 0.05}
RANK_SPREAD = 0.8
#: rounds of the ensemble: a second round when some note's round-1 grades are more than one grade apart
V8B_MAX_ROUNDS = 2


def pass_score(q, mapping):
    """One judge pass's score of a note: ``q`` has grade, bucket (its index), buckets (the bucket sizes) and
    bucket_grades (each bucket's grade), as arc3_note_judge.judge_group_v8b writes them; checked for consistency."""
    if mapping not in V8B_MAPPINGS:
        raise ValueError(f'v8b mapping {mapping!r} is not one of {V8B_MAPPINGS}')
    sizes, grades, j, g = q['buckets'], q['bucket_grades'], q['bucket'], q['grade']
    if len(sizes) != len(grades) or not 0 <= j < len(sizes) or grades[j] != g or min(sizes) < 1:
        raise ValueError(f'v8b pass: bucket {j} of sizes {sizes} / grades {grades} with grade {g}')
    if [ORDER.index(x) for x in grades] != sorted(ORDER.index(x) for x in grades):
        raise ValueError(f'v8b pass: bucket grades {grades} get better further down')
    if mapping == 'tier':
        return TIERS[g]
    if mapping in ('tier+rank', 'band'):
        same = [k for k, x in enumerate(grades) if x == g]
        if len(same) == 1:
            return TIERS[g]
        p = same.index(j) / (len(same) - 1)
        if mapping == 'band':
            return TIER_HIGH[g] - (0.05 + 0.9 * p) * (TIER_HIGH[g] - TIER_LOW[g])
        return TIERS[g] - p * RANK_SPREAD * (TIERS[g] - TIER_LOW[g])
    n = sum(sizes)
    q_pct = 0.5 if n == 1 else (sum(sizes[:j]) + (sizes[j] - 1) / 2) / (n - 1)
    return 0.9 * (1 - q_pct)


def v8b_note(passes, actions, baseline, mapping):
    """A v8b note from its passes -> (score, remaining_actions, reward)."""
    if not passes:
        raise ValueError('a v8b note needs at least one pass')
    _count('actions', actions)
    score = sum(pass_score(q, mapping) for q in passes) / len(passes)
    rem = median_remaining([q['remaining_actions'] for q in passes])
    total = actions + rem
    cap = 1.0 if total == 0 else clear_reward(total, baseline)
    return score, rem, min(score, cap)


def judge_rewards_v8b(doc):
    """judge_rewards for a v8b judge.json: every pass of every note re-scored with the file's ``mapping``; the stored
    score / remaining_actions / reward must equal the re-derived ones; every judge of ``judges`` must have a round-1
    pass on every note, and round 2 exists for every judge or none."""
    mapping = doc.get('mapping')
    if mapping not in V8B_MAPPINGS:
        raise ValueError(f'v8b judge.json mapping {mapping!r} is not one of {V8B_MAPPINGS}')
    judges = [j['backend'] for j in doc['judges']]
    out = {}
    for n in doc['notes']:
        if n['id'] in out:
            raise ValueError(f'judge.json grades note {n["id"]!r} twice')
        rounds = sorted({(q['round'], q['backend']) for q in n['passes']})
        if len(rounds) != len(n['passes']) or rounds not in ([(1, b) for b in sorted(judges)],
                                                            [(r, b) for r in (1, 2) for b in sorted(judges)]):
            raise ValueError(f'judge.json: note {n["id"]!r} has passes {rounds}, its judges are {judges}')
        score, rem, reward = v8b_note(n['passes'], n['actions'], doc['baseline'], mapping)
        if (n['score'], n['remaining_actions'], n['reward']) != (score, rem, reward):
            raise ValueError(f'judge.json: note {n["id"]!r} stores score / remaining / reward {n["score"]} / '
                             f'{n["remaining_actions"]} / {n["reward"]}, its passes give {score} / {rem} / {reward}')
        out[n['id']] = {'grade': None, 'score': score, 'remaining_actions': rem, 'reward': reward,
                        'actions': n['actions'], 'resets': n['resets']}
    return out


def judge_rewards(doc, game, level, baseline):
    """A group's ``judge.json`` (arc3_note_judge.judge_group's output) -> {note id: {grade, remaining_actions, reward,
    actions, resets}}. The game / level / baseline must be the group's; every note's grade must be ``combine_grades`` of
    its passes' grades and its remaining_actions ``median_remaining`` of theirs, and its reward is re-derived from grade /
    actions / remaining_actions with ``note_reward`` and must equal the stored one (the file is not trusted for the
    numbers). A v8b file (``version`` v8b) goes to judge_rewards_v8b."""
    for key in ('game', 'level', 'baseline', 'notes'):
        if key not in doc:
            raise ValueError(f'judge.json has no {key!r}')
    if (doc['game'], doc['level'], doc['baseline']) != (game, level, baseline):
        raise ValueError(f'judge.json is for {doc["game"]} level {doc["level"]} baseline {doc["baseline"]}, the group is '
                         f'{game} level {level} baseline {baseline}')
    if doc.get('version') == V8B_VERSION:
        return judge_rewards_v8b(doc)
    if 'version' in doc:
        raise ValueError(f'judge.json version {doc["version"]!r}: only {V8B_VERSION!r} (or none, v8) is known')
    out = {}
    for n in doc['notes']:
        if n['id'] in out:
            raise ValueError(f'judge.json grades note {n["id"]!r} twice')
        if n['grade'] not in TIERS:
            raise ValueError(f'judge.json: note {n["id"]!r} has grade {n["grade"]!r}, not one of {ORDER}')
        passes = n['passes']
        if not 2 <= len(passes) <= JUDGE_MAX_PASSES:
            raise ValueError(f'judge.json: note {n["id"]!r} has {len(passes)} passes, want 2 or 3')
        combined = combine_grades([q['grade'] for q in passes])
        if combined != n['grade']:
            raise ValueError(f'judge.json: note {n["id"]!r} grade {n["grade"]}, its passes {[q["grade"] for q in passes]} '
                             f'combine to {combined}')
        median = median_remaining([q['remaining_actions'] for q in passes])
        if median != n['remaining_actions']:
            raise ValueError(f'judge.json: note {n["id"]!r} remaining_actions {n["remaining_actions"]}, the median of its '
                             f'passes {[q["remaining_actions"] for q in passes]} is {median}')
        reward = note_reward(n['grade'], n['actions'], n['remaining_actions'], baseline)
        if n['reward'] != reward:
            raise ValueError(f'judge.json: note {n["id"]!r} grade {n["grade"]} at {n["actions"]} + {n["remaining_actions"]} '
                             f'actions is worth {reward}, the file says {n["reward"]}')
        out[n['id']] = {'grade': n['grade'], 'remaining_actions': n['remaining_actions'], 'reward': reward,
                        'actions': n['actions'], 'resets': n['resets']}
    return out
