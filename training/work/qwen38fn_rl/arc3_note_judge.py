"""ARC3 RL v8 note judge (docs/plans/arc3_rl_v8_window_note_judge_20261029.md §3-§4).

An episode runs from a level seed to the first window change. A clear is scored by ``clear_reward`` (the official level
score, exponential penalty past baseline). An uncleared episode's handoff note is graded by an LLM judge that reads the
game's rules and sees every uncleared note of the group at once (with each note's actions used); the grade maps to
``TIERS`` and is capped by ``clear_reward`` at the same action count.

Library + CLI. The CLI grades one group read from a JSON file and writes the result JSON:
    python arc3_note_judge.py <group.json> <out.json> [--model claude-opus-5-5] [--effort high] [--seed N]
    python arc3_note_judge.py <group.json> <out.json> --backend codex [--codex-model gpt-6-astra] [--codex-effort xhigh]
group.json: {"game": "sk48", "level": 5, "baseline": 230, "notes": [{"id": ..., "text": ..., "actions": ..., "resets": ...}]}

Two backends, the same prompt and JSON schema:
- api (default): the Anthropic API. Needs the anthropic SDK (isolated venv $ARC_DATA/venvs/arc3-judge) and
  ANTHROPIC_API_KEY_2 in the environment (source the project .env).
- codex: OpenAI through the local ``codex exec`` (the ChatGPT login in ~/.codex, no API key): read-only sandbox, a fresh
  empty working directory next to the output file, no session kept, --ignore-user-config with an explicit 800k context
  (``CodexClient``). ``codex`` on PATH or --codex-bin.
The result records the backend, model and effort. Every failure raises: there is no default grade.
"""

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RULES = Path(__file__).resolve().parents[1] / 'arc_agi3' / 'public_rules' / 'en'
NOTE_CHARS = 30_000                   # §4: longer notes are cut, and the cut is marked (v7 handoffs: 12-16k)
# the reward formulas live in src/arcforge/arc3_note.py (prepare_rl and the orchestrator use the same ones); the judge's
# venv has no arcforge installed, so the repo's src goes on the path (arc3_note is standard library only)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from arcforge.arc3_note import (DECAY, ORDER, PASS_ATTEMPTS, TIERS, API_MAX_RETRIES,  # noqa: E402,F401
                                 API_OVERLOAD_WAIT_SECONDS, V8B_MAPPINGS, V8B_MAX_ROUNDS, V8B_VERSION, clear_reward,
                                 combine_grades, level_score, median_remaining, note_reward, v8b_note)


SYSTEM = """You grade handoff notes written by an AI agent playing one level of an ARC-AGI-3 game.

The agent played the level from its start until its context window filled up. The note is what its successor receives:
the successor is the same agent, starting with an empty context and only this handoff (its notes, its latest level
file, the server state), and continues the same level from the same game state, with the actions already spent
counted against it. The level is scored by the official rule: level score = min(115, 100 * (baseline /
total_actions)^2), where total_actions counts every action of the level including RESETs, including the actions
already spent. A clear needing more than the baseline is a poor outcome.

You have the game's full rules below (accurate; the agent never saw them). For each note answer one question: **if the
successor takes over with this note and the actions already spent, how likely is it to clear, and with how many more
actions?** Work in three steps per note, then grade:
1. confirmed: what the note gets right that the successor can act on (mechanics, this level's goal, a working plan
   or tool), checked against the rules.
2. open: what is still unresolved. A note that has narrowed the possibilities to a few and states a cheap test that
   tells them apart has made real progress; a bare list of guesses with no commitment and no test has not.
3. misleading: claims that are wrong in a way that would lead the successor to wrong actions. Errors that do not
   affect how the level is solved do not count here.
Then estimate remaining_actions: the further actions (RESETs included) the successor would most likely need to clear
from the current state by following this note (if it would likely fail, estimate what a clear would take anyway).

Grades (absolute; they do not depend on the other notes, and every note in a group may share one grade):
- S: the goal and the needed mechanics are right and the next steps are concrete and workable: a clear is very likely.
- A: the core is right; one piece is missing or the plan has a fixable error: a clear is likely.
- B: part of the key mechanics is right, but a decisive piece is missing or wrong: a clear is uncertain.
- C: only peripheral observations are right; the core goal or mechanic is wrong or unknown.
- D: almost nothing usable.
- F: misleading claims about the core of the level that would send the successor the wrong way, or nothing about
  this level at all.
The action cost is applied separately from your remaining_actions estimate, so do not lower a grade for actions
already spent; do make remaining_actions honest.

Rules for grading:
- Credit what the note commits to or has confirmed, and real narrowing of uncertainty; a list of mutually exclusive
  guesses without a deciding test earns nothing.
- Claims the agent makes about its own work (a simulator "validated", a plan "ready", "two steps from the goal") are
  the agent's statements, not facts: credit them only as far as the note's content makes them plausible against the
  rules and the server state.
- Length, confidence and polish earn nothing. A note cut at the length limit is graded on what is shown.
- You see the whole group so you can calibrate between notes; do not spread grades just to separate them.
- Notes may carry text from earlier levels of the same game; grade what matters for this level.

Return JSON only: one entry per note id with confirmed, open, misleading (short), remaining_actions, grade, and a
one-sentence reason quoting the note."""

SCHEMA = {
    'type': 'object',
    'properties': {'grades': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {'note': {'type': 'string'}, 'confirmed': {'type': 'string'}, 'open': {'type': 'string'},
                       'misleading': {'type': 'string'}, 'remaining_actions': {'type': 'integer'},
                       'grade': {'type': 'string', 'enum': list(ORDER)}, 'reason': {'type': 'string'}},
        'required': ['note', 'confirmed', 'open', 'misleading', 'remaining_actions', 'grade', 'reason'],
        'additionalProperties': False}}},
    'required': ['grades'], 'additionalProperties': False}


def rules_text(game):
    common = (RULES / 'COMMON.md').read_text()
    page = (RULES / 'games' / f'{game}.md').read_text()
    return f'# Common semantics of all games\n\n{common}\n\n# Rules of {game.upper()}\n\n{page}'


def user_text(game, level, baseline, notes, order):
    parts = [f'Game {game.upper()}, level {level}. Baseline: {baseline} actions for this level.',
             f'{len(notes)} notes follow (ids are anonymous; the order is arbitrary).']
    for label, i in order:
        n = notes[i]
        text = n['text']
        cut = len(text) > NOTE_CHARS
        if cut:
            text = text[:NOTE_CHARS]
        parts.append(f'=== note {label} === actions spent in this level: {n["actions"]} '
                     f'({n["actions"] / baseline:.2f} x baseline), of which RESET: {n["resets"]}'
                     + (f'; the note was cut at {NOTE_CHARS} characters' if cut else '')
                     + f'\n{text}\n=== end of note {label} ===')
    return '\n\n'.join(parts)


def client_from_env(timeout=900.0):
    """The Anthropic client: ``timeout`` seconds per request, API_MAX_RETRIES retries (arcforge.arc3_note: the
    orchestrator's per-attempt limit is derived from these, judge_attempt_seconds)."""
    import anthropic
    key = os.environ.get('ANTHROPIC_API_KEY_2')       # user 10-29: the judge uses key 2
    if not key:
        raise RuntimeError('ANTHROPIC_API_KEY_2 not set: source the project .env')
    return anthropic.Anthropic(api_key=key, max_retries=API_MAX_RETRIES, timeout=timeout)


BACKENDS = ('api', 'codex', 'claude')
API_EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
#: the claude backend: Claude Code headless (``claude -p``) on the user's subscription login (v8b, user 11-01)
CLAUDE_MODEL = 'claude-opus-5-5'
CLAUDE_EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
#: every ``claude -p`` call: no tools, no settings files (so no hooks / plugins), no MCP, no session kept, the system
#: prompt replaced. Tested 11-01: the model then sees no CLAUDE.md and no auto-memory, only ~800 tokens of environment
#: info. ``--bare`` would also drop those but takes an API key only (no subscription login).
CLAUDE_FLAGS = ('--tools', '', '--setting-sources', '', '--strict-mcp-config', '--no-session-persistence',
                '--output-format', 'json')
#: model_reasoning_effort values codex exec takes
CODEX_EFFORTS = ('minimal', 'low', 'medium', 'high', 'xhigh')
CODEX_MODEL = 'gpt-6-astra'
#: the codex backend's prompt: SYSTEM, this line, the rules, the notes (the model has a shell; it needs none)
CODEX_NO_TOOLS = 'Everything you need is in this message: do not run commands or read files.'
#: the per-request limit when none is given (api: per SDK request; codex: per codex exec)
CALL_TIMEOUT = 900.0
#: every codex exec gets these explicitly: --ignore-user-config drops ~/.codex/config.toml, whose 800k window the model
#: would otherwise lose (gpt-6-astra's built-in default is 272k)
CODEX_CONFIG = (('model_context_window', 800000), ('model_auto_compact_token_limit', 600000))
#: seconds to wait for a killed codex exec's pipes before SIGKILLing its whole process group
CODEX_KILL_WAIT = 30
#: the codex exec children running now (each its own process group): the CLI's SIGTERM handler kills their groups (the
#: orchestrator also sweeps the judge's whole session)
_CODEX_CHILDREN = set()
#: api overload retries (429 / 529 after the SDK's retries): backoff seconds (the last repeats), statuses, and the time
#: spent so far in this process (refused calls + sleeps), at most API_OVERLOAD_WAIT_SECONDS
API_OVERLOAD_BACKOFF = (30, 60, 120, 240, 300)
API_OVERLOAD_STATUS = (429, 529)
_OVERLOAD = {'spent': 0.0}


def _kill_codex(proc):
    """SIGKILL a codex exec's whole process group (it was started with its own: process_group=0)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def api_create(client, **request):
    """``client.messages.create(**request)``; a refusal as rate-limited / overloaded (status API_OVERLOAD_STATUS, after the
    SDK's own retries) is retried after API_OVERLOAD_BACKOFF, logged each time, while this process's overload time
    (refused calls + sleeps) stays within API_OVERLOAD_WAIT_SECONDS; then it raises. Timeouts and every other error
    raise at once (they count against the judge's attempts as before)."""
    k = 0
    while True:
        t0 = time.monotonic()
        try:
            return client.messages.create(**request)
        except Exception as e:
            status = getattr(e, 'status_code', None)
            if status not in API_OVERLOAD_STATUS:
                raise
            _OVERLOAD['spent'] += time.monotonic() - t0
            wait = API_OVERLOAD_BACKOFF[min(k, len(API_OVERLOAD_BACKOFF) - 1)]
            if _OVERLOAD['spent'] + wait > API_OVERLOAD_WAIT_SECONDS:
                raise RuntimeError(f'api still overloaded (status {status}) after {_OVERLOAD["spent"]:.0f}s of '
                                   f'{API_OVERLOAD_WAIT_SECONDS:g}s overload budget: {e}') from e
            print(f'api overloaded (status {status}): retry {k + 1} in {wait}s (overload time '
                  f'{_OVERLOAD["spent"]:.0f}/{API_OVERLOAD_WAIT_SECONDS:g}s): {e}', file=sys.stderr, flush=True)
            time.sleep(wait)
            _OVERLOAD['spent'] += wait
            k += 1


class CodexClient:
    """The codex backend: ``codex_bin exec`` with the model and reasoning effort it was built with, a fresh empty working
    directory under ``workdir`` for every call, at most ``call_timeout`` seconds per call. It runs with the caller's
    CODEX_HOME (the default ~/.codex: the user's ChatGPT login; never a copy or link of auth.json, whose token refresh
    could desync two logins) and ``--ignore-user-config`` (no $CODEX_HOME/config.toml; every setting the judge depends on
    is an explicit flag). Not covered by that flag: codex may still load ~/.codex/AGENTS.md as user instructions."""

    backend = 'codex'

    def __init__(self, model, effort, workdir, call_timeout=CALL_TIMEOUT, codex_bin='codex'):
        if effort not in CODEX_EFFORTS:
            raise ValueError(f'codex effort {effort!r} is not one of {CODEX_EFFORTS}')
        if not Path(workdir).is_dir():
            raise ValueError(f'codex workdir {workdir} is not a directory')
        self.model, self.effort, self.workdir, self.call_timeout = model, effort, Path(workdir), call_timeout
        self.codex_bin = codex_bin


class ClaudeCodeClient:
    """The claude backend: ``claude_bin -p`` (CLAUDE_FLAGS) with the model and effort it was built with, a fresh empty
    working directory under ``workdir`` for every call, at most ``call_timeout`` seconds per call. It runs on the
    caller's Claude Code login (the user's subscription, user 11-01)."""

    backend = 'claude'

    def __init__(self, model, effort, workdir, call_timeout=CALL_TIMEOUT, claude_bin='claude'):
        if effort not in CLAUDE_EFFORTS:
            raise ValueError(f'claude effort {effort!r} is not one of {CLAUDE_EFFORTS}')
        if not Path(workdir).is_dir():
            raise ValueError(f'claude workdir {workdir} is not a directory')
        self.model, self.effort, self.workdir, self.call_timeout = model, effort, Path(workdir), call_timeout
        self.claude_bin = claude_bin


def check_grades(data, order, n, where):
    """The judge's JSON -> {index: grade entry}: every note of ``order`` [(label, index)] exactly once, no other id."""
    by_label = {label: i for label, i in order}
    got = {}
    for g in data['grades']:
        if g['note'] not in by_label or by_label[g['note']] in got:
            raise IncompleteGrades(f'judge returned unknown or repeated note id {g["note"]!r} ({where})')
        got[by_label[g['note']]] = g
    if len(got) != n:
        raise IncompleteGrades(f'judge graded {len(got)} of {n} notes ({where})')
    return got


def _run_child(cmd, stdin_text, timeout, name, cwd=None):
    """Run ``cmd`` in its own process group with ``stdin_text`` on stdin; returns (returncode, stdout, stderr). A
    timeout kills the whole group and raises RuntimeError with the stderr tail; SIGTERM of the judge (the handler's
    SystemExit) or any error never leaves the child behind."""
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            process_group=0, cwd=cwd)
    _CODEX_CHILDREN.add(proc)
    try:
        stdout, stderr = proc.communicate(stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=CODEX_KILL_WAIT)
        except subprocess.TimeoutExpired:       # its children still hold the pipes: the whole group goes
            _kill_codex(proc)
            try:
                stdout, stderr = proc.communicate(timeout=CODEX_KILL_WAIT)
            except subprocess.TimeoutExpired:
                stderr = (f'(stderr not read: the pipes are still held {CODEX_KILL_WAIT}s after SIGKILL of {name} '
                          f'process group {proc.pid})')
        raise RuntimeError(f'{name} timed out after {timeout}s: {stderr[-2000:]}') from None
    finally:                                    # SIGTERM (SystemExit from the handler) or any error: never leave it
        _kill_codex(proc)                       # also any leftover of a finished call
        if proc.poll() is None:
            proc.wait(timeout=CODEX_KILL_WAIT)
        _CODEX_CHILDREN.discard(proc)
    return proc.returncode, stdout, stderr


def codex_call(client, prompt, schema):
    """One ``codex exec`` with ``prompt`` on stdin and ``schema`` as the output schema; the last message (``-o``) parsed
    as JSON. A non-zero exit or no output file raises RuntimeError with the stderr tail. Returns (data, meta)."""
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=client.workdir, prefix='codex-judge-') as d:
        schema_f, out = Path(d) / 'schema.json', Path(d) / 'out.json'
        schema_f.write_text(json.dumps(schema))
        work = Path(d) / 'work'                 # codex's -C: empty, nothing to read
        work.mkdir()
        config = [a for k, v in CODEX_CONFIG for a in ('-c', f'{k}={v}')]
        rc, _, stderr = _run_child([client.codex_bin, 'exec', '--ignore-user-config', '-m', client.model, '-c',
                                    f'model_reasoning_effort="{client.effort}"', *config, '--disable', 'memories',
                                    '-s', 'read-only', '--ephemeral', '--skip-git-repo-check', '-C', str(work),
                                    '--output-schema', str(schema_f), '-o', str(out), '-'],
                                   prompt, client.call_timeout, 'codex exec')
        if rc != 0 or not out.exists():
            raise RuntimeError(f'codex exec rc {rc}{"" if out.exists() else ", no output file"}: {stderr[-2000:]}')
        data = json.loads(out.read_text())
    return data, {'backend': 'codex', 'model': client.model, 'effort': client.effort, 'config': dict(CODEX_CONFIG),
                  'seconds': round(time.time() - t0, 1)}


def claude_call(client, system, prompt, schema):
    """One ``claude -p`` (CLAUDE_FLAGS) with ``system`` as the whole system prompt, ``prompt`` on stdin and ``schema``
    as the structured-output schema, run in a fresh empty directory. Raises RuntimeError on a non-zero exit, an error
    result or a missing structured output. Returns (data, meta)."""
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=client.workdir, prefix='claude-judge-') as d:
        rc, stdout, stderr = _run_child([client.claude_bin, '-p', '--system-prompt', system, *CLAUDE_FLAGS,
                                         '--model', client.model, '--effort', client.effort,
                                         '--json-schema', json.dumps(schema)],
                                        prompt, client.call_timeout, 'claude -p', cwd=d)
    if rc != 0:
        raise RuntimeError(f'claude -p rc {rc}: {stderr[-2000:]} {stdout[-2000:]}')
    res = json.loads(stdout)
    if res.get('is_error') or res.get('subtype') != 'success' or res.get('structured_output') is None:
        raise RuntimeError(f'claude -p result is_error={res.get("is_error")} subtype={res.get("subtype")} '
                           f'api_error_status={res.get("api_error_status")} structured_output '
                           f'{"missing" if res.get("structured_output") is None else "present"}: '
                           f'{str(res.get("result"))[:2000]}')
    u = res.get('usage', {})
    return res['structured_output'], {
        'backend': 'claude', 'model': client.model, 'effort': client.effort, 'seconds': round(time.time() - t0, 1),
        'input': u.get('input_tokens'), 'output': u.get('output_tokens'), 'cache_read': u.get('cache_read_input_tokens'),
        'cache_write': u.get('cache_creation_input_tokens'), 'session_id': res.get('session_id')}


def codex_grade_once(client, game, level, baseline, notes, order):
    """grade_once on the codex backend: the same prompt and schema through ``codex exec`` (codex_call)."""
    prompt = (SYSTEM + '\n\n' + CODEX_NO_TOOLS + '\n\n' + rules_text(game) + '\n\n'
              + user_text(game, level, baseline, notes, order))
    data, meta = codex_call(client, prompt, SCHEMA)
    got = check_grades(data, order, len(notes), f'codex {client.model}')
    return got, meta


def grade_once(client, game, level, baseline, notes, order, model, effort):
    """One judge call over the notes in ``order`` [(label, index)]; returns {index: grade entry} and usage. A
    ``CodexClient`` goes to codex_grade_once (its own model / effort); otherwise the Anthropic client."""
    if isinstance(client, CodexClient):
        return codex_grade_once(client, game, level, baseline, notes, order)
    resp = api_create(
        client, model=model, max_tokens=32000,
        output_config={'effort': effort, 'format': {'type': 'json_schema', 'schema': SCHEMA}},
        system=[{'type': 'text', 'text': SYSTEM},
                {'type': 'text', 'text': rules_text(game), 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}],
        messages=[{'role': 'user', 'content': user_text(game, level, baseline, notes, order)}])
    if resp.stop_reason != 'end_turn':
        raise RuntimeError(f'judge stop_reason {resp.stop_reason} ({getattr(resp, "stop_details", None)}), '
                           f'request {resp._request_id}')
    data = json.loads(next(b.text for b in resp.content if b.type == 'text'))
    got = check_grades(data, order, len(notes), f'request {resp._request_id}')
    usage = resp.usage
    return got, {'request_id': resp._request_id, 'input': usage.input_tokens, 'output': usage.output_tokens,
                 'cache_read': usage.cache_read_input_tokens, 'cache_write': usage.cache_creation_input_tokens}


class IncompleteGrades(RuntimeError):
    """The judge's JSON left out, repeated or invented a note id (Opus did this in ~5% of v7 validation calls)."""


def grade_pass(client, game, level, baseline, notes, order, model, effort):
    """grade_once, retried (same order) when the reply does not grade every note exactly once; other errors raise."""
    for attempt in range(1, PASS_ATTEMPTS + 1):
        try:
            return grade_once(client, game, level, baseline, notes, order, model, effort)
        except IncompleteGrades as e:
            print(f'judge pass attempt {attempt}/{PASS_ATTEMPTS}: {e}', file=sys.stderr, flush=True)
            if attempt == PASS_ATTEMPTS:
                raise


def labels_for(n, rng):
    idx = list(range(n))
    rng.shuffle(idx)
    names = rng.sample(range(100, 1000), n)
    return [(f'N{names[k]}', i) for k, i in enumerate(idx)]


def judge_group(group, model='claude-opus-5-5', effort='high', seed=0, client=None):
    """Grade every note of ``group`` (see the module docstring) with two shuffled passes, a third when they disagree by
    more than one grade. Returns the per-note result with the reward and every pass for audit. ``client``: None = the
    Anthropic API (``model`` / ``effort``); a ``CodexClient`` = the codex backend, whose model and effort must be
    ``model`` / ``effort``."""
    notes, b = group['notes'], group['baseline']
    if not notes:
        raise ValueError('a group with no uncleared notes needs no judge')
    client = client or client_from_env()
    backend = 'codex' if isinstance(client, CodexClient) else 'api'
    if backend == 'codex' and (client.model, client.effort) != (model, effort):
        raise ValueError(f'judge_group model {model} effort {effort}, its codex client {client.model} {client.effort}')
    rng = random.Random(seed)
    passes, usage = [], []
    for _ in range(2):
        got, u = grade_pass(client, group['game'], group['level'], b, notes, labels_for(len(notes), rng), model, effort)
        passes.append(got)
        usage.append(u)
    final = {i: combine_grades([p[i]['grade'] for p in passes]) for i in range(len(notes))}
    if any(v is None for v in final.values()):
        got, u = grade_pass(client, group['game'], group['level'], b, notes, labels_for(len(notes), rng), model, effort)
        passes.append(got)
        usage.append(u)
        final = {i: combine_grades([p[i]['grade'] for p in passes]) if final[i] is None else final[i]
                 for i in range(len(notes))}
    out = []
    for i, n in enumerate(notes):
        rem = median_remaining([p[i]['remaining_actions'] for p in passes])   # raises on a negative estimate
        out.append({'id': n['id'], 'actions': n['actions'], 'resets': n['resets'], 'grade': final[i],
                    'remaining_actions': rem, 'reward': note_reward(final[i], n['actions'], rem, b),
                    'passes': [{k: p[i][k] for k in ('grade', 'remaining_actions', 'confirmed', 'open', 'misleading',
                                                     'reason')} for p in passes]})
    return {'game': group['game'], 'level': group['level'], 'baseline': b, 'backend': backend, 'model': model,
            'effort': effort, 'seed': seed, 'notes': out, 'usage': usage}


# ---------------------------------------------------------------------------------------------------------------------
# v8b (docs/plans/arc3_rl_v8b_judge_ensemble_20261101.md): a per-level answer key, a group ranking with a flexible
# quota on top of the absolute grades, and an ensemble of judges (Claude Code + codex), one pass each per round.

KEYS = Path(__file__).resolve().parent / 'arc3_level_keys'

SYSTEM_V8B = """You grade handoff notes written by an AI agent playing one level of an ARC-AGI-3 game.

The agent played the level from its start until its context window filled up. The note is what its successor receives:
the successor is the same agent, starting with an empty context and only this handoff (its notes, its latest level
file, the server state), and continues the same level from the same game state, with the actions already spent
counted against it. The level is scored by the official rule: level score = min(115, 100 * (baseline /
total_actions)^2), where total_actions counts every action of the level including RESETs, including the actions
already spent. A clear needing more than the baseline is a poor outcome.

You have the game's full rules (accurate; the agent never saw them) and an answer key for this level prepared from
those rules: the goal, the decisive pieces (D1, D2, ...), common traps (T1, T2, ...), a solution sketch and, when the
rules state one, the length of a clearing sequence from the level start. Use the key as your checklist; the rules
settle anything the key leaves open. For each note answer one question: **if the successor takes over with this note
and the actions already spent, how likely is it to clear, and with how many more actions?**

Per note:
1. key_confirmed: ids of the decisive pieces the note gets right for this level and treats as established or acts on (a
   piece counts only if its decisive part is right). Knowledge carried over from earlier levels of the game counts only
   where the note applies it to this level's board (its objects, positions, goal); notes may carry earlier levels' text.
2. key_wrong: ids of decisive pieces the note contradicts and of traps the note holds as established. A wrong idea the
   note only lists as one untested possibility does not count here.
3. confirmed / open / misleading: short. confirmed = what the successor can act on; open = what is unresolved (a note
   that has narrowed the possibilities to a few and states a cheap test that tells them apart has made real progress);
   misleading = claims that would lead the successor to wrong actions.
4. remaining_actions: the further actions (RESETs included) the successor would most likely need to clear from the
   current state by following this note, using the known clearing length and how far the note says the level has
   progressed. If it would likely fail, estimate what a clear would take anyway.
5. grade (absolute; it does not depend on the other notes, and every note in a group may share one grade):
   - S: every decisive piece confirmed, no core error, concrete workable next steps: a clear is very likely.
   - A: all decisive pieces but one confirmed (or all, with a fixable plan error), no core error: a clear is likely.
   - B: some decisive pieces confirmed, a decisive piece missing, no confidently held core error: a clear is uncertain.
   - C: no decisive piece confirmed, but correct peripheral observations (controls, objects) the successor can use.
   - D: almost nothing usable about this level.
   - F: the note's goal for this level or its plan rests on a trap or on a contradicted decisive piece, so it would send
     the successor the wrong way; or its level-specific content (goal, objects, positions) is not about this level; or
     nothing about this level at all.
   A wrong belief the plan does not rest on is not F: it lowers the grade by at most one step. A note with confirmed
   decisive pieces and a confidently held core error: an error the note's own evidence would soon overturn is B or C;
   an error its plan builds on is F.
The action cost is applied separately (from the actions already spent plus your remaining_actions), so neither the
grade nor the ranking below is lowered for actions already spent; do make remaining_actions honest.

Ranking (the comparison training uses): after grading, order all notes of the group from best to worst as a list of
buckets, best first; notes in one bucket are equally good for the successor. All notes in a bucket share one grade, and
grades never get better further down the list. Within a grade (F included), separate notes by what matters for
clearing this level: more decisive pieces confirmed and more precisely, a better-founded and more concrete plan, fewer
or less harmful errors, fewer further actions needed. Flexible quota: with n notes use roughly n/3 to n/2 buckets and keep any
bucket to about a third of the notes. Use fewer or larger buckets only when notes are genuinely indistinguishable on
those criteria, and say why in ranking_note. Never separate notes on length, polish, confidence or formatting.

Rules for grading:
- Credit what the note commits to or has confirmed, and real narrowing of uncertainty; a list of mutually exclusive
  guesses without a deciding test earns nothing.
- Claims the agent makes about its own work (a simulator "validated", a plan "ready", "two steps from the goal") are
  the agent's statements, not facts: credit them only as far as the note's content makes them plausible against the
  rules, the key and the server state.
- Length, confidence and polish earn nothing. A note cut at the length limit is graded on what is shown.
- Notes may carry text from earlier levels of the same game; grade what matters for this level.

Return JSON only: one entry per note id, and the ranking as lists of note ids."""

SCHEMA_V8B = {
    'type': 'object',
    'properties': {
        'grades': {'type': 'array', 'items': {
            'type': 'object',
            'properties': {'note': {'type': 'string'},
                           'key_confirmed': {'type': 'array', 'items': {'type': 'string'}},
                           'key_wrong': {'type': 'array', 'items': {'type': 'string'}},
                           'confirmed': {'type': 'string'}, 'open': {'type': 'string'},
                           'misleading': {'type': 'string'}, 'remaining_actions': {'type': 'integer'},
                           'grade': {'type': 'string', 'enum': list(ORDER)}, 'reason': {'type': 'string'}},
            'required': ['note', 'key_confirmed', 'key_wrong', 'confirmed', 'open', 'misleading',
                         'remaining_actions', 'grade', 'reason'],
            'additionalProperties': False}},
        'ranking': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string'}}},
        'ranking_note': {'type': 'string'}},
    'required': ['grades', 'ranking', 'ranking_note'], 'additionalProperties': False}


def level_key(game, level):
    """The answer key of ``game`` level ``level`` (arc3_level_keys/<game>.json, built by
    probes/arc3v8b_20261101/build_level_keys.py). A missing game file or level raises."""
    doc = json.loads((KEYS / f'{game}.json').read_text())
    for k in doc['levels']:
        if k['level'] == level:
            return k
    raise KeyError(f'no answer key for {game} level {level} in {KEYS / f"{game}.json"}')


def key_text(game, level):
    k = level_key(game, level)
    sol = (f'{k["solution_actions"]} actions' if k['solution_actions'] is not None else 'not stated by the rules')
    lines = [f'# Answer key for {game.upper()} level {level} (prepared from the rules; the agent never saw it)',
             f'Goal: {k["goal"]}', 'Decisive pieces:', *[f'- {d["id"]}: {d["text"]}' for d in k['decisive']],
             'Traps:', *[f'- {t["id"]}: {t["text"]}' for t in k['traps']], f'Solution sketch: {k["sketch"]}',
             f'Known clearing length from the level start: {sol} (human baseline {k["baseline"]})']
    if k['uncertain']:
        lines.append(f'Unclear in the rules: {k["uncertain"]}')
    return '\n'.join(lines)


def check_v8b(data, order, n, key, where):
    """The v8b JSON -> ({index: grade entry}, buckets [[index]]): every note once in grades and once in the ranking,
    one grade per bucket, grades never better further down, key ids from the key. Anything else is IncompleteGrades
    (the pass is retried)."""
    got = check_grades(data, order, n, where)
    by_label = {label: i for label, i in order}
    seen, buckets = set(), []
    for b in data['ranking']:
        if not b:
            raise IncompleteGrades(f'empty bucket in the ranking ({where})')
        idx = []
        for label in b:
            if label not in by_label or by_label[label] in seen:
                raise IncompleteGrades(f'ranking has unknown or repeated note id {label!r} ({where})')
            seen.add(by_label[label])
            idx.append(by_label[label])
        buckets.append(idx)
    if len(seen) != n:
        raise IncompleteGrades(f'ranking covers {len(seen)} of {n} notes ({where})')
    prev = -1
    for idx in buckets:
        gs = {got[i]['grade'] for i in idx}
        if len(gs) != 1:
            raise IncompleteGrades(f'a ranking bucket mixes grades {sorted(gs)} ({where})')
        g = ORDER.index(gs.pop())
        if g < prev:
            raise IncompleteGrades(f'ranking puts a better grade below a worse one ({where})')
        prev = g
    dec = {d['id'] for d in key['decisive']}
    ids = dec | {t['id'] for t in key['traps']}
    for i, e in got.items():
        if not set(e['key_confirmed']) <= dec or not set(e['key_wrong']) <= ids:
            raise IncompleteGrades(f'note {order[[j for _, j in order].index(i)][0]}: key ids '
                                   f'{e["key_confirmed"]} / {e["key_wrong"]} not in the key ({where})')
    return got, buckets


def schema_v8b(key):
    """SCHEMA_V8B with the key's ids as enums (key_confirmed: the decisive ids; key_wrong: decisive and trap ids), so the
    structured output cannot cite an id the key does not have."""
    schema = json.loads(json.dumps(SCHEMA_V8B))
    props = schema['properties']['grades']['items']['properties']
    dec = [d['id'] for d in key['decisive']]
    props['key_confirmed']['items'] = {'type': 'string', 'enum': dec}
    props['key_wrong']['items'] = {'type': 'string', 'enum': dec + [t['id'] for t in key['traps']]}
    return schema


def grade_once_v8b(client, game, level, baseline, notes, order):
    """One v8b pass of ``client`` (ClaudeCodeClient or CodexClient) over the notes in ``order``."""
    key = level_key(game, level)
    body = rules_text(game) + '\n\n' + key_text(game, level) + '\n\n' + user_text(game, level, baseline, notes, order)
    schema = schema_v8b(key)
    if isinstance(client, ClaudeCodeClient):
        data, meta = claude_call(client, SYSTEM_V8B, body, schema)
    elif isinstance(client, CodexClient):
        data, meta = codex_call(client, SYSTEM_V8B + '\n\n' + CODEX_NO_TOOLS + '\n\n' + body, schema)
    else:
        raise TypeError(f'v8b judge client {client!r}: want ClaudeCodeClient or CodexClient')
    got, buckets = check_v8b(data, order, len(notes), key, f'{meta["backend"]} {client.model}')
    return got, buckets, data['ranking_note'], meta


def grade_pass_v8b(client, game, level, baseline, notes, order):
    """grade_once_v8b, retried (same order) on IncompleteGrades; other errors raise."""
    for attempt in range(1, PASS_ATTEMPTS + 1):
        try:
            return grade_once_v8b(client, game, level, baseline, notes, order)
        except IncompleteGrades as e:
            print(f'v8b pass attempt {attempt}/{PASS_ATTEMPTS} ({client.backend}): {e}', file=sys.stderr, flush=True)
            if attempt == PASS_ATTEMPTS:
                raise


def judge_group_v8b(group, clients, mapping, seed=0, min_rounds=1, max_rounds=V8B_MAX_ROUNDS):
    """Grade every note of ``group`` with each of ``clients`` (one pass each per round, in parallel, every pass its own
    shuffled order). A second round runs when some note's grades across the round-1 passes are more than one grade
    apart (or when ``min_rounds`` is 2). Returns every pass per note (grade, its bucket in that pass's ranking, the key
    ids, remaining_actions) and the note's score / remaining_actions / reward under ``mapping``
    (arcforge.arc3_note.v8b_note: the mean pass score, capped by the clear reward at the estimated total)."""
    from concurrent.futures import ThreadPoolExecutor
    notes, b = group['notes'], group['baseline']
    if not notes:
        raise ValueError('a group with no uncleared notes needs no judge')
    if not clients or len({c.backend for c in clients}) != len(clients):
        raise ValueError(f'v8b judge clients {[c.backend for c in clients]}: want one client per backend')
    if not 1 <= min_rounds <= max_rounds <= V8B_MAX_ROUNDS:
        raise ValueError(f'rounds {min_rounds}..{max_rounds}: want 1 <= min <= max <= {V8B_MAX_ROUNDS}')
    if mapping not in V8B_MAPPINGS:
        raise ValueError(f'v8b mapping {mapping!r} is not one of {V8B_MAPPINGS}')
    rng = random.Random(seed)
    passes = []
    for rnd in range(1, max_rounds + 1):
        orders = [labels_for(len(notes), rng) for _ in clients]
        with ThreadPoolExecutor(len(clients)) as ex:
            futs = [ex.submit(grade_pass_v8b, c, group['game'], group['level'], b, notes, o)
                    for c, o in zip(clients, orders)]
            for c, f in zip(clients, futs):
                got, buckets, rnote, meta = f.result()
                passes.append({'round': rnd, 'backend': c.backend, 'model': c.model, 'effort': c.effort,
                               'got': got, 'buckets': buckets, 'ranking_note': rnote, 'meta': meta})
        spread = max(max(ORDER.index(p['got'][i]['grade']) for p in passes)
                     - min(ORDER.index(p['got'][i]['grade']) for p in passes) for i in range(len(notes)))
        if rnd >= min_rounds and spread <= 1:
            break
    out = []
    for i, n in enumerate(notes):
        per = []
        for p in passes:
            e = p['got'][i]
            bucket = next(j for j, idx in enumerate(p['buckets']) if i in idx)
            per.append({'round': p['round'], 'backend': p['backend'], 'grade': e['grade'], 'bucket': bucket,
                        'buckets': [len(x) for x in p['buckets']],
                        'bucket_grades': [p['got'][x[0]]['grade'] for x in p['buckets']],
                        **{k: e[k] for k in ('remaining_actions', 'key_confirmed', 'key_wrong', 'confirmed', 'open',
                                             'misleading', 'reason')}})
        score, rem, reward = v8b_note(per, n['actions'], b, mapping)
        out.append({'id': n['id'], 'actions': n['actions'], 'resets': n['resets'], 'score': score,
                    'remaining_actions': rem, 'reward': reward, 'passes': per})
    return {'game': group['game'], 'level': group['level'], 'baseline': b, 'version': V8B_VERSION, 'mapping': mapping,
            'prompt_sha256': hashlib.sha256((SYSTEM_V8B + json.dumps(SCHEMA_V8B, sort_keys=True)).encode()).hexdigest(),
            'seed': seed,
            'judges': [{'backend': c.backend, 'model': c.model, 'effort': c.effort} for c in clients],
            'rounds': max(p['round'] for p in passes), 'notes': out,
            'rankings': [{'round': p['round'], 'backend': p['backend'], 'ranking_note': p['ranking_note'],
                          'buckets': [[notes[i]['id'] for i in idx] for idx in p['buckets']]} for p in passes],
            'usage': [p['meta'] for p in passes]}


def _on_sigterm(signum, frame):
    """SIGTERM (the orchestrator's judge timeout or stop): kill the codex exec children, then exit -- the temp dirs go
    with the unwinding ``with`` blocks."""
    for child in list(_CODEX_CHILDREN):
        _kill_codex(child)
    raise SystemExit(128 + signum)


CLI_BACKENDS = ('api', 'codex', 'ensemble', 'claude-code')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('group', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--backend', choices=CLI_BACKENDS, default='api',
                    help='api: the Anthropic API (--model / --effort, ANTHROPIC_API_KEY_2); codex: codex exec with the '
                         'ChatGPT login (--codex-model / --codex-effort, no API key); ensemble: v8b, Claude Code (the '
                         'subscription login: --claude-model / --claude-effort / --claude-bin) and codex, one pass each '
                         'per round, scored with --mapping; claude-code: v8b with Claude Code alone, one pass (user 11-01 '
                         'after the test set)')
    ap.add_argument('--model', default=None, help='api model (default claude-opus-5-5)')
    ap.add_argument('--effort', default=None, choices=API_EFFORTS, help='api effort (default high)')
    ap.add_argument('--codex-model', default=None, help=f'codex model (default {CODEX_MODEL})')
    ap.add_argument('--codex-effort', default=None, choices=CODEX_EFFORTS, help='codex reasoning effort (default xhigh)')
    ap.add_argument('--codex-bin', default=None, help='codex: the codex executable (default: codex on PATH)')
    ap.add_argument('--claude-model', default=None,
                    help=f'ensemble, claude-code: the Claude Code model (default {CLAUDE_MODEL})')
    ap.add_argument('--claude-effort', default=None, choices=CLAUDE_EFFORTS,
                    help='ensemble, claude-code: Claude Code effort')
    ap.add_argument('--claude-bin', default=None,
                    help='ensemble, claude-code: the claude executable (default: claude on PATH)')
    ap.add_argument('--mapping', default=None, choices=V8B_MAPPINGS,
                    help='ensemble, claude-code: the pass score mapping')
    ap.add_argument('--call-timeout', type=float, default=CALL_TIMEOUT,
                    help=f'seconds per backend request (api: per SDK request, {API_MAX_RETRIES} retries; codex: per codex '
                         f'exec; claude: per claude -p). Default {CALL_TIMEOUT:g}')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_sigterm)
    own = {'api': ('--model', '--effort'), 'codex': ('--codex-model', '--codex-effort', '--codex-bin'),
           'ensemble': ('--codex-model', '--codex-effort', '--codex-bin', '--claude-model', '--claude-effort',
                        '--claude-bin', '--mapping'),
           'claude-code': ('--claude-model', '--claude-effort', '--claude-bin', '--mapping')}[args.backend]
    every = ('--model', '--effort', '--codex-model', '--codex-effort', '--codex-bin', '--claude-model', '--claude-effort',
             '--claude-bin', '--mapping')
    given = [f for f in every if f not in own and getattr(args, f[2:].replace('-', '_')) is not None]
    if given:
        ap.error(f'{given} are not flags of --backend {args.backend}')
    t0 = time.time()
    group = json.loads(args.group.read_text())
    if args.backend in ('ensemble', 'claude-code'):
        need = ('--claude-effort', '--mapping') + (('--codex-effort',) if args.backend == 'ensemble' else ())
        missing = [f for f in need if getattr(args, f[2:].replace('-', '_')) is None]
        if missing:
            ap.error(f'--backend {args.backend} needs {missing} (no defaults: the run config pins them)')
        work = args.out.absolute().parent
        clients = [ClaudeCodeClient(args.claude_model or CLAUDE_MODEL, args.claude_effort, work, args.call_timeout,
                                    args.claude_bin or 'claude')]
        if args.backend == 'ensemble':
            clients.append(CodexClient(args.codex_model or CODEX_MODEL, args.codex_effort, work, args.call_timeout,
                                       args.codex_bin or 'codex'))
        # claude-code: one judge, so no second round (arcforge.arc3_note.judge_attempt_seconds bounds one round)
        result = judge_group_v8b(group, clients, args.mapping, args.seed,
                                 max_rounds=V8B_MAX_ROUNDS if args.backend == 'ensemble' else 1)
        summary = [[q['grade'] for q in n['passes']] for n in result['notes']]
    else:
        if args.backend == 'api':
            model, effort = args.model or 'claude-opus-5-5', args.effort or 'high'
            client = client_from_env(args.call_timeout)
        else:
            model, effort = args.codex_model or CODEX_MODEL, args.codex_effort or 'xhigh'
            client = CodexClient(model, effort, args.out.absolute().parent, args.call_timeout, args.codex_bin or 'codex')
        result = judge_group(group, model, effort, args.seed, client)
        summary = [n['grade'] for n in result['notes']]
    result['seconds'] = round(time.time() - t0, 1)
    tmp = args.out.with_suffix(args.out.suffix + '.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    tmp.rename(args.out)
    print(json.dumps({'out': str(args.out), 'grades': summary, 'seconds': result['seconds']}), file=sys.stderr)


if __name__ == '__main__':
    main()
