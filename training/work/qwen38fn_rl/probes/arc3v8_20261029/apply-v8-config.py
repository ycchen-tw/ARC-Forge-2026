"""v8 config after configure_b200 (before the first start), as ../arc3v7b_20261029/apply-v7b-config.py:
prepare.max_d_cross_fraction 0.05 and prepare.max_routing_m_fraction 0.03 (no in-loop held-out); plan.json
config_sha256 follows. Checks the v8 settings (v8.env) in plan.json and both harness configs first. Exact-once; backups
*.before-v8-<stamp>.json.
    python3 apply-v8-config.py <RL_RUN> <worktree with configure_b200.py> <train ids> [smoke]"""
import json
import shutil
import sys
import time
from pathlib import Path

run, repo, ntrain = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
smoke = sys.argv[4:] == ["smoke"]          # launch-v8.sh RL_V8_SMOKE=1: smoke/harness-v8-smoke.json, smoke template
if sys.argv[4:] not in ([], ["smoke"]):
    raise SystemExit(f"extra arguments {sys.argv[4:]}: only 'smoke'")
sys.path.insert(0, str(repo / "work/qwen38fn_rl"))
from configure_b200 import config_fingerprint  # noqa: E402

prepare = {"max_d_cross_fraction": 0.05, "max_routing_m_fraction": 0.03}
cfg_p, plan_p = run / "config.json", run / "plan.json"
cfg, plan = json.loads(cfg_p.read_text()), json.loads(plan_p.read_text())
if set(prepare) & set(cfg["prepare"]):
    raise SystemExit(f"{cfg_p}: already applied")
if plan["config_sha256"] != config_fingerprint(cfg):
    raise SystemExit(f"{plan_p}: config_sha256 does not match config.json before the change")
if any(k.startswith("heldout_") for k in cfg["stream"]):
    raise SystemExit(f"{cfg_p}: stream has held-out keys {[k for k in cfg['stream'] if k.startswith('heldout_')]}")
want = {"seed_note_done": True, "packet_level_docs": "latest", "pace_line": False, "window_change": True,
        "stop_after_first_window": True, "context_token_limit": None, "arc3_reward": {"mode": "note"},
        "train_tasks": ntrain, "heldout_tasks": 2, "sessions_per_replica": 64, "serve_mamba_slots": 384,
        "rollout_slots": 256, "ramp_seconds": 600.0, "output_token_budget": 300000, "train_output_token_budget": None,
        "group_size": 16, "serve_memfrac": 0.94, "trainer_trainable": "full", "trainer_lr": 3e-6,
        "trainer_lr_warmup_steps": 2, "trainer_keep_checkpoints": 3,
        "serve_profile": "perf-main-fp16-ser-opt-lazyfix-l8-a16-topp-inldev"}
have = {k: plan.get(k, "<missing>") for k in want}
if isinstance(have["serve_profile"], dict):
    have["serve_profile"] = have["serve_profile"].get("name")
if have != want:
    raise SystemExit(f"{plan_p}: {have} is not v8's {want}")
if plan.get("init") is not None:
    raise SystemExit(f"{plan_p}: init {plan['init']}: v8 starts from QAD-R")
if plan["note_judge"]["backend"] not in ("api", "codex") or plan["note_judge"] != cfg["note_judge"]:
    raise SystemExit(f"{plan_p}: note_judge {plan.get('note_judge')} / config {cfg.get('note_judge')}")
if plan["topology"]["name"] != "tp1x4" or plan["trainer_gpus"] != "4,5,6,7":
    raise SystemExit(f"{plan_p}: topology {plan['topology']} / trainer {plan['trainer_gpus']} is not tp1x4")
for name, first_window in (("harness-train.json", True), ("harness-eval.json", None)):
    h = json.loads((run / name).read_text())
    got = {k: h.get(k) for k in ("output_token_budget", "seed_note_done", "packet_level_docs", "pace_line", "window_change",
                                 "stop_after_first_window", "context_token_limit", "model_context_window",
                                 "auto_compact_token_limit", "serve_context_length", "episode_wall_seconds")}
    exp = {"output_token_budget": 300000, "seed_note_done": True, "packet_level_docs": "latest", "pace_line": None,
           "window_change": True, "stop_after_first_window": first_window, "context_token_limit": None,
           "model_context_window": 100000 if smoke else 235000, "auto_compact_token_limit": 50000 if smoke else 130000,
           "serve_context_length": 240000, "episode_wall_seconds": 3600 if smoke else 10800}
    if got != exp or "output_token_budget_override" in h:
        raise SystemExit(f"{run / name}: {got} is not v8's {exp}")
if not cfg["prepare"].get("allow_compaction"):
    raise SystemExit(f"{cfg_p}: prepare.allow_compaction must be true (window_change)")
if cfg["stream"]["group_timeout_seconds"] != (7200 if smoke else 14400) or not cfg["stream"]["group_ready"]:
    raise SystemExit(f"{cfg_p}: stream {cfg['stream']} is not v7's")
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
shutil.copy2(cfg_p, run / f"config.before-v8-{stamp}.json")
shutil.copy2(plan_p, run / f"plan.before-v8-{stamp}.json")
cfg["prepare"].update(prepare)
plan["config_sha256"] = config_fingerprint(cfg)
cfg_p.write_text(json.dumps(cfg, indent=1) + "\n")
plan_p.write_text(json.dumps(plan, indent=1) + "\n")
print(f"prepare += {prepare}; plan config_sha256 -> {plan['config_sha256'][:16]}; judge {plan['note_judge']}")
