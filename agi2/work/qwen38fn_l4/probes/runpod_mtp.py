"""Strict h29 MTP profile and native request-counter receipts; stdlib only."""
from __future__ import annotations

import json
import math


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_mtp_steps(mtp_steps):
    require(type(mtp_steps) is int and mtp_steps in (1, 3), "MTP steps must be explicitly 1 or 3")
    return mtp_steps


def check_profile(info, *, mtp_steps=3):
    validate_mtp_steps(mtp_steps)
    verify_tokens = mtp_steps + 1
    expected = {
        "tp_size": 4, "ep_size": 4, "dp_size": 2, "enable_dp_attention": True,
        "speculative_num_steps": mtp_steps, "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": verify_tokens, "speculative_draft_kv_cache_dtype": "fp8_e4m3",
        "speculative_moe_runner_backend": "triton",
        "speculative_attention_mode": "prefill", "enable_linear_replayssm_spec": True,
        "linear_replayssm_cache_len": 16, "mamba_ssm_dtype": "float16",
        "disable_overlap_schedule": False, "speculative_adaptive": False,
        "enable_multi_layer_eagle": False, "enable_hisparse": True,
        "disable_radix_cache": True, "radix_cache_backend": "qsa_hisparse_session",
    }
    if mtp_steps == 1:
        expected["speculative_use_rejection_sampling"] = True
    require(info["speculative_algorithm"] in ("EAGLE", "NEXTN"), "MTP requires NEXTN/EAGLE")
    for key, value in expected.items():
        require(info[key] == value, f"MTP server {key}={info[key]!r}, expected {value!r}")
    config = info["hisparse_config"]
    config = json.loads(config) if isinstance(config, str) else config
    require(config["qsa_session_resume"] is True, "MTP requires retained session snapshots")
    require(config["qsa_hot_blocks"] >= 512 * verify_tokens,
            f"MTP verify{verify_tokens} selected union needs B>={512 * verify_tokens}")
    require(config["qsa_snapshot_staging_slots"] == 4, "MTP requires four staging slots")
    require(config["qsa_snapshot_interval"] > verify_tokens, "MTP snapshots must exceed verify width")
    require(config["qsa_session_idle_timeout_s"] == 0, "MTP session expiry must be disabled")
    return expected | {"speculative_algorithm": info["speculative_algorithm"]}


def request_counters(meta, *, mtp_steps=3):
    """Read final native counts, never infer acceptance from delivery rate.

    The explicit profile determines proposals per verify (default NEXTN3). Its
    accept-length metric uses delivered completion tokens (including the first
    prefill token and any final truncation), unlike correct+verify below.
    """
    validate_mtp_steps(mtp_steps)
    keys = ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts", "completion_tokens")
    for key in keys:
        require(type(meta[key]) is int and meta[key] >= 0, f"invalid native MTP {key}")
    verify, correct, proposed, completion = (meta[key] for key in keys)
    require(verify > 0, "MTP request completed without a native verify step")
    require(proposed == mtp_steps * verify, f"native proposed draft count differs from NEXTN{mtp_steps}")
    require(correct <= proposed, "native accepted drafts exceed proposals")
    rate, length = correct / proposed, completion / verify
    for key, expected in (("spec_accept_rate", rate), ("spec_accept_length", length)):
        require(type(meta[key]) in (int, float) and math.isfinite(meta[key])
                and math.isclose(meta[key], expected, rel_tol=1e-6, abs_tol=1e-8),
                f"native MTP {key} disagrees with its counters")
    return {"verify_count": verify, "correct_drafts": correct, "proposed_drafts": proposed,
            "accepted_tokens_with_bonus": correct + verify, "delivered_completion_tokens": completion,
            "accept_rate": rate, "delivered_tokens_per_verify": length}


def aggregate_counters(records, *, mtp_steps=3):
    validate_mtp_steps(mtp_steps)
    require(bool(records), "missing native MTP request receipts")
    rows = [record["mtp_counters"] for record in records]
    require(all(row["proposed_drafts"] == mtp_steps * row["verify_count"] for row in rows),
            "aggregate request counters differ from explicit MTP steps")
    totals = {key: sum(row[key] for row in rows) for key in
              ("verify_count", "correct_drafts", "proposed_drafts", "accepted_tokens_with_bonus",
               "delivered_completion_tokens")}
    return totals | {"requests": len(rows), "accept_rate": totals["correct_drafts"] / totals["proposed_drafts"],
                     "delivered_tokens_per_verify": totals["delivered_completion_tokens"] / totals["verify_count"]}
