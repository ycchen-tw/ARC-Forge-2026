"""One-off L4/HiSparse gates; run with the deployed runtime's Python.

kernel: local attention TP2 geometry, packed planes/identity tables, graph reuse.
http: both DP routes, exact generated-id continuation, asymmetric session churn;
--cold-control adds two same-input cold controls per resume and a separate verdict.
JSON artifacts keep measured checks separate from untested server invariants.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import platform
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runpod_mtp import aggregate_counters, check_profile, request_counters


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def decode_json_response(path: str, status: int, raw: bytes):
    if not raw and 200 <= status < 300 and path in ("/close_session", "/abort_request"):
        return None
    return json.loads(raw)


def compare_output_prefix(left: dict, right: dict, steps: int = 8) -> dict:
    """Align top-20 by token ID, only while the preceding generated IDs agree."""
    require(len(left["ids"]) >= steps and len(right["ids"]) >= steps, "fewer than eight output IDs")
    first_difference = next((i for i in range(steps) if left["ids"][i] != right["ids"][i]), None)
    aligned_steps = steps if first_difference is None else first_difference + 1
    rows = []
    for step in range(aligned_steps):
        distributions = []
        for record in (left, right):
            top = record["meta_info"]["output_top_logprobs"]
            require(len(top) >= steps and len(top[step]) == 20, "missing eight steps of top-20 logprobs")
            probabilities = {}
            for entry in top[step]:
                require(len(entry) >= 2 and type(entry[1]) is int and entry[1] >= 0,
                        "invalid top-20 token ID")
                require(type(entry[0]) in (int, float) and math.isfinite(entry[0]) and entry[0] <= 1e-5,
                        "invalid top-20 log probability")
                require(entry[1] not in probabilities, "duplicate top-20 token ID")
                probabilities[entry[1]] = math.exp(entry[0])
            distributions.append(probabilities)
        a, b = distributions
        common = a.keys() & b.keys()
        deltas = {str(token): abs(a[token] - b[token]) for token in sorted(common)}
        rows.append({"step": step, "membership_equal": a.keys() == b.keys(),
                     "only_left_ids": sorted(a.keys() - b.keys()), "only_right_ids": sorted(b.keys() - a.keys()),
                     "absolute_probability_deltas": deltas,
                     "max_common_probability_delta": max(deltas.values(), default=0.),
                     "common_probability_l1_delta": sum(deltas.values()),
                     "retained_probability_mass": [sum(a.values()), sum(b.values())]})
    return {"first8_ids_equal": first_difference is None, "first_id_difference": first_difference,
            "identical_context_steps": aligned_steps,
            "top20_membership_equal": all(row["membership_equal"] for row in rows),
            "max_common_probability_delta": max(row["max_common_probability_delta"] for row in rows),
            "steps": rows}


def cold_equivalence(resumed: dict, cold: list[dict], probability_atol: float) -> dict:
    """A bounded output diagnostic; instability in repeated cold controls invalidates it."""
    verdict = {"verdict": "UNVERIFIED", "probability_atol": probability_atol,
               "scope": "first eight IDs and aligned top-20 probabilities; no full-output or quality claim"}
    if any(record["meta_info"]["cached_tokens"] != 0 for record in cold):
        return verdict | {"reason": "fresh cold control reused cached tokens"}
    try:
        noise = compare_output_prefix(cold[0], cold[1])
        comparisons = [compare_output_prefix(resumed, control) for control in cold]
    except (KeyError, TypeError, IndexError, RuntimeError) as exc:
        return verdict | {"reason": f"unusable output evidence: {type(exc).__name__}: {exc}"}
    verdict.update(cold_vs_cold=noise, resumed_vs_cold=comparisons)
    stable = (noise["first8_ids_equal"] and noise["top20_membership_equal"] and
              noise["max_common_probability_delta"] <= probability_atol)
    verdict["cold_controls_stable"] = stable
    if not stable:
        return verdict | {"reason": "cold controls are not stable at the recorded probability tolerance"}
    budget = probability_atol + noise["max_common_probability_delta"]
    equivalent = all(row["first8_ids_equal"] and row["top20_membership_equal"] and
                     row["max_common_probability_delta"] <= budget for row in comparisons)
    return verdict | {"verdict": "PASSED" if equivalent else "FAILED", "probability_budget": budget,
                      "reason": "resumed output within cold-control noise plus tolerance" if equivalent else
                      "resumed output differs from stable cold controls; batching/prefill paths also differ"}


def kernel_gate(args, result: dict) -> None:
    import torch
    from sglang.srt.layers.attention.qsa import sparse_attn

    require(torch.cuda.is_available(), "kernel gate requires CUDA; no skipped pass")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    props = torch.cuda.get_device_properties(device)
    result["runtime"] = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": props.name, "capability": [props.major, props.minor],
        "total_memory": props.total_memory,
        "source": str(Path(sparse_attn.__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(sparse_attn.__file__).read_bytes()).hexdigest(),
    }
    require((props.major, props.minor) == (8, 9), "this gate requires the requested SM89 hardware")
    direct = sparse_attn.qsa_sparse_decode_direct
    generator = torch.Generator(device=device).manual_seed(1838)
    n, dim, q_heads, kv_heads, width = 12000, 256, 12, 1, 2051
    lut = torch.tensor((0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6),
                       device=device, dtype=torch.float32)
    result["geometry"] = {"q_heads": q_heads, "kv_heads": kv_heads, "dim": dim, "width": width}
    result["checks"] = []

    def unpack(packed, sf, scale):
        values = torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1).flatten(-2)
        return values * sf.float().repeat_interleave(16, -1) * scale

    for kind in ("nvfp4", "fp8"):
        if kind == "nvfp4":
            k, v = [torch.randint(0, 256, (n, kv_heads, dim // 2), device=device,
                                  dtype=torch.uint8, generator=generator) for _ in range(2)]
            ks, vs = [(torch.rand(n, kv_heads, dim // 16, device=device, generator=generator) * .4 + .05)
                      .to(torch.float8_e4m3fn) for _ in range(2)]
            kg, vg = .75, 1.25
        else:
            k, v = [(torch.randn(n, kv_heads, dim, device=device, generator=generator) * .5)
                    .to(torch.float8_e4m3fn) for _ in range(2)]
            ks = vs = None
            kg = vg = 1.
        for rows in (4, 32):
            q = (torch.randn(rows, q_heads, dim, device=device, generator=generator) * .5).bfloat16()
            table = torch.randperm(n, generator=generator, device=device).int().view(1, -1)
            table_rows = torch.zeros(rows, device=device, dtype=torch.int32)
            lens = torch.full((rows,), n, device=device, dtype=torch.int32)
            indices = torch.stack([torch.randperm(n, device=device, generator=generator)[:width]
                                   for _ in range(rows)]).int().contiguous()
            indices[0] = -1  # CUDA graph padding: output must be zero, never NaN.
            indices[1, 17:21] = -1  # valid tokens after holes test the full validity mask.
            indices[2, -1] = n  # excluded by the length bound.

            def reference():
                expected = torch.zeros_like(q, dtype=torch.float32)
                for row in range(rows):
                    pos = indices[row].long()
                    take = (pos >= 0) & (pos < lens[row])
                    if not bool(take.any()):
                        continue
                    slots = table[0, pos[take]].long()
                    kk = k[slots].float() if ks is None else unpack(k[slots], ks[slots], kg)
                    vv = v[slots].float() if vs is None else unpack(v[slots], vs[slots], vg)
                    scores = torch.einsum("hd,td->ht", q[row].float(), kk[:, 0]) * dim ** -.5
                    expected[row] = torch.softmax(scores, -1) @ vv[:, 0]
                return expected

            def call(decode, chunk):
                return direct(q, k, v, ks, vs, table, table_rows, indices, lens,
                              dim ** -.5, kg, vg, decode=decode, chunk=chunk,
                              block_n=64, warps=4, stages=2)

            ref = reference()
            for chunk in (None, 512):
                outs = [call(decode, chunk) for decode in ((0, 1) if ks is not None else (0,))]
                require(all(torch.equal(outs[0], out) for out in outs[1:]), "arithmetic/bit FP4 decode differs")
                error = (outs[0].float() - ref).norm(dim=(1, 2)) / ref.norm(dim=(1, 2)).clamp_min(1e-20)
                maximum = float(error.max())
                require(maximum < .01, f"{kind} rows={rows} chunk={chunk}: relative error {maximum}")
                require(torch.isfinite(outs[0]).all().item(), "non-finite output")
                require(torch.count_nonzero(outs[0][0]).item() == 0, "padded row was read")
                result["checks"].append({"kind": kind, "rows": rows, "chunk": chunk, "max_rel_l2": maximum})

            # HiSparse: logical indices were expanded into scattered device rows;
            # this identity table must produce exactly the same selected K/V.
            identity = torch.arange(n, device=device, dtype=torch.int32).view(1, -1)
            hot_rows = torch.where((indices >= 0) & (indices < lens[:, None]),
                                   table[0, indices.clamp(0, n - 1).long()], -1).int().contiguous()
            eager = call(0, 512)
            def identity_call():
                return direct(q, k, v, ks, vs, identity, table_rows, hot_rows, lens,
                              dim ** -.5, kg, vg, decode=0, chunk=512, block_n=64, warps=4, stages=2)

            via_identity = identity_call()
            require(torch.equal(eager, via_identity), "HiSparse hot rows/identity mapping differs")
            identity_call()  # warm allocations and both kernel specializations
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                captured = identity_call()
            torch.cuda.current_stream(device).wait_stream(stream)
            graph.replay()
            torch.cuda.synchronize(device)
            require(torch.equal(eager, captured), "initial graph replay differs")
            # Persistent address tables and planes are reused after slot changes.
            # A same-input replay alone cannot detect stale metadata or stale KV.
            table.copy_(torch.roll(table, 127, 1))
            hot_rows.copy_(torch.where((indices >= 0) & (indices < lens[:, None]),
                                      table[0, indices.clamp(0, n - 1).long()], -1).int())
            q.mul_(.75)
            if ks is not None:
                v.bitwise_xor_(51)
                vs.copy_((vs.float() * .75).to(vs.dtype))
            else:
                v.copy_((v.float() * -.5).to(v.dtype))
            expected = call(0, 512)
            require(not torch.equal(eager, expected), "mutation failed to change the reference output")
            graph.replay()
            torch.cuda.synchronize(device)
            require(torch.equal(expected, captured), "graph replay used stale table/query/KV")
            result["checks"].append({"kind": kind, "rows": rows, "identity_bitwise": True,
                                     "graph_initial_bitwise": True, "graph_mutation_bitwise": True})
    result["untested"] = ["coordinator swap-in/lifecycle", "distributed snapshot consensus", "full-model quality"]


def http_gate(args, result: dict) -> None:
    mtp_steps = vars(args).get("mtp_steps", 3)
    require(args.mtp or mtp_steps == 3, "--mtp-steps 1 requires --mtp")
    prompts = json.loads(args.prompts.read_text())
    ids = prompts[args.prompt_key] if isinstance(prompts, dict) else prompts
    require(isinstance(ids, list) and all(isinstance(x, int) and x >= 0 for x in ids), "invalid token fixture")
    require(len(ids) >= args.prompt_tokens, "fixture shorter than requested prompt")
    ids = ids[:args.prompt_tokens]
    require(args.prompt_tokens > 512 and args.new_tokens > args.snapshot_interval, "gate must cross snapshots")
    result["fixture"] = {"source": str(args.prompts), "ids_sha256": digest(ids),
                         "prompt_tokens": len(ids), "new_tokens": args.new_tokens,
                         "sessions_per_dp": args.sessions_per_dp, "kv_dtype": args.kv_dtype}
    if args.mtp:
        result["fixture"]["speculative_profile"] = f"NEXTN{mtp_steps}/verify{mtp_steps + 1}/ReplaySSM-spec/cache16"
    result["calls"] = []
    result["output_equivalence"] = {
        "verdict": "PENDING" if args.cold_control else "NOT_RUN", "requested": args.cold_control,
        "reason": "cold controls requested; complete paired evidence is pending" if args.cold_control else
                  "--cold-control was not requested",
    }
    trace = args.output.with_suffix(".jsonl")
    trace_lock = threading.Lock()
    prefix = "gate-" + uuid.uuid4().hex[:12]
    sessions = [(dp, f"{prefix}-dp{dp}-s{index}") for dp in (0, 1)
                for index in range(args.sessions_per_dp)]
    owned_sessions = list(sessions)

    def request(path, body=None):
        payload = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.base_url.rstrip("/") + path, data=payload,
                                     headers={"Content-Type": "application/json"})
        receipt = {"path": path, "method": req.get_method(), "request": body,
                   "request_raw": None if payload is None else payload.decode(),
                   "started_monotonic_s": time.monotonic()}
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                raw = response.read()
                receipt.update(http_status=response.status, response_raw=raw.decode())
            return decode_json_response(path, receipt["http_status"], raw)
        except urllib.error.HTTPError as exc:
            receipt.update(http_status=exc.code, response_raw=exc.read().decode(errors="replace"),
                           error=f"{type(exc).__name__}: {exc}")
            raise
        except Exception as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receipt["elapsed_s"] = time.monotonic() - receipt["started_monotonic_s"]
            with trace_lock, trace.open("a") as output:
                output.write(json.dumps(receipt) + "\n")

    def generate(tag, dp, sid, input_ids, new=None, min_cached=None):
        count = args.new_tokens if new is None else new
        body = {"rid": f"{sid}-{tag}", "session_id": sid, "routed_dp_rank": dp,
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": count, "ignore_eos": True},
                "return_logprob": True, "top_logprobs_num": 20}
        started = time.monotonic()
        response = request("/generate", body)
        meta = response["meta_info"]
        steps = meta["output_token_logprobs"]
        record = {"tag": tag, "dp": dp, "sid": sid, "input_sha256": digest(input_ids),
                  "input_tokens": len(input_ids), "latency_s": time.monotonic() - started,
                  "ids": [step[1] for step in steps], "meta_info": meta, "text": response.get("text")}
        require(meta["prompt_tokens"] == len(input_ids), f"{tag}: wrong prompt count")
        require(meta["completion_tokens"] == count and len(record["ids"]) == count, f"{tag}: incomplete output")
        require(meta["dp_rank"] == dp, f"{tag}: wrong or missing DP receipt")
        require(meta["num_retractions"] == 0, f"{tag}: session request retracted")
        require(meta["finish_reason"]["type"] == "length", f"{tag}: unexpected finish reason")
        if min_cached is not None:
            require(meta["cached_tokens"] >= min_cached, f"{tag}: resume cached {meta['cached_tokens']} < {min_cached}")
        if args.mtp:
            record["mtp_counters"] = request_counters(meta, mtp_steps=mtp_steps)
        return record

    result["server_info"] = request("/server_info")
    info = result["server_info"]
    expected_config = {"tp_size": 4, "ep_size": 4, "dp_size": 2, "enable_dp_attention": True,
                       "speculative_algorithm": None, "enable_linear_replayssm": False,
                       "enable_linear_replayssm_spec": False, "mamba_ssm_dtype": "float16",
                       "kv_cache_dtype": args.kv_dtype}
    if args.mtp:
        expected_config.update(check_profile(info, mtp_steps=mtp_steps))
    if args.arm == "hisparse":
        expected_config.update(enable_hisparse=True, radix_cache_backend="qsa_hisparse_session")
        config = info["hisparse_config"]
        config = json.loads(config) if isinstance(config, str) else config
        require(config["qsa_snapshot_interval"] == args.snapshot_interval, "snapshot interval differs")
        require(config["qsa_session_resume"] is True, "session resume disabled")
    for key, value in expected_config.items():
        require(info[key] == value, f"server {key}={info[key]!r}, expected {value!r}")
    result["startup_config_checked"] = expected_config
    first = {}
    continuations = {}
    try:
        # Serial fresh requests establish each route; unequal contexts keep the
        # two independent DPs from accidentally polling matching-length queues.
        for dp, sid in sessions:
            base = ids if dp == 0 else ids[:len(ids) - 257]
            rec = generate("initial", dp, sid, base)
            require(rec["meta_info"]["cached_tokens"] == 0, "unique session reused an unrelated prefix")
            first[sid] = (base, rec)
            result["calls"].append(rec)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sessions)) as executor:
            pending = []
            for dp, sid in sessions:
                base, previous = first[sid]
                continuation = base + previous["ids"] + ids[:128 + dp * 193]
                continuations[sid] = continuation
                pending.append(executor.submit(generate, "resume", dp, sid, continuation,
                                               min_cached=len(base) - args.snapshot_interval))
            for future in pending:
                result["calls"].append(future.result())
        if args.cold_control:
            pairs = []
            # Every resume has finished before serial controls start. Each control
            # resends exactly that request's input, including its actual generated IDs.
            for resumed in [record for record in result["calls"] if record["tag"] == "resume"]:
                dp, sid = resumed["dp"], resumed["sid"]
                controls = []
                for repeat in (1, 2):
                    cold_sid = f"{sid}-cold{repeat}"
                    owned_sessions.append((dp, cold_sid))
                    control = generate(f"cold-control-{repeat}", dp, cold_sid, continuations[sid], new=8)
                    result["calls"].append(control)
                    controls.append(control)
                    request("/close_session", {"session_id": cold_sid})
                    owned_sessions.remove((dp, cold_sid))
                comparison = cold_equivalence(resumed, controls, args.cold_prob_atol)
                comparison.update(dp=dp, resumed_sid=sid, cold_sids=[record["sid"] for record in controls],
                                  input_sha256=resumed["input_sha256"],
                                  cached_tokens=[resumed["meta_info"]["cached_tokens"],
                                                 *[record["meta_info"]["cached_tokens"] for record in controls]])
                pairs.append(comparison)
            verdicts = [pair["verdict"] for pair in pairs]
            result["output_equivalence"] = {
                "requested": True,
                "verdict": "FAILED" if "FAILED" in verdicts else
                           "UNVERIFIED" if "UNVERIFIED" in verdicts else "PASSED",
                "pairs": pairs, "probability_atol": args.cold_prob_atol,
                "context": {"resumed_dispatch": "concurrent requests", "cold_dispatch": "serial after all resumes",
                            "prefill": "snapshot suffix for resume; complete identical input for cold controls",
                            "scheduler_batch_shapes": "not observed",
                            "chunked_prefill_size": info.get("chunked_prefill_size")},
                "scope": "first eight IDs and top-20 on identical prediction contexts; no quality claim",
            }
        # Corrupt a token below every valid snapshot; exact token validation must
        # recompute instead of restoring a state belonging to the older prefix.
        for dp, sid in sessions:
            base, previous = first[sid]
            changed = base + previous["ids"] + ids[:128]
            changed[1] = next(token for token in ids if token != changed[1])
            rec = generate("mismatch", dp, sid, changed, new=8)
            require(rec["meta_info"]["cached_tokens"] == 0, "mismatching prefix reused stale state")
            result["calls"].append(rec)
        # Closing then reusing the same session name must create a new generation.
        for dp, sid in sessions:
            request("/close_session", {"session_id": sid})
            rec = generate("after-close", dp, sid, first[sid][0], new=8)
            require(rec["meta_info"]["cached_tokens"] == 0, "closed session retained cached generation")
            result["calls"].append(rec)
    finally:
        errors = []
        for _, sid in owned_sessions:
            try:
                request("/close_session", {"session_id": sid})
            except Exception as exc:
                errors.append(f"{sid}: {type(exc).__name__}: {exc}")
        result["cleanup_errors"] = errors
    require(not result["cleanup_errors"], "session cleanup failed")
    result["server_info_after_close"] = request("/server_info")
    if args.mtp:
        result["mtp_acceptance"] = aggregate_counters(result["calls"], mtp_steps=mtp_steps)
        result["mtp_acceptance_by_dp"] = {
            str(dp): aggregate_counters([record for record in result["calls"] if record["dp"] == dp], mtp_steps=mtp_steps)
            for dp in (0, 1)}
    result["metadata_lifecycle_verdict"] = "PASSED"
    if args.baseline is not None:
        reference = json.loads(args.baseline.read_text())
        require(reference["fixture"] == result["fixture"], "baseline fixture/config differs")
        require(len(reference["calls"]) == len(result["calls"]), "baseline call roster differs")
        comparisons = []
        for before, after in zip(reference["calls"], result["calls"]):
            require((before["tag"], before["dp"], before["input_sha256"]) ==
                    (after["tag"], after["dp"], after["input_sha256"]), "baseline inputs differ")
            equal_ids = before["ids"] == after["ids"]
            equal_top = before["meta_info"]["output_top_logprobs"] == after["meta_info"]["output_top_logprobs"]
            comparisons.append({"tag": after["tag"], "dp": after["dp"], "ids_equal": equal_ids,
                                "top20_equal": equal_top})
        result["baseline_comparisons"] = comparisons
        require(all(row["ids_equal"] and row["top20_equal"] for row in comparisons), "fixed-prefix baseline differs")
    result["untested"] = ["abort/LRU lifecycle", "per-rank idle counters", "long-context capacity",
                          "semantic quality", "throughput"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    kernel = commands.add_parser("kernel")
    kernel.add_argument("--device", type=int, default=0)
    kernel.add_argument("--output", type=Path, required=True)
    http = commands.add_parser("http")
    http.add_argument("--base-url", required=True)
    http.add_argument("--prompts", type=Path, required=True)
    http.add_argument("--prompt-key", default="p9001")
    http.add_argument("--prompt-tokens", type=int, default=8192)
    http.add_argument("--new-tokens", type=int, default=320)
    http.add_argument("--snapshot-interval", type=int, default=256)
    http.add_argument("--arm", choices=("hisparse", "resident"), default="hisparse")
    http.add_argument("--mtp", action="store_true", help="require the opt-in h29 NEXTN3 profile and native acceptance counters")
    http.add_argument("--mtp-steps", type=int, choices=(1, 3), default=3)
    http.add_argument("--kv-dtype", choices=("nvfp4", "fp8_e4m3"), default="nvfp4")
    http.add_argument("--sessions-per-dp", type=int, default=2)
    http.add_argument("--timeout", type=float, default=600)
    http.add_argument("--baseline", type=Path)
    http.add_argument("--cold-control", action="store_true", help="compare each resume with two exact-input cold controls")
    http.add_argument("--cold-prob-atol", type=float, default=1e-5,
                      help="absolute probability tolerance for cold stability and resume comparison")
    http.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), "refusing to overwrite gate output")
    if args.mode == "http":
        require(not args.mtp or args.arm == "hisparse", "--mtp currently requires the HiSparse h29 arm")
        require(args.mtp or args.mtp_steps == 3, "--mtp-steps 1 requires --mtp")
        require(not args.output.with_suffix(".jsonl").exists(), "refusing to append an older request trace")
        require(args.sessions_per_dp >= 1, "sessions-per-dp must be positive")
        require(math.isfinite(args.cold_prob_atol) and args.cold_prob_atol >= 0, "invalid cold probability tolerance")
    result = {"mode": args.mode, "platform": platform.platform(), "status": "running"}
    started = time.monotonic()
    try:
        (kernel_gate if args.mode == "kernel" else http_gate)(args, result)
        verdict = result.get("output_equivalence", {}).get("verdict", "NOT_RUN")
        result["status"] = {"PENDING": "unverified", "UNVERIFIED": "unverified", "FAILED": "failed"}.get(verdict, "passed")
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in ("mode", "status", "elapsed_s")}), flush=True)
    return {"passed": 0, "failed": 1, "unverified": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
