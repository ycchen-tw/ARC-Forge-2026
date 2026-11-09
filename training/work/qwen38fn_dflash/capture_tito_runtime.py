"""Capture server for recorded TITO segments on the sglang-arc-2 runtime, images included.

Server half of capture_tito.py (capture_runtime.py is the text-only one for the old qwen4 runtime). The capture server
is the recording server -- the ledger instance's server.json: its SGLANG_* env, its launch command, its model files --
minus everything that serves traffic (speculative decoding, sessions, the ledger, parsers) plus capture isolation: one
running request, no radix, no CUDA graph, no overlap. A recorded flag that is in neither list raises, so a new flag
gets a decision instead of a silent copy.

No installed file changes. ``main()`` installs the hooks in the server process; every process SGLang spawns re-imports
this file as ``__mp_main__`` and installs them again:

- ``Qwen4ExpForConditionalGeneration.forward``: while ``$TITO_CAPTURE_ROOT/active.json`` names a job, TP rank 0 writes
  each extend chunk's input ids as the model read them (the multimodal embed rewrites media positions in place: to
  vocab_size - 1, or to <|image_pad|> / <|video_pad|> under ``--ple-mm-placeholder-ids``; ``ple_placeholder_ids`` says
  which), positions, M-RoPE positions and the final pre-collapse GR -- the tensor the DFlash drafter reads
  (arc_dflash/worker.py) -- to ``<root>/<job>/parts/<prefix>.safetensors``.
- ``TokenizerManager._tokenize_one_request``: ``image_data`` dicts ``{"url", "detail"}`` become ``ImageData``, as the
  OpenAI chat/Responses paths make them (the detail only enters the image's radix pad value; sent for fidelity).

Media: the request sends each image as one placeholder and each video as <|vision_start|><|video_pad|><|vision_end|>; the
server expands them from the media and keeps every other id. For a video that needs sglang-arc-2 4b57f3c45 or later
(``base_processor.splice_timestamped_media_blocks``; upstream re-tokenizes such a prompt and its ids drift), which
``install()`` checks by importing it.

    TITO_CAPTURE_ROOT=<queue> python capture_tito_runtime.py --recorded <instance>/server.json --port <port> \
        [--target-profile <server.json of the model to capture with>] [--context-length <n>] \
        [--model-path <a copy of the target model>] [--tp <n>]
    TITO_CAPTURE_ROOT=<queue> python capture_tito_runtime.py --target-profile <profile.json> --port <port> ...

``--target-profile`` captures with another deployment than the recording's -- e.g. a ledger recorded on the FP8 model
replayed through the REAP320 NVFP4 target the drafter will serve with. The profile is any server.json of that
deployment (another ledger's will do): its model, flags and runtime env replace the recording's; the instance, the
served name and the segments stay the recording's. check_target refuses a profile that would read a segment
differently: other tokenizer, template or media processors, special token ids, vocabulary, GR width or vision patching.

A profile whose env has ``ARC_INT3_DEPLOY=1`` is the packed gEAN320 INT3 deployment: ``install_int3`` sets up
``work/qwen38fn_eval/int3_deploy_runtime.py``'s plugin as its ``launch`` does (here, and through its sitecustomize in
every interpreter SGLang spawns), its ``ARC_INT3_ACT`` / ``ARC_PLE_INT5`` come with the profile's env. The fork must be
one that serves the package (lock ``b200-rl-int3``).

``--target-profile`` without ``--recorded`` is profile mode: one server for every source of a data plan
(capture_plan.py -- TITO segments of any ledger and plain token rows). Its served name is ``profile_served_name``:
the profile's instance and the sha256 of the profile file, so a producer can tell it is the server of the very
profile its plan was built for. capture_plan.py runs check_target for each ledger instance before it sends a segment.
"""

import argparse
import functools
import hashlib
import inspect
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT_ENV = "TITO_CAPTURE_ROOT"
GR_SHAPE = (4, 2560)

# Recorded flags copied as they are: they decide the target's numerics, how its inputs are processed, or how it loads.
KEEP_FLAGS = {
    "--model-path", "--tokenizer-path", "--tp", "--tp-size", "--ep", "--ep-size", "--context-length",
    "--mem-fraction-static", "--chunked-prefill-size", "--linear-attn-prefill-backend", "--linear-attn-decode-backend",
    "--linear-attn-verify-backend", "--mamba-ssm-dtype", "--kv-cache-dtype", "--quantization", "--mm-feature-transport",
    "--image-processor-backend", "--mm-process-config", "--mm-literal-image-sentinel", "--no-mm-literal-image-sentinel",
    "--attention-backend", "--moe-runner-backend", "--trust-remote-code", "--dtype", "--page-size",
    "--ple-offload-embedding", "--json-model-override-args", "--model-loader-extra-config", "--random-seed",
    "--ple-mm-placeholder-ids", "--fp8-gemm-backend", "--disable-shared-experts-fusion",
}
# Recorded flags dropped: they only serve traffic (scheduling, caching across requests, speculative decoding, parsers,
# metrics, the ledger), or the capture sets its own value below. HiSparse keeps KV off-GPU between requests; one
# isolated prefill has nothing to keep.
DROP_FLAGS = {
    "--served-model-name", "--host", "--port", "--max-running-requests", "--max-mamba-cache-size",
    "--cuda-graph-max-bs-decode", "--cuda-graph-max-bs", "--cuda-graph-bs", "--reasoning-parser", "--tool-call-parser",
    "--sampling-defaults", "--enable-cache-report", "--enable-metrics", "--decode-log-interval",
    "--prefill-max-requests", "--enable-prefill-delayer", "--prefill-delayer-queue-min-ratio",
    "--prefill-delayer-max-delay-ms", "--prefill-delayer-max-delay-passes", "--enable-linear-replayssm-spec",
    "--disable-radix-cache", "--enable-session-radix-cache", "--radix-cache-backend", "--mamba-radix-cache-strategy",
    "--mamba-max-states-per-path", "--enable-hisparse", "--hisparse-config", "--enable-return-routed-experts",
    "--session-replay", "--session-ledger-dir", "--session-ledger-all", "--session-ledger-logprobs",
    "--session-ledger-routing", "--session-ledger-max-bytes", "--session-replay-empty-think", "--watchdog-timeout",
    "--weight-version", "--stream-interval", "--scheduler-recv-interval",
}
DROP_PREFIXES = ("--speculative-",)
# Files and config.json fields that decide how a segment's ids and media become the target's input. A target profile
# must agree with the recording on all of them; the rest of config.json is the model itself (experts, quantization).
INPUT_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json",
               "video_preprocessor_config.json")
TEXT_FIELDS = ("vocab_size", "hidden_size", "hc_count")
VISION_LABELS = {"dtype", "model_type"}  # differ between checkpoints of the same vision tower

# Recorded env (the ledger keeps every ARC_* and SGLANG_*) is the recording's runtime configuration -- QSA/KV kernel
# switches included -- except where this machine keeps its files and the run's own identity.
ENV_SKIP = {"SGLANG_RUN_ID", "ARC_ROOT", "ARC_DATA", "ARC_MODELS", "ARC_RUNS", "ARC_RAM_RUNS", "ARC_SGLANG_FORK",
            "ARC_SGLANG_SRC", "ARC_ARC3_CORPUS"}


def parse_flags(argv):
    """``[(flag, [values])]`` from an argv of ``--flag [value ...]`` groups, in order."""
    out = []
    for token in argv:
        if token.startswith("--"):
            out.append((token, []))
        elif not out:
            raise ValueError(f"recorded launch command starts with a value, not a flag: {token!r}")
        else:
            out[-1][1].append(token)
    return out


def recorded_flags(launch_command):
    """SGLang records ``" ".join(argv)`` (server_args.py), so split on single spaces, not shlex: a JSON value keeps its
    quotes, and a value that itself held spaces comes back as several tokens that are joined again below. Every kept
    flag takes one value or none."""
    return [(flag, [" ".join(values)] if values else []) for flag, values in parse_flags(launch_command.split(" "))]


def ledger_served_name(instance):
    """The served name of the capture server for one ledger instance (capture_tito.py checks it)."""
    return f"tito-capture.{instance}"


def profile_served_name(profile_path):
    """The served name of a profile-mode server: the profile's instance and the sha256 of the profile file."""
    raw = Path(profile_path).read_bytes()
    return f"dflash-capture.{json.loads(raw)['instance']}.{hashlib.sha256(raw).hexdigest()[:16]}"


def capture_argv(recorded, port, *, instance=None, served_name=None, model_path=None, tp=None, context_length=None,
                 ple_mm_placeholder_ids=False):
    """The capture server's argv: the recorded launch command (the recording's, or a target profile's), filtered,
    plus capture isolation. The server is named served_name, else ledger_served_name(instance) (default instance:
    the launch command's own). ``ple_mm_placeholder_ids`` adds --ple-mm-placeholder-ids when the recording ran
    without it (the PLE reads the media placeholder tokens, not vocab_size - 1)."""
    if served_name is not None and instance is not None:
        raise ValueError("pass instance or served_name, not both")
    argv = []
    context_seen = False
    for flag, values in recorded_flags(recorded["launch_command"]):
        if flag in DROP_FLAGS or flag.startswith(DROP_PREFIXES):
            continue
        if flag not in KEEP_FLAGS:
            raise ValueError(f"recorded flag {flag} {values} is neither kept nor dropped; decide whether it changes "
                             "the target's numerics or input processing and add it to KEEP_FLAGS or DROP_FLAGS")
        if flag == "--model-path" and model_path is not None:
            values = [model_path]
        if flag in ("--tp", "--tp-size", "--ep", "--ep-size") and tp is not None:
            values = [str(tp)]
        if flag == "--context-length":
            context_seen = True
            if context_length is not None:
                values = [str(context_length)]
        argv += [flag, *values]
    if context_length is not None and not context_seen:
        argv += ["--context-length", str(context_length)]
    if ple_mm_placeholder_ids and "--ple-mm-placeholder-ids" not in argv:
        argv += ["--ple-mm-placeholder-ids"]
    if served_name is None:
        served_name = ledger_served_name(recorded["instance"] if instance is None else instance)
    return argv + [
        "--served-model-name", served_name, "--host", "127.0.0.1", "--port", str(port),
        "--max-running-requests", "1", "--max-mamba-cache-size", "4", "--disable-radix-cache",
        "--disable-cuda-graph", "--disable-prefill-cuda-graph", "--disable-overlap-schedule", "--skip-server-warmup",
    ]


INT3_DIR = Path(__file__).resolve().parents[1] / "qwen38fn_eval"


def install_int3(flags):
    """The target is the packed INT3 package: what ``int3_deploy_runtime.py launch`` does before it runs SGLang -- audit
    the package, put its sitecustomize first on PYTHONPATH (each spawned interpreter installs the plugin at startup,
    before it imports this file as __mp_main__) and install the plugin in this process."""
    if os.environ.get("ARC_INT3_DEPLOY") != "1":
        raise RuntimeError("install_int3 without ARC_INT3_DEPLOY=1 in the profile env")
    if flags.get("--moe-runner-backend") != ["humming"]:
        raise ValueError("an INT3 target profile needs --moe-runner-backend humming")
    if os.environ.get("ARC_PINNED_HOST_REGISTER", "0") != "0":
        raise ValueError("ARC_PINNED_HOST_REGISTER=1 needs the allocator setup of int3_deploy_runtime.py launch")
    sys.path.insert(0, str(INT3_DIR))
    import int3_deploy_runtime

    print(json.dumps(int3_deploy_runtime.audit_model(Path(flags["--model-path"][0]))), file=sys.stderr, flush=True)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(INT3_DIR / "int3_deploy_patch"), str(INT3_DIR), os.environ.get("PYTHONPATH", "")])
    int3_deploy_runtime.install()


def recorded_env(recorded):
    return {k: v for k, v in recorded["env"].items() if k.startswith(("SGLANG_", "ARC_")) and k not in ENV_SKIP}


def check_model_files(recorded, model_path):
    """The model's small files (config, tokenizer, processor, index) must be the recorded ones."""
    want = recorded["model_files_sha256"]
    if not want:
        raise ValueError("server.json records no model file digests")
    for name, digest in sorted(want.items()):
        got = hashlib.sha256((Path(model_path) / name).read_bytes()).hexdigest()
        if got != digest:
            raise ValueError(f"{model_path}/{name}: sha256 {got} != recorded {digest}")


def input_spec(config):
    """What in a config.json decides how ids and media become the model's input."""
    text, vision = config["text_config"], config["vision_config"]
    return {
        "token_ids": {k: v for k, v in config.items() if k.endswith("_token_id")},
        "text": {k: text[k] for k in TEXT_FIELDS},
        "vision": {k: v for k, v in vision.items() if k not in VISION_LABELS},
    }


def check_target(recorded, target, target_model_path):
    """A target profile must read every segment exactly as the recording did. Raises on the first difference."""
    have, want = target["model_files_sha256"], recorded["model_files_sha256"]
    for name in INPUT_FILES:
        if have[name] != want[name]:
            raise ValueError(f"target {name} sha256 {have[name]} != the recording's {want[name]}")
    if "DFLASH_LOCAL_CONFIGS" in os.environ:
        from local_storage import require_local

        config = require_local(Path(os.environ["DFLASH_LOCAL_CONFIGS"]) / (want["config.json"] + ".json"))
        raw = config.read_bytes()
        if hashlib.sha256(raw).hexdigest() != want["config.json"]:
            raise ValueError("local recorded config does not match the ledger SHA256")
    else:
        recorded_model = Path(recorded["server_args"]["model_path"])
        check_model_files({"model_files_sha256": {"config.json": want["config.json"]}}, recorded_model)
        raw = (recorded_model / "config.json").read_bytes()
    recorded_spec = input_spec(json.loads(raw))
    target_spec = input_spec(json.loads((Path(target_model_path) / "config.json").read_text()))
    if recorded_spec != target_spec:
        diff = {k: (recorded_spec[k], target_spec[k]) for k in recorded_spec if recorded_spec[k] != target_spec[k]}
        raise ValueError(f"target config reads inputs differently from the recording's: {diff}")


def as_image_data(items, image_data_cls):
    """``{"url", "detail"}`` dicts -> ImageData. Anything else in a capture request is a driver bug."""
    if not isinstance(items, list):
        raise TypeError(f"capture image_data must be a list, got {type(items).__name__}")
    out = []
    for item in items:
        if not isinstance(item, dict) or "url" not in item or set(item) - {"url", "detail"}:
            raise ValueError(f"capture image_data items are {{url, detail}} dicts, got {str(item)[:120]}")
        out.append(image_data_cls(url=item["url"], detail=item.get("detail")))
    return out


def install():
    import torch
    from safetensors.torch import save_file
    from sglang.srt.distributed import get_tensor_model_parallel_rank
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
    from sglang.srt.multimodal.processors.base_processor import splice_timestamped_media_blocks  # noqa: F401
    from sglang.srt.utils import ImageData

    root = Path(os.environ[ROOT_ENV])
    cls = Qwen4ExpForConditionalGeneration
    original = cls.forward
    if "hc_hidden_states = self.model.last_hc_hidden_states" not in inspect.getsource(original):
        raise RuntimeError("unrecognized final-GR handoff in Qwen4ExpForConditionalGeneration.forward; re-audit")

    @functools.wraps(original)
    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        output = original(self, input_ids, positions, forward_batch, *args, **kwargs)
        active = root / "active.json"
        if get_tensor_model_parallel_rank() != 0 or not active.is_file():
            return output
        job = json.loads(active.read_text())
        mode = forward_batch.forward_mode
        if mode != ForwardMode.EXTEND:
            raise RuntimeError(f"capture saw a {mode} forward; it replays prefill only")
        if forward_batch.batch_size != 1:
            raise RuntimeError(f"capture needs one request per forward, got {forward_batch.batch_size}")
        n = int(forward_batch.extend_seq_lens_cpu[0])
        prefix = int(forward_batch.extend_prefix_lens_cpu[0])
        ids = forward_batch.input_ids
        if tuple(ids.shape) != (n,) or tuple(positions.shape) != (n,):
            raise RuntimeError(f"chunk of {n}: ids {tuple(ids.shape)}, positions {tuple(positions.shape)}")
        h = self.model.last_hc_hidden_states
        width = GR_SHAPE[0] * GR_SHAPE[1]
        if h is None or tuple(h.shape) != (n, width) or h.dtype != torch.bfloat16:
            got = None if h is None else (tuple(h.shape), h.dtype)
            raise RuntimeError(f"final GR {got} != ({n}, {width}) bf16")
        # mm_utils.embed_mm_inputs: a model without the attribute (a fork before --ple-mm-placeholder-ids) clamps
        placeholder = getattr(self, "mm_media_token_ids", None) is not None
        tensors = {
            "ple_placeholder_ids": torch.tensor([placeholder]),
            "input_ids": ids.detach().cpu().contiguous(),
            "positions": positions.detach().cpu().contiguous(),
            "target_gr": h.detach().view(n, *GR_SHAPE).cpu().contiguous(),
        }
        mrope = forward_batch.mrope_positions
        if mrope is not None:
            if tuple(mrope.shape) != (3, n):
                raise RuntimeError(f"M-RoPE positions {tuple(mrope.shape)} for a chunk of {n}")
            tensors["mrope_positions"] = mrope.detach().cpu().contiguous()
        directory = root / job["job_id"] / "parts"
        destination = directory / f"{prefix:012d}.safetensors"
        if destination.exists():
            raise RuntimeError(f"{destination} already written: a chunk was replayed")
        temp = destination.with_suffix(".partial")
        save_file(tensors, str(temp))
        temp.rename(destination)
        return output

    cls.forward = forward

    tokenize = TokenizerManager._tokenize_one_request

    @functools.wraps(tokenize)
    async def tokenize_one(self, obj):
        if obj.image_data is not None:
            obj.image_data = as_image_data(obj.image_data, ImageData)
        return await tokenize(self, obj)

    TokenizerManager._tokenize_one_request = tokenize_one



def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--recorded", type=Path, help="the ledger instance's server.json (omit for profile mode)")
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--target-profile", type=Path,
                   help="server.json of the deployment to capture with instead of the recording's (check_target); "
                        "alone: profile mode, one server for every source of a data plan")
    p.add_argument("--context-length", type=int, help="override the target's context length (longer segments skip)")
    p.add_argument("--model-path", help="a copy of the target model (checked against its recorded file digests)")
    p.add_argument("--tp", type=int, help="override the recorded TP/EP size (numerics shift by reduction order)")
    p.add_argument("--ple-mm-placeholder-ids", action="store_true",
                   help="serve with --ple-mm-placeholder-ids even if the recording did not (the PLE reads "
                        "<|image_pad|> / <|video_pad|> at media positions, as HF and vLLM do)")
    p.add_argument("--vlm-cache-mb", type=int, required=True,
                   help="vision embedding cache; large enough that no image is re-encoded between chunks")
    args = p.parse_args()
    if args.recorded is None and args.target_profile is None:
        raise ValueError("--recorded (a ledger instance's capture server) or --target-profile (profile mode) needed")
    target = None if args.target_profile is None else json.loads(args.target_profile.read_text())
    recorded = target if args.recorded is None else json.loads(args.recorded.read_text())
    target = recorded if target is None else target
    if not Path(os.environ[ROOT_ENV]).is_dir():
        raise FileNotFoundError(f"{ROOT_ENV}={os.environ[ROOT_ENV]} is not a directory")
    if args.recorded is None:
        names = {"served_name": profile_served_name(args.target_profile)}
    else:
        names = {"instance": recorded["instance"]}
    argv = capture_argv(target, args.port, model_path=args.model_path, tp=args.tp, context_length=args.context_length,
                        ple_mm_placeholder_ids=args.ple_mm_placeholder_ids, **names)
    flags = dict((f, v) for f, v in parse_flags(argv))
    check_model_files(target, flags["--model-path"][0])
    if args.recorded is not None and target is not recorded:
        check_target(recorded, target, flags["--model-path"][0])
    for key, value in recorded_env(target).items():
        if os.environ.get(key, value) != value:
            print(f"· {key}={value} (recorded; was {os.environ[key]})", file=sys.stderr)
        os.environ[key] = value
    os.environ["SGLANG_VLM_CACHE_SIZE_MB"] = str(args.vlm_cache_mb)
    if os.environ.get("ARC_INT3_DEPLOY") == "1":
        install_int3(flags)
    install()

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(argv)
    isolation = {
        "max_running_requests": server_args.max_running_requests == 1,
        "disable_radix_cache": server_args.disable_radix_cache,
        "disable_cuda_graph": server_args.disable_cuda_graph,
        "disable_prefill_cuda_graph": server_args.disable_prefill_cuda_graph,
        "disable_overlap_schedule": server_args.disable_overlap_schedule,
        "no speculative decoding": server_args.speculative_algorithm is None,
        "no session ledger": server_args.session_ledger_dir is None and not server_args.session_replay,
    }
    broken = [k for k, ok in isolation.items() if not ok]
    if broken:
        raise ValueError(f"capture isolation needs {broken}")
    import sglang

    tree = Path(sglang.__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], check=True, capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no"], check=True,
                           capture_output=True, text=True).stdout.strip()
    profile = None if args.target_profile is None else str(args.target_profile.resolve())
    profile_sha = None if profile is None else hashlib.sha256(args.target_profile.read_bytes()).hexdigest()
    meta = {"instance": None if args.recorded is None else recorded["instance"],
            "recorded": None if args.recorded is None else str(args.recorded.resolve()), "target_profile": profile,
            "target_profile_sha256": profile_sha, "target_instance": target["instance"],
            "served_model_name": flags["--served-model-name"][0], "argv": argv, "env": recorded_env(target),
            "sglang_tree": str(tree), "sglang_commit": commit, "sglang_dirty": bool(dirty), "port": args.port}
    (Path(os.environ[ROOT_ENV]) / "capture_server.json").write_text(json.dumps(meta, indent=1) + "\n")
    if args.recorded is None:
        what = f"every source, with the target of {target['instance']}"
    else:
        what = recorded["instance"] + ("" if profile is None else f" with the target of {target['instance']}")
    print(f"· capture server for {what} (sglang {commit[:12]}{' dirty' if dirty else ''}): "
          f"{shlex.join(argv)}", file=sys.stderr)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
elif __name__ == "__mp_main__":
    install()
