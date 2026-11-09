"""TITO ledger segments → main-model LoRA samples (≤120k, images + video), proven bit-identical to deployed sglang's input.

Runs in the deployed venv + fork (``prepare_tito_mm.sh`` sets PYTHONPATH / ffmpeg / visible GPUs). Format and reader:
see ``tito_mm_data.py``.

1. Samples: one per segment (readable under strict, has gen, ≤``--max-tokens``). ``input_ids`` = ledger S verbatim;
   ``loss_mask`` = the ledger's gen spans (``Segment.gen_spans``: rewound tails, the ``<|im_end|>`` swapped in by
   stop_substituted, template-filled tokens, tool output, prompt and media pads are all excluded). Task-level split:
   group = the joined task (arc2 task / arc3 game), the thread when not joined.
2. Media: every unique sha256 goes through **the deployed sglang processor object** (``QwenVLImageProcessor`` rebuilt
   from the resolved server_args recorded in server.json, entry point ``process_mm_data_async``, the prompt being the
   chat template's single placeholder). In deployment the HF processor runs on ``cuda:{base_gpu_id}``. ``--device cuda``
   (production data) is exactly the deployed setup: one visible GPU, no code path changed. ``--device cpu-emulated``
   (v1; for when the GPU is off limits) switches the device to CPU, sends bicubic through torchvision's float path and
   records the number of elements near the .5 rounding boundary -- GPU re-verification (``verify_tito_mm_media.py``)
   showed it still differs from deployment by ±1 (0.135% of the elements of sensitive media), so it cannot be used as
   training data. ``--device cpu-exact`` does not touch the GPU: bicubic is replaced by a bit-exact reproduction of the
   CUDA kernel (``cuda_bicubic_aa``; 0 differences on the 9,176 resized media checked against the GPU on 10-21 (TITO v2
   8,853 + RL packet 323)), frames identical to ``cuda``.
   Asserted one by one: blob sha256; pixels = ``patchify(uint8 frames)`` bit-identical;
   grid = ledger; each range length = h·w/4; the media's whole block in the ledger (vision_start, timestamp text, pads,
   vision_end) = the processor's expansion; video block = sglang ``build_input_ids_with_timestamps`` (timestamps from
   the HF processor's ``_calculate_timestamps``).
3. M-RoPE: sglang ``MRotaryEmbedding.get_rope_index`` (the path deployment takes for requests with video) over the
   whole segment; image-only segments are also computed with ``_compute_image_only_mrope_positions_from_offsets``, the
   path deployment actually takes, and asserted equal; text-only segments are asserted = arange.
4. report.json: pass counts of all assertions, samples / tokens / loss tokens, splits, media sizes, per-stage timings
   and the slowest single item.
   ``--compare-to <old version>``: asserts ids / mask / position / index are byte-identical to the old version and the
   media metadata identical; only the frame differences are counted.

Any inconsistency raises immediately (no skipping, no patching).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import hashlib
import json
import logging
import multiprocessing as mp_ctx
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tito_mm_data import (  # noqa: E402
    FORMAT,
    PIXEL_CENTER,
    PIXEL_SCALE,
    SPLITS,
    check_media_layout,
    frame_count,
    media_path,
    pad_ranges,
    patchify,
)

IMAGE_PROMPT = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PROMPT = "<|vision_start|><|video_pad|><|vision_end|>"
RISK_EPS = 1e-3                    # results this close to x.5 can round the other way on a CPU/GPU ulp difference
#: bicubic stats recorded per media: at_risk* = float32 result elements within eps of x.5, exact_half = exactly x.5,
#: f64_round_diff = float64 interpolation rounds differently from float32 (precision-sensitive), native_u8_diff = the
#: CPU uint8 fixed-point path differs from the GPU path
RISK_KEYS = ("bicubic_calls", "bicubic_elems", "at_risk", "at_risk_1e4", "at_risk_1e5", "exact_half", "f64_round_diff",
             "native_u8_diff")
#: sglang preprocessing sources this program actually executes (sha256 recorded in the manifest: identifies the code
#: even when the fork is not a git checkout)
SGLANG_PATH_SOURCES = (
    "sglang/srt/multimodal/processors/qwen_vl.py", "sglang/srt/multimodal/processors/base_processor.py",
    "sglang/srt/utils/video_decoder.py", "sglang/srt/utils/common.py", "sglang/srt/managers/schedule_batch.py",
    "sglang/srt/layers/rotary_embedding/mrope.py", "sglang/srt/layers/rotary_embedding/mrope_rope_index.py",
    "sglang/srt/managers/multimodal_processor.py", "sglang/srt/server_args.py")
#: where the HF processor runs (recorded in the manifest; part of the processor digest)
#: The text values are kept verbatim: they feed processor_digest, so recorded datasets' digests depend on them. Gloss:
#: cuda = deployed as is: HF processor on cuda:{base_gpu_id} (one visible GPU), torchvision GPU bicubic, code path
#: unchanged; cpu-emulated = HF processor on CPU (deployed on cuda:base_gpu_id), bicubic emulates torchvision's GPU path
#: (uint8→float32 interpolate(antialias)→clamp→round), bilinear untouched (preprocess_video is on CPU in deployment),
#: warning: differs from deployment by ±1 on GPU re-check, reference only for when the GPU is off limits;
#: cpu-exact = bicubic is cuda_bicubic_aa (bit-exact float32/FMA reproduction of torch 2.13 CUDA
#: upsample_gen2d_aa_out_frame → clamp → round), any other interpolate raises (only bicubic + antialias allowed).
DEVICE_PATHS = {
    "cuda": "部署原樣：HF 處理器在 cuda:{base_gpu_id}（一張可見的卡），torchvision GPU bicubic；程式路徑不改",
    "cpu-emulated": {"hf_processor_device": "cpu（部署是 cuda:base_gpu_id）",
                     "bicubic": "uint8→float32 interpolate(antialias)→clamp→round（模擬 torchvision 的 GPU 路徑）",
                     "bilinear": "不動（preprocess_video 部署時就在 CPU）", "risk_eps": RISK_EPS,
                     "warning": "GPU 複驗與部署差 ±1；只供不准用 GPU 時參考"},
    "cpu-exact": {"hf_processor_device": "cpu（部署是 cuda:base_gpu_id）",
                  "bicubic": "cuda_bicubic_aa.upsample_bicubic2d_aa：torch 2.13 CUDA upsample_gen2d_aa_out_frame"
                             "的 float32／FMA 逐位元重現 → clamp → round（torchvision 的 GPU 路徑）",
                  "other_interpolate": "raise（geo.interpolate 只准 bicubic＋antialias）",
                  "torch": "2.13.0"},
}
#: --compare-to: these must be byte-identical to the reference version
TOKEN_FILES = ("input_ids.i32", "loss_mask.u8", "mrope.i32", "index.jsonl")
#: --compare-to: media metadata (other than frames and timings / stats) must be identical
MEDIA_META = ("modality", "grid_thw", "frames_shape", "block", "timestamps", "fps", "total_frames", "frames_indices")
#: server_args keys allowed to differ between two instances (unrelated to preprocessing)
SERVER_ARGS_MAY_DIFFER = {"port", "base_gpu_id", "session_ledger_dir", "random_seed"}


# ----------------------------------------------------------------------------- pure functions (tests need no sglang)


def loss_mask_from_spans(n: int, gen_spans) -> np.ndarray:
    """[start, end) gen spans → uint8 mask. Spans must be sorted, non-overlapping, non-empty and exclude position 0."""
    mask = np.zeros(n, np.uint8)
    last = 0
    for s, e in gen_spans:
        if not (0 < s < e <= n) or s < last:
            raise RuntimeError(f"gen span [{s}, {e}) invalid (n={n}, previous one ends at {last})")
        mask[s:e] = 1
        last = e
    return mask


def segment_media(seg) -> list[dict]:
    """The segment's media, in order of appearance in ids. Within one ledger append images are listed before videos
    (the order of the processor's mm_items), so only ledger order = appearance order "within one modality" is required
    (raise otherwise), then they are sorted by position. Media without bytes raise."""
    out = []
    for m in seg.media:
        if m.sha256 is None or m.unmapped or m.grid_thw is None:
            raise RuntimeError(f"{seg.key}: media without bytes or grid (sha256={m.sha256}, unmapped={m.unmapped})")
        out.append({"modality": m.modality, "sha256": m.sha256, "ranges": [list(r) for r in m.ranges],
                    "grid_thw": list(m.grid_thw)})
    for modality in ("image", "video"):
        starts = [m["ranges"][0][0] for m in out if m["modality"] == modality]
        if starts != sorted(starts):
            raise RuntimeError(f"{seg.key}: ledger {modality} order {starts[:6]} is not the order of appearance in ids")
    return sorted(out, key=lambda m: m["ranges"][0][0])


def split_of(group: str, seed: int, val_frac: float) -> str:
    """Task-level split: validation if the group's sha256 falls in [0, val_frac). A group always lands on the same side."""
    h = int.from_bytes(hashlib.sha256(f"tito-mm-split:{seed}:{group}".encode()).digest()[:8], "big")
    return "validation" if h / 2 ** 64 < val_frac else "train"


def unpatchify_u8(pixels, modality: str, grid_thw, patch: int, temporal_patch: int, merge: int) -> np.ndarray:
    """Processor float32 pixel_values → uint8 frames (T, 3, H, W), asserting ``patchify(frames)`` is bit-identical to
    the input."""
    import torch

    t, h, w = grid_thw
    c = 3
    if pixels.dtype != torch.float32 or tuple(pixels.shape) != (t * h * w, c * temporal_patch * patch * patch):
        raise RuntimeError(f"pixel_values {pixels.dtype} {tuple(pixels.shape)}, grid {grid_thw}")
    u = torch.round(pixels * PIXEL_SCALE + PIXEL_CENTER)
    if float(u.min()) < 0 or float(u.max()) > 255:
        raise RuntimeError(f"uint8 inferred from pixel_values out of range [{float(u.min())}, {float(u.max())}]")
    x = u.to(torch.uint8).view(t, h // merge, w // merge, merge, merge, c, temporal_patch, patch, patch)
    perm = (0, 3, 6, 4, 7, 2, 1, 5, 8)                      # the permute of patchify
    inv = tuple(perm.index(k) for k in range(9))
    frames = x.permute(*inv).reshape(t * temporal_patch, c, h * patch, w * patch).contiguous().numpy()
    if modality == "image":
        if t != 1 or not all(np.array_equal(frames[0], f) for f in frames[1:]):
            raise RuntimeError("image temporal copies are not identical")
        frames = frames[:1]
    if frames.shape[0] != frame_count(modality, grid_thw, temporal_patch):
        raise RuntimeError(f"{modality} frames {frames.shape}, grid {grid_thw}")
    if not torch.equal(patchify(frames, patch, temporal_patch, merge), pixels):
        raise RuntimeError("patchify(uint8 frames) is not bit-identical to the processor's pixel_values (normalize is not (u8-127.5)/127.5?)")
    return frames


def check_media_block(ids: np.ndarray, m: dict, rec: dict, vision: dict, where: str) -> None:
    """This media's whole block in the ledger = the processor's expansion of a single placeholder (vision_start,
    timestamp text, pads, vision_end), and the pad runs inside the block = the ledger's ranges."""
    if tuple(m["grid_thw"]) != tuple(rec["grid_thw"]) or m["modality"] != rec["modality"]:
        raise RuntimeError(f"{where}: ledger {m['modality']} grid {m['grid_thw']}, processor {rec['modality']} {rec['grid_thw']}")
    block = rec["block"]
    pad = vision["image_pad"] if m["modality"] == "image" else vision["video_pad"]
    first_pad = block.index(pad)
    start = m["ranges"][0][0] - first_pad
    if start < 0 or start + len(block) > len(ids):
        raise RuntimeError(f"{where}: block start {start}, length {len(block)} exceeds segment length {len(ids)}")
    got = ids[start:start + len(block)]
    if not np.array_equal(got, np.asarray(block)):
        bad = int(np.flatnonzero(got != np.asarray(block))[0])
        raise RuntimeError(f"{where}: ledger block token {bad} is {int(got[bad])}, the processor expansion has {block[bad]}"
                           f" (block start {start})")
    runs = [[a + start, b + start] for a, b in pad_ranges(np.asarray(block), pad)]
    if runs != [list(r) for r in m["ranges"]]:
        raise RuntimeError(f"{where}: processor pad runs {runs[:3]}, ledger ranges {m['ranges'][:3]}")


def image_block(n_pad: int, vision: dict) -> list[int]:
    return [vision["vision_start"]] + [vision["image_pad"]] * n_pad + [vision["vision_end"]]


def vision_spec(vocab) -> dict:
    """The manifest's ``vision`` (the reader ``TitoMMDataset`` checks pads and patchify with it). Shared with
    ``prepare_rl.py``."""
    return {"patch": 16, "temporal_patch": 2, "merge": 2, "image_pad": vocab.image_pad, "video_pad": vocab.video_pad,
            "vision_start": vocab.vision_start, "vision_end": vocab.vision_end,
            "pixels_dtype": "float32", "frames_dtype": "uint8", "pixel_center": PIXEL_CENTER,
            "pixel_scale": PIXEL_SCALE}


# ----------------------------------------------------------------------------- worker (the deployed sglang processor)

_W: dict = {}


def _install_cuda_resize_emulation(stats: dict) -> None:
    """On CPU, torchvision v2 runs uint8 bicubic through a fixed-point uint8 kernel; on GPU, through float32
    interpolation + clamp + round. Deployment runs on GPU: disable the CPU uint8 shortcut (bilinear untouched -- sglang's
    preprocess_video runs on CPU in deployment anyway), and for every float32 bicubic interpolation record the number of
    elements close to the rounding boundary and how many values the CPU uint8 path would differ by (measuring how much
    depends on the GPU)."""
    import torch
    import torchvision.transforms.v2.functional._geometry as geo
    from torchvision.transforms import InterpolationMode

    orig_native, orig_interp = geo._do_native_uint8_resize_on_cpu, geo.interpolate

    def native(interpolation):
        if interpolation in (InterpolationMode.BICUBIC, InterpolationMode.LANCZOS):
            return False
        return orig_native(interpolation)

    def interp(image, *args, **kwargs):
        out = orig_interp(image, *args, **kwargs)
        if kwargs.get("mode") == "bicubic" and image.dtype == torch.float32:
            if not torch.equal(image, image.round()) or float(image.min()) < 0 or float(image.max()) > 255:
                raise RuntimeError("bicubic input is not float converted from uint8 (not resize_image's GPU path)")
            inside = (out > -0.5) & (out < 255.5)
            dist = ((out - torch.floor(out)) - 0.5).abs()
            emulated = out.clamp(0, 255).round().to(torch.uint8)
            native_u8 = orig_interp(image.to(torch.uint8), *args, **kwargs)
            f64 = orig_interp(image.to(torch.float64), *args, **kwargs).clamp(0, 255).round().to(torch.uint8)
            stats["bicubic_calls"] += 1
            stats["bicubic_elems"] += out.numel()
            stats["at_risk"] += int(((dist < RISK_EPS) & inside).sum())
            stats["at_risk_1e4"] += int(((dist < 1e-4) & inside).sum())
            stats["at_risk_1e5"] += int(((dist < 1e-5) & inside).sum())
            stats["exact_half"] += int(((dist == 0) & inside).sum())
            stats["f64_round_diff"] += int((f64 != emulated).sum())
            stats["native_u8_diff"] += int((native_u8 != emulated).sum())
        return out

    geo._do_native_uint8_resize_on_cpu = native
    geo.interpolate = interp


def _install_cuda_resize_exact(stats: dict) -> None:
    """``cpu-exact``: torchvision v2's uint8 bicubic follows the GPU path (uint8→float32→interpolate→clamp→round), with
    interpolate replaced by the bit-exact reproduction of the CUDA kernel (``cuda_bicubic_aa``). Any other
    geo.interpolate call (other modes, scale_factor, no antialias, non-float32 / non-uint8 source) is outside what is
    reproduced and raises."""
    import torch
    import torchvision
    import torchvision.transforms.v2.functional._geometry as geo
    from cuda_bicubic_aa import upsample_bicubic2d_aa
    from torchvision.transforms import InterpolationMode

    if torch.__version__.split("+")[0] != DEVICE_PATHS["cpu-exact"]["torch"]:
        raise RuntimeError(f"cpu-exact reproduces the CUDA kernel of torch {DEVICE_PATHS['cpu-exact']['torch']}, "
                           f"this is {torch.__version__} (if the deployed torch changes, re-verify against the GPU)")
    orig_native = geo._do_native_uint8_resize_on_cpu
    want = {"size", "mode", "align_corners", "antialias"}

    def native(interpolation):
        if interpolation in (InterpolationMode.BICUBIC, InterpolationMode.LANCZOS):
            return False
        return orig_native(interpolation)

    def interp(image, **kwargs):
        if set(kwargs) != want or kwargs["mode"] != "bicubic" or kwargs["antialias"] is not True \
                or kwargs["align_corners"] is not False:
            raise RuntimeError(f"cpu-exact only reproduces bicubic + antialias, align_corners=False resizes, got {kwargs}"
                               f" (torchvision {torchvision.__version__})")
        if image.dtype != torch.float32 or image.device.type != "cpu" or image.dim() != 4:
            raise RuntimeError(f"cpu-exact: bicubic input is {image.dtype} {image.device} {tuple(image.shape)}")
        if not torch.equal(image, image.round()) or float(image.min()) < 0 or float(image.max()) > 255:
            raise RuntimeError("cpu-exact: bicubic input is not float converted from uint8 (not resize_image's GPU path)")
        out_h, out_w = (int(s) for s in kwargs["size"])
        out = torch.from_numpy(upsample_bicubic2d_aa(image.contiguous().numpy(), out_h, out_w))
        stats["bicubic_calls"] += 1
        stats["bicubic_elems"] += out.numel()
        return out

    geo._do_native_uint8_resize_on_cpu = native
    geo.interpolate = interp


def _build_processor(server_json: Path):
    """Rebuild the deployed sglang multimodal processor from the resolved server_args recorded in server.json
    (resolution is not run: it needs a GPU)."""
    from sglang.srt import runtime_context
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
    from sglang.srt.managers.tokenizer_manager import get_processor_wrapper
    from sglang.srt.multimodal.transport import determine_tensor_transport_mode
    from sglang.srt.server_args import ServerArgs, record_fields

    rec = json.loads(server_json.read_text())["server_args"]
    names = {f.name for f in record_fields(ServerArgs)}
    if set(rec) != names:
        raise RuntimeError(f"server.json server_args fields differ from this ServerArgs version: extra {sorted(set(rec) - names)[:5]}, "
                           f"missing {sorted(names - set(rec))[:5]}")
    sa = ServerArgs(**rec)
    sa._resolution_finished = True        # what was recorded are the resolved values
    runtime_context.publish(sa, role="test")
    mc = ModelConfig.from_server_args(sa)
    import_processors("sglang.srt.multimodal.processors")
    hf = get_processor_wrapper()
    mp = get_mm_processor(mc.hf_config, sa, hf, determine_tensor_transport_mode(), model_config=mc)
    return sa, mc, hf, mp


class FailLoudPool:
    """Subset of ``multiprocessing.Pool``'s ``imap`` / ``imap_unordered`` on top of ``ProcessPoolExecutor``.

    When the initializer of a ``multiprocessing.Pool`` raises, it respawns workers forever and the main process waits
    forever (2026-10-07: prepare_rl hung for 20 minutes when it used the wrong sglang tree). ProcessPoolExecutor marks
    the pool broken in the same situation and the caller gets ``BrokenProcessPool``. No signals are sent: after the main
    process dies a worker's ``getppid()`` points at the subreaper that adopted it (systemd --user), and sending SIGTERM
    there stops the whole user manager."""

    def __init__(self, ctx, workers: int, initializer, initargs: tuple):
        self.executor = concurrent.futures.ProcessPoolExecutor(workers, mp_context=ctx, initializer=initializer,
                                                               initargs=initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def imap(self, fn, items, chunksize: int = 1):
        return self.executor.map(fn, items, chunksize=chunksize)

    def imap_unordered(self, fn, items, chunksize: int = 1):
        del chunksize                     # every item is its own future; as_completed gives completion order
        futures = [self.executor.submit(fn, item) for item in items]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()


def _init_worker(server_json: str, torch_threads: int, device: str = "cpu-emulated") -> None:
    """``cpu-emulated`` (default, data production): HF processor moved to CPU + emulation of the GPU resize path +
    stats. ``cuda``: deployed as is (``cuda:{base_gpu_id}``), only for re-verification by ``verify_tito_mm_media.py`` when
    GPU access is granted. ``cpu-exact``: HF processor on CPU, bicubic through ``cuda_bicubic_aa`` (bit-identical to
    cuda)."""
    import torch

    torch.set_num_threads(torch_threads)
    logging.getLogger("sglang").setLevel(logging.WARNING)
    import sglang.srt.multimodal.processors.qwen_vl as qwen_vl

    sa, mc, hf, mp = _build_processor(Path(server_json))
    deployed = mp._fast_image_processor_device(hf)
    if deployed != f"cuda:{sa.base_gpu_id}":
        raise RuntimeError(f"the deployed HF processor device is {deployed}; this program's GPU emulation assumes cuda")
    stats = collections.Counter()
    if device == "cpu-emulated":
        mp._fast_image_processor_device = lambda processor: "cpu"
        _install_cuda_resize_emulation(stats)
    elif device == "cpu-exact":
        mp._fast_image_processor_device = lambda processor: "cpu"
        _install_cuda_resize_exact(stats)
    elif device != "cuda":
        raise ValueError(f"device={device!r}")
    video_meta: list = []
    orig_preprocess_video = qwen_vl.preprocess_video

    async def preprocess_video(vr, *args, **kwargs):
        video, meta = await orig_preprocess_video(vr, *args, **kwargs)
        video_meta.append(meta)
        return video, meta

    qwen_vl.preprocess_video = preprocess_video
    v = mp.mm_tokens
    ip, vp = hf.image_processor, hf.video_processor
    if (ip.patch_size, ip.temporal_patch_size, ip.merge_size) != (vp.patch_size, vp.temporal_patch_size, vp.merge_size):
        raise RuntimeError("image and video processors differ in patch / temporal / merge")
    _W.update(mp=mp, hf=hf, stats=stats, video_meta=video_meta, qwen_vl=qwen_vl, vision=dict(
        patch=hf.image_processor.patch_size, temporal_patch=hf.image_processor.temporal_patch_size,
        merge=hf.image_processor.merge_size, image_pad=v.image_token_id, video_pad=v.video_token_id,
        vision_start=mp.vision_start_token_id, vision_end=mp.vision_end_token_id))


def _process_media(sha: str, modality: str, src: str):
    """Run one media item through the deployed processor (single-placeholder prompt) → (item, input_ids, grid, uint8
    frames)."""
    import torch
    from sglang.srt.managers.schedule_batch import Modality

    mp, vision = _W["mp"], _W["vision"]
    raw = Path(src).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise RuntimeError(f"{src}: content sha256 does not match the file name")
    _W["stats"].clear()
    _W["video_meta"].clear()
    rid = f"tito-mm:{sha[:16]}"
    if modality == "image":
        req = SimpleNamespace(video_data=None, audio_data=None, rid=rid)
        out = asyncio.run(mp.process_mm_data_async([raw], IMAGE_PROMPT, req))
        want_modality, grid_key = Modality.IMAGE, "image_grid_thw"
    else:
        req = SimpleNamespace(video_data=[raw], audio_data=None, rid=rid)
        out = asyncio.run(mp.process_mm_data_async(None, VIDEO_PROMPT, req))
        want_modality, grid_key = Modality.VIDEO, "video_grid_thw"
    (item,) = out.mm_items
    if item.modality != want_modality:
        raise RuntimeError(f"{sha}: processor gave {item.modality}, expected {want_modality}")
    grid = tuple(int(x) for x in torch.as_tensor(item.model_specific_data[grid_key]).reshape(-1).tolist())
    if len(grid) != 3:
        raise RuntimeError(f"{sha}: grid {grid}")
    pixels = item.feature
    if not isinstance(pixels, torch.Tensor) or pixels.device.type != "cpu":
        raise RuntimeError(f"{sha}: feature is not a CPU tensor ({type(pixels)})")
    frames = unpatchify_u8(pixels, modality, grid, vision["patch"], vision["temporal_patch"], vision["merge"])
    return item, out.input_ids, grid, frames


def _media_task(task: tuple) -> dict:
    """One unique media item: deployed processor → uint8 frames (bit-verified) + expanded block; writes the npz."""
    import torch

    sha, modality, src, out_dir = task
    mp, hf, vision = _W["mp"], _W["hf"], _W["vision"]
    t0 = time.time()
    item, input_ids, grid, frames = _process_media(sha, modality, src)
    block = [int(x) for x in input_ids]
    pad = vision["image_pad"] if modality == "image" else vision["video_pad"]
    runs = pad_ranges(np.asarray(block), pad)
    if runs != [tuple(int(x) for x in o) for o in item.offsets]:
        raise RuntimeError(f"{sha}: block pad runs {runs[:3]} ≠ item.offsets {item.offsets[:3]}")
    n_pad = grid[1] * grid[2] // vision["merge"] ** 2
    rec = {"sha256": sha, "modality": modality, "grid_thw": list(grid), "frames_shape": list(frames.shape)}
    checks = collections.Counter(media_blob_sha256=1, pixels_bit_exact_u8=1)
    if modality == "image":
        if block != image_block(n_pad, vision):
            raise RuntimeError(f"{sha}: image expansion is not vision_start + {n_pad} pad + vision_end")
        checks["image_block_shape"] += 1
    else:
        if len(runs) != grid[0] or any(b - a + 1 != n_pad for a, b in runs):
            raise RuntimeError(f"{sha}: video has {len(runs)} temporal groups {runs[:3]}, grid {grid}")
        (meta,) = _W["video_meta"]
        stamps = hf._calculate_timestamps(list(meta["frames_indices"]), meta["fps"], vision["temporal_patch"])
        sg_ids, sg_offsets, sg_mods = mp.build_input_ids_with_timestamps(
            [vision["vision_start"], vision["video_pad"], vision["vision_end"]], None, None,
            torch.tensor([grid], dtype=torch.long), [stamps])
        if list(sg_ids) != block or [tuple(o) for o in sg_offsets] != runs or len(sg_mods) != grid[0]:
            raise RuntimeError(f"{sha}: sglang build_input_ids_with_timestamps differs from the processor expansion")
        checks["video_block_eq_build_input_ids_with_timestamps"] += 1
        rec.update(block=block, timestamps=[f"{s:.1f}" for s in stamps], fps=float(meta["fps"]),
                   total_frames=int(meta["total_num_frames"]),
                   frames_indices=[int(i) for i in meta["frames_indices"]])
    path = media_path(out_dir, sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, frames=frames)
    os.replace(tmp, path)
    rec.update(file_bytes=path.stat().st_size, u8_bytes=int(frames.nbytes),
               **{k: int(_W["stats"][k]) for k in RISK_KEYS}, seconds=time.time() - t0, checks=dict(checks))
    return rec


def _verify_task(task: tuple) -> dict:
    """Stored frames vs recomputed (``verify_tito_mm_media.py``; cuda = deployed as is)."""
    sha, modality, src, data_dir = task
    _, _, grid, frames = _process_media(sha, modality, src)
    with np.load(media_path(data_dir, sha)) as z:
        stored = z["frames"]
    if stored.shape != frames.shape:
        raise RuntimeError(f"{sha}: stored {stored.shape}, recomputed {frames.shape}")
    diff = stored.astype(np.int16) - frames.astype(np.int16)
    return {"sha256": sha, "modality": modality, "grid_thw": list(grid), "n_diff": int(np.count_nonzero(diff)),
            "max_abs_diff": int(np.abs(diff).max()), "elems": int(diff.size)}


def _segment_task(task: tuple) -> tuple:
    """One segment: media block / range alignment, loss mask, sglang M-RoPE. Returns (key, ids, loss, positions, checks,
    seconds)."""
    import torch
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

    key, ids, gen_spans, media, records = task
    mp, vision = _W["mp"], _W["vision"]
    t0 = time.time()
    n = len(ids)
    checks = collections.Counter()
    check_media_layout(ids, media, vision, key)
    checks["pads_eq_ranges_union"] += 1
    for i, m in enumerate(media):
        rec = records[m["sha256"]]
        if m["modality"] == "image":         # processor's image expansion (asserted in _media_task): pad count follows the processor's grid
            rec = dict(rec, block=image_block(rec["grid_thw"][1] * rec["grid_thw"][2] // vision["merge"] ** 2, vision))
        check_media_block(ids, m, rec, vision, f"{key} media[{i}]")
        checks[f"{m['modality']}_grid_eq_ledger"] += 1
        checks[f"{m['modality']}_block_eq_processor"] += 1
        checks[f"{m['modality']}_range_len_eq_hw_over_4"] += len(m["ranges"])
    loss = loss_mask_from_spans(n, gen_spans)
    if loss[(ids == vision["image_pad"]) | (ids == vision["video_pad"])].any():
        raise RuntimeError(f"{key}: loss falls on a pad")
    checks["loss_mask_eq_gen_spans"] += 1
    images = [m for m in media if m["modality"] == "image"]
    videos = [m for m in media if m["modality"] == "video"]
    input_ids = torch.from_numpy(ids.astype(np.int64)).unsqueeze(0)
    grids = lambda ms: torch.tensor([m["grid_thw"] for m in ms], dtype=torch.long) if ms else None  # noqa: E731
    pos, delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=mp._spatial_merge_size, image_token_id=mp.mm_tokens.image_token_id,
        video_token_id=mp.mm_tokens.video_token_id, vision_start_token_id=mp.vision_start_token_id,
        model_type=mp.model_type, tokens_per_second=mp._tokens_per_second, input_ids=input_ids,
        image_grid_thw=grids(images), video_grid_thw=grids(videos), second_per_grid_ts=None,
        use_audio_in_video=False, audio_seqlens=None, audio_token_id=getattr(mp.hf_config, "audio_token_id", None),
        audio_start_token_id=mp.audio_start_token_id,
        position_id_per_seconds=getattr(mp.hf_config, "position_id_per_seconds", None))
    pos = pos.squeeze(1)
    if tuple(pos.shape) != (3, n):
        raise RuntimeError(f"{key}: positions {tuple(pos.shape)}")
    if images and not videos:
        items = [MultimodalDataItem(modality=Modality.IMAGE, feature=None, offsets=[tuple(m["ranges"][0])],
                                    model_specific_data={"image_grid_thw": torch.tensor([m["grid_thw"]])})
                 for m in images]
        alt = mp._compute_image_only_mrope_positions_from_offsets(n, items, torch.long, torch.device("cpu"))
        if alt is None or not torch.equal(alt[0].squeeze(1), pos) or int(alt[1]) != int(delta):
            raise RuntimeError(f"{key}: image-only M-RoPE path differs from get_rope_index")
        checks["mrope_image_only_path_eq_get_rope_index"] += 1
    if not media:
        if not torch.equal(pos, torch.arange(n).expand(3, n)):
            raise RuntimeError(f"{key}: positions of a text-only segment are not arange")
        checks["mrope_text_only_is_arange"] += 1
    last = pos[:, -1]
    if not (last == last[0]).all() or int(last[0]) != n - 1 + int(delta):
        raise RuntimeError(f"{key}: last token position {last.tolist()}, n-1+delta={n - 1 + int(delta)}")
    checks["mrope_decode_delta_consistent"] += 1
    if int(pos.min()) < 0 or int(pos.max()) >= 2 ** 31:
        raise RuntimeError(f"{key}: positions exceed int32")
    return key, ids.astype("<i4"), loss, pos.numpy().astype("<i4"), dict(checks), time.time() - t0


# ----------------------------------------------------------------------------- main


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _environment(model_dir: Path, server_jsons: list[Path], device: str) -> dict:
    """Inputs of the processor digest: the model directory's processor / tokenizer files, fork commit, package versions,
    the mm fields of the deployed server_args."""
    import sglang
    import torch
    import torchvision
    import transformers

    fork_python = Path(sglang.__file__).resolve().parents[1]
    fork = fork_python.parent
    servers = [json.loads(p.read_text()) for p in server_jsons]
    base = servers[0]
    for p, s in zip(server_jsons[1:], servers[1:]):
        diff = {k for k in base["server_args"] if base["server_args"][k] != s["server_args"][k]}
        if not diff <= SERVER_ARGS_MAY_DIFFER:
            raise RuntimeError(f"{p}: server_args keys differing from {server_jsons[0]}: {sorted(diff - SERVER_ARGS_MAY_DIFFER)}")
        if s["model_files_sha256"] != base["model_files_sha256"]:
            raise RuntimeError(f"{p} and {server_jsons[0]}: model_files_sha256 differs")
    is_git = (fork / ".git").exists()
    path_sources = {rel: _sha256(fork_python / rel) for rel in SGLANG_PATH_SOURCES}
    files = {name: _sha256(model_dir / name) for name in base["model_files_sha256"]}
    if files != base["model_files_sha256"]:
        bad = sorted(k for k in files if files[k] != base["model_files_sha256"][k])
        raise RuntimeError(f"{model_dir}: {bad} differ from the server.json recorded at collection time")
    mm_keys = sorted(k for k in base["server_args"]
                     if any(s in k for s in ("mm_", "image", "video", "processor", "interpolation", "tokenizer")))
    return {
        "model_dir": str(model_dir), "model_files_sha256": files,
        "server_args_mm": {k: base["server_args"][k] for k in mm_keys},
        "server_env": {k: base["env"][k] for k in sorted(base["env"]) if k.startswith(("ARC_MROPE", "SGLANG_IMAGE",
                                                                                         "SGLANG_MM", "VIDEO"))},
        "collected_sglang_version": base["sglang_version"],
        "fork": str(fork), "fork_commit": _git("-C", str(fork), "rev-parse", "HEAD") if is_git else None,
        "fork_dirty": _git("-C", str(fork), "status", "--porcelain", "--", "python") != "" if is_git else None,
        "sglang_path_sources_sha256": path_sources,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": device,
        "device_path": DEVICE_PATHS[device],
    }


def _select_subset(cands: list[dict], n: int, seed: int) -> list[dict]:
    """Small dev sample: at most 8 video segments, at least 1 with a rewind, the rest images / text-only at 3:2."""
    rng = np.random.default_rng(seed)
    video = [c for c in cands if any(m["modality"] == "video" for m in c["media"])]
    image = [c for c in cands if c["media"] and c not in video]
    text = [c for c in cands if not c["media"]]
    rewind = [c for c in cands if c["n_dropped"]]
    pick = list(rng.choice(len(video), min(8, len(video), n), replace=False))
    out = [video[i] for i in pick]
    if rewind and not any(c["n_dropped"] for c in out):
        out.append(rewind[int(rng.integers(len(rewind)))])
    rest = n - len(out)
    n_img = min(len(image), rest * 3 // 5)
    out += [image[i] for i in rng.choice(len(image), n_img, replace=False)]
    out += [text[i] for i in rng.choice(len(text), min(len(text), rest - n_img), replace=False)]
    keys = {c["key"] for c in out}
    return [c for c in cands if c["key"] in keys]


#: parity subset: (category, length quantile). Video takes 8 evenly spaced quantiles of the segments with video; for the
#: rest short = low quantiles, long = high quantiles
PARITY_PLAN = (("video", tuple(i / 7 for i in range(8))), ("text/short", (0.1, 0.25)), ("text/long", (0.75, 0.95)),
               ("multi_image/short", (0.1, 0.25, 0.4)), ("multi_image/long", (0.6, 0.8, 0.95)),
               ("single_image/short", (0.1, 0.25)), ("single_image/long", (0.75, 0.95)), ("rewind", (0.25, 0.75)))


def _parity_set(rows: list[dict]) -> list[dict]:
    """About 24 segments for a later per-token logprob comparison against sglang serving: one segment per (category,
    quantile) of PARITY_PLAN, no repeats. rewind = a segment with a rewound gen tail that is not chosen yet. A category
    with no segments yields fewer picks (the caller asserts the total on a full run)."""
    kind = {"video": lambda r: r["n_videos"] > 0,
            "text": lambda r: not r["n_images"] and not r["n_videos"],
            "multi_image": lambda r: r["n_images"] >= 2 and not r["n_videos"],
            "single_image": lambda r: r["n_images"] == 1 and not r["n_videos"],
            "rewind": lambda r: r["n_dropped"] > 0}
    out, chosen = [], set()
    for reason, quantiles in PARITY_PLAN:
        group = sorted((r for r in rows if kind[reason.split("/")[0]](r) and r["key"] not in chosen),
                       key=lambda r: (r["length"], r["key"]))
        if not group:
            continue
        for q in quantiles:
            r = group[min(len(group) - 1, round(q * (len(group) - 1)))]
            if r["key"] in chosen:
                continue
            chosen.add(r["key"])
            out.append(dict(key=r["key"], split=r["split"], length=r["length"], n_images=r["n_images"],
                            n_videos=r["n_videos"], n_dropped=r["n_dropped"], reason=f"{reason}@{q:.2f}"))
    return out


def _frame_diff(task: tuple) -> dict:
    sha, modality, out, ref = task
    with np.load(media_path(out, sha)) as z:
        a = z["frames"]
    with np.load(media_path(ref, sha)) as z:
        b = z["frames"]
    if a.shape != b.shape or a.dtype != b.dtype:
        raise RuntimeError(f"{sha}: frames {a.dtype}{a.shape} vs reference {b.dtype}{b.shape}")
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"sha256": sha, "modality": modality, "elems": int(d.size), "n_diff": int(np.count_nonzero(d)),
            "max_abs_diff": int(d.max())}


def compare_to_reference(out: Path, ref: Path, pool) -> dict:
    """New vs reference: token / mask / position / index byte-identical, the manifest's split / parity / tokenizer
    identical, media set and metadata identical (raise otherwise); returns only the frame-difference statistics."""
    for split in SPLITS:
        for name in TOKEN_FILES:
            if _sha256(out / split / name) != _sha256(ref / split / name):
                raise RuntimeError(f"{split}/{name} differs from reference {ref} (only media frames may differ)")
    mo, mr = (json.loads((d / "manifest.json").read_text()) for d in (out, ref))
    for k in ("tokenizer_sha256", "splits", "parity_set", "split", "max_tokens", "subset", "vision"):
        if mo[k] != mr[k]:
            raise RuntimeError(f"manifest[{k!r}] differs from the reference")

    def records(d):
        with open(d / "media.jsonl") as f:
            return {r["sha256"]: r for r in map(json.loads, f)}

    ro, rr = records(out), records(ref)
    if set(ro) != set(rr):
        raise RuntimeError(f"media set differs from the reference: extra {sorted(set(ro) - set(rr))[:3]}, missing {sorted(set(rr) - set(ro))[:3]}")
    for sha in ro:
        a = {k: ro[sha][k] for k in MEDIA_META if k in ro[sha]}
        b = {k: rr[sha][k] for k in MEDIA_META if k in rr[sha]}
        if a != b:
            keys = sorted(k for k in set(a) | set(b) if k not in a or k not in b or a[k] != b[k])
            raise RuntimeError(f"{sha}: media metadata differs from the reference: {keys}")
    diffs = list(pool.imap_unordered(_frame_diff, [(sha, ro[sha]["modality"], out, ref) for sha in sorted(ro)],
                                     chunksize=16))
    by_mod = {}
    for mod in ("image", "video"):
        ds = [d for d in diffs if d["modality"] == mod]
        by_mod[mod] = {"media": len(ds), "media_with_diff": sum(bool(d["n_diff"]) for d in ds),
                       "elems": sum(d["elems"] for d in ds), "diff_elems": sum(d["n_diff"] for d in ds),
                       "max_abs_diff": max((d["max_abs_diff"] for d in ds), default=0)}
    return {"reference": str(ref), "token_files_identical": list(TOKEN_FILES), "media_meta_identical": list(MEDIA_META),
            "media": len(diffs), "media_with_diff": sum(bool(d["n_diff"]) for d in diffs),
            "elems": sum(d["elems"] for d in diffs), "diff_elems": sum(d["n_diff"] for d in diffs),
            "max_abs_diff": max(d["max_abs_diff"] for d in diffs), "by_modality": by_mod,
            "top": sorted((d for d in diffs if d["n_diff"]), key=lambda d: -d["n_diff"])[:20]}


def check_device(device: str) -> str | None:
    """``--device`` matches the visible GPUs (cpu-emulated must see none, cuda exactly one); returns
    CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device in ("cpu-emulated", "cpu-exact") and visible != "":
        raise RuntimeError(f"--device {device} must not touch the GPU: set CUDA_VISIBLE_DEVICES to an empty string")
    if device == "cuda" and (not visible or "," in visible):
        raise RuntimeError(f"--device cuda needs CUDA_VISIBLE_DEVICES to be exactly one GPU, got {visible!r}")
    return visible


def process_media(pool, media_src: dict, out: Path, checks: collections.Counter) -> dict[str, dict]:
    """Runs ``_media_task`` in the pool (``_init_worker``) for every media item of ``media_src`` (sha256 → (modality,
    blob path)): writes ``out/media/…npz``, returns sha256 → record; each item's assertion counts are added to
    ``checks``."""
    t0 = time.time()
    records: dict[str, dict] = {}
    tasks = [(sha, mod, src, str(out)) for sha, (mod, src) in sorted(media_src.items())]
    for i, rec in enumerate(pool.imap_unordered(_media_task, tasks, chunksize=4), 1):
        records[rec["sha256"]] = rec
        checks.update(rec.pop("checks"))
        if i % 2000 == 0 or i == len(tasks):
            print(f"  media {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    return records


def segment_task(key: str, ids: np.ndarray, gen_spans, media: list[dict], records: dict[str, dict]) -> tuple:
    """Input of ``_segment_task``: the segment's media carry only the processor-record fields needed for alignment."""
    recs = {}
    for m in media:
        r = records[m["sha256"]]
        recs[m["sha256"]] = {k: r[k] for k in ("modality", "grid_thw", "block") if k in r}
    return key, ids, gen_spans, media, recs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", nargs="+", required=True, help="ledger instance directories or their parent (media/ must already be unpacked)")
    ap.add_argument("--manifests", nargs="*", default=[], help="the harness's tito_manifest.json (for joining tasks)")
    ap.add_argument("--model", required=True, help="model directory (tokenizer, processor config)")
    ap.add_argument("--out", required=True, help="output directory (must not exist)")
    ap.add_argument("--max-tokens", type=int, default=120_000)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--subset", type=int, default=0, help="only N segments (for development checks)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--max-gb", type=float, default=120.0, help="stop if the output exceeds this")
    ap.add_argument("--device", choices=tuple(DEVICE_PATHS), required=True,
                    help="cuda = deployed as is (CUDA_VISIBLE_DEVICES exactly one GPU); cpu-emulated = approximation for when the GPU is off limits")
    ap.add_argument("--compare-to", help="reference output directory: token files must be byte-identical; frame differences are counted")
    ap.add_argument("--routing", action="store_true",
                    help="attach per-position MoE routing (the ledger must be recorded with --session-ledger-routing): <split>/routing.i16, "
                         "960 B/token, about 250 GB for the full data; only for small subsets or RL batches")
    args = ap.parse_args(argv)
    visible = check_device(args.device)
    ref = Path(args.compare_to) if args.compare_to else None
    if ref is not None and not (ref / "manifest.json").is_file():
        raise FileNotFoundError(f"--compare-to {ref} has no manifest.json")

    from arcforge.tito.join import Joiner, load_manifests
    from arcforge.tito.ledger import Vocab, assemble_routing, instance_dirs, load_instance

    t_start = time.time()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} already exists: batches are not mixed")
    if shutil.disk_usage(out.parent).free < args.max_gb * 1e9 + 50e9:
        raise RuntimeError(f"{out.parent} has {shutil.disk_usage(out.parent).free / 1e9:.0f} GB free, "
                           f"not enough for --max-gb {args.max_gb} plus 50 GB headroom")
    out.mkdir(parents=True)
    timings: dict[str, float] = {}
    model_dir = Path(args.model)
    vocab = Vocab(model_dir)
    joiner = Joiner(load_manifests(args.manifests)) if args.manifests else None
    dirs = instance_dirs(args.ledger)
    env = _environment(model_dir, [d / "server.json" for d in dirs], args.device)
    print(f"fork {env['fork']} commit={env['fork_commit']} dirty={env['fork_dirty']}"
          f"  transformers {env['transformers']}  device {args.device} (CUDA_VISIBLE_DEVICES={visible!r})", flush=True)

    # ---- 1. ledger → candidate segments
    t0 = time.time()
    stats = collections.Counter()
    cands, segs, media_src = [], {}, {}
    for d in dirs:
        inst = load_instance(d, vocab, strict=True)
        for seg in inst.segments.values():
            stats["segments"] += 1
            if not seg.appends or not seg.gen_spans:
                stats["skip_no_gen"] += 1
                continue
            if len(seg) > args.max_tokens:
                stats["skip_over_max_tokens"] += 1
                stats["skip_over_max_tokens_tokens"] += len(seg)
                continue
            media = segment_media(seg)
            hit = joiner.lookup(seg) if joiner is not None else None
            thread = None if seg.sid is None else seg.thread
            group = hit.task if hit else (thread if thread is not None else f"single:{seg.key}")
            for m in media:
                media_src.setdefault(m["sha256"], (m["modality"], str(inst.media_path(m["sha256"]))))
                if media_src[m["sha256"]][0] != m["modality"]:
                    raise RuntimeError(f"{m['sha256']} is both image and video")
            cands.append({
                "key": seg.key, "instance": seg.instance, "seg": seg.seg, "sid": seg.sid, "thread": thread,
                "window": -1 if seg.sid is None else seg.window, "task": hit.task if hit else None,
                "harness": hit.manifest.harness if hit else None, "group": group, "cause": seg.cause,
                "split_reason": seg.split_reason, "close_reason": None if seg.close is None else seg.close["reason"],
                "closed": seg.status == "closed", "length": len(seg), "media": media,
                "n_appends": len(seg.appends), "n_rewind_appends": sum(a.rewind > 0 for a in seg.appends),
                "n_stop_substituted": sum(a.stop_substituted for a in seg.appends),
                "n_dropped": sum(s.dropped for s in seg.gen_spans),
                "gen_spans": [(s.start, s.end) for s in seg.gen_spans]})
            if args.routing:
                cands[-1].update(routing_refs=[a.routing for a in seg.appends], ledger_dir=str(inst.path))
            segs[seg.key] = seg.ids.astype("<i4")
        del inst
    if args.subset:
        cands = _select_subset(cands, args.subset, args.seed)
        keep = {c["key"] for c in cands}
        segs = {k: v for k, v in segs.items() if k in keep}
        used = {m["sha256"] for c in cands for m in c["media"]}
        media_src = {k: v for k, v in media_src.items() if k in used}
    for c in cands:
        c["split"] = split_of(c["group"], args.seed, args.val_frac)
    routing_shape = None
    if args.routing:
        missing = [c["key"] for c in cands if any(r is None for r in c["routing_refs"])]
        if missing:
            raise RuntimeError(f"--routing: {len(missing)} segments have appends without recorded routing, e.g. {missing[:3]}")
        shapes = {r.shape[1:] for c in cands for r in c["routing_refs"]}
        if len(shapes) != 1:
            raise RuntimeError(f"--routing: more than one routing (layers, top_k): {shapes}")
        routing_shape = shapes.pop()
        routing_bytes = sum(c["length"] for c in cands) * int(np.prod(routing_shape)) * 2
        if routing_bytes > args.max_gb * 1e9:
            raise RuntimeError(f"routing {routing_bytes / 1e9:.1f} GB exceeds --max-gb {args.max_gb}")
        # the ledger's model is --model (_environment compared model_files_sha256): expert ids are its expert numbering
        num_experts = int(json.loads((model_dir / "config.json").read_text())["text_config"]["num_experts"])
    else:
        routing_bytes = 0
    timings["ledger_seconds"] = time.time() - t0
    print(f"candidates: {len(cands)} segments, unique media {len(media_src)} ({timings['ledger_seconds']:.0f}s)", flush=True)

    ctx = mp_ctx.get_context("spawn")
    server_json = str(dirs[0] / "server.json")
    checks = collections.Counter()
    with FailLoudPool(ctx, args.workers, _init_worker, (server_json, 1, args.device)) as pool:
        # ---- 2. media
        t0 = time.time()
        records = process_media(pool, media_src, out, checks)
        timings["media_seconds"] = time.time() - t0
        media_bytes = sum(r["file_bytes"] for r in records.values())
        if media_bytes + routing_bytes > args.max_gb * 1e9:
            raise RuntimeError(f"media {media_bytes / 1e9:.1f} GB + routing {routing_bytes / 1e9:.1f} GB "
                               f"exceeds --max-gb {args.max_gb}")
        # ---- 3. segments: alignment, loss, M-RoPE → write the splits
        t0 = time.time()
        writers = {}
        for split in SPLITS:
            (out / split).mkdir()
            writers[split] = {"ids": open(out / split / "input_ids.i32", "wb"),
                              "loss": open(out / split / "loss_mask.u8", "wb"),
                              "pos": open(out / split / "mrope.i32", "wb"),
                              "index": open(out / split / "index.jsonl", "w"), "offset": 0}
            if args.routing:
                writers[split]["routing"] = open(out / split / "routing.i16", "wb")
        by_key = {c["key"]: c for c in cands}
        seg_tasks = [segment_task(c["key"], segs[c["key"]], c["gen_spans"], c["media"], records) for c in cands]
        slow_seg = (0.0, None)
        rows = []
        for i, (key, ids, loss, pos, seg_checks, sec) in enumerate(
                pool.imap(_segment_task, seg_tasks, chunksize=2), 1):
            c = by_key[key]
            if not np.array_equal(ids, segs[key]):
                raise RuntimeError(f"{key}: ids returned by the worker differ from the ledger")
            checks.update(seg_checks)
            slow_seg = max(slow_seg, (sec, key))
            w = writers[c["split"]]
            w["ids"].write(ids.tobytes())
            w["loss"].write(loss.tobytes())
            w["pos"].write(np.ascontiguousarray(pos).tobytes())
            if args.routing:
                w["routing"].write(assemble_routing(Path(c["ledger_dir"]), key, c["routing_refs"], len(ids)).tobytes())
            row = {k: c[k] for k in ("key", "instance", "seg", "sid", "thread", "window", "task", "harness", "group",
                                     "cause", "split_reason", "close_reason", "closed", "length", "n_appends",
                                     "n_rewind_appends", "n_stop_substituted", "n_dropped", "media")}
            row.update(offset=w["offset"], n_loss=int(loss.sum()), n_gen_spans=len(c["gen_spans"]))
            w["index"].write(json.dumps(row, ensure_ascii=False) + "\n")
            w["offset"] += len(ids)
            rows.append(dict(row, split=c["split"], n_images=sum(m["modality"] == "image" for m in c["media"]),
                             n_videos=sum(m["modality"] == "video" for m in c["media"])))
            if i % 500 == 0 or i == len(seg_tasks):
                print(f"  segments {i}/{len(seg_tasks)} ({time.time() - t0:.0f}s)", flush=True)
        for w in writers.values():
            for f in ("ids", "loss", "pos", "index", *(("routing",) if args.routing else ())):
                w[f].close()
        timings["segment_seconds"] = time.time() - t0

    # ---- 4. manifest / report
    with open(out / "media.jsonl", "w") as f:
        for sha in sorted(records):
            f.write(json.dumps(records[sha]) + "\n")
    splits = {}
    for split in SPLITS:
        rs = [r for r in rows if r["split"] == split]
        splits[split] = {"samples": len(rs), "tokens": sum(r["length"] for r in rs),
                         "loss_tokens": sum(r["n_loss"] for r in rs), "groups": len({r["group"] for r in rs}),
                         "with_images": sum(bool(r["n_images"]) for r in rs),
                         "with_videos": sum(bool(r["n_videos"]) for r in rs)}
    overlap = {r["group"] for r in rows if r["split"] == "train"} & {r["group"] for r in rows if r["split"] != "train"}
    if overlap:
        raise RuntimeError(f"train / validation share groups: {sorted(overlap)[:5]}")
    resized = [r for r in records.values() if r["bicubic_calls"]]
    if args.device == "cuda" and resized:
        raise RuntimeError("cuda mode should have no emulated bicubic stats")
    slow_media = max(records.values(), key=lambda r: r["seconds"])
    vision = vision_spec(vocab)
    digest =hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
    parity = _parity_set(rows)
    if not args.subset and len(parity) != sum(len(q) for _, q in PARITY_PLAN):
        raise RuntimeError(f"parity subset got only {len(parity)} segments: {[p['reason'] for p in parity]}")
    # The descriptive text values below are written into manifest.json and kept verbatim as recorded ("split" is
    # compared by --compare-to). Gloss: split.key "group = task (arc2 task / arc3 game), else thread"; loss_mask "ledger
    # Segment.gen_spans (tokens actually sampled; rewound tails, template-filled tokens, tool output, prompt, pads do not
    # count)"; mrope "..., from 0 within the segment; stored (3, T) int32, read as int64"; routing.layout "(T, layers,
    # top_k) concatenated per token, same offsets as input_ids"; routing.rows "position p = the experts chosen with token
    # p as input (global logical id); the last position of each segment was never forwarded, filled with -1";
    # routing.source "ledger routing, arcforge.tito.ledger.assemble_routing (a later append overwrites an earlier one)".
    manifest = {
        "format": FORMAT, "created_unix": time.time(), "argv": sys.argv, "tokenizer_sha256": vocab.sha256,
        "ledger": [str(d) for d in dirs],
        "manifests": list(args.manifests), "max_tokens": args.max_tokens,
        "split": {"seed": args.seed, "val_frac": args.val_frac, "key": "group＝task（arc2 題目／arc3 遊戲），否則 thread",
                  "hash": "sha256('tito-mm-split:{seed}:{group}')[:8] / 2^64 < val_frac → validation"},
        "splits": splits, "subset": args.subset, "processor_digest": digest, "processor": env, "vision": vision,
        "parity_set": parity,
        "loss_mask": "帳本 Segment.gen_spans（真的取樣出來的 token；回捲尾端、模板補的 token、tool 輸出、prompt、pad 都不算）",
        "mrope": "sglang MRotaryEmbedding.get_rope_index（qwen4_exp），段內從 0 起；(3, T) int32 存、int64 讀",
        "routing": None if routing_shape is None else {
            "layers": int(routing_shape[0]), "top_k": int(routing_shape[1]), "dtype": "int16",
            "num_experts": num_experts,
            "layout": "(T, layers, top_k) 逐 token 串接，與 input_ids 同 offset",
            "rows": "位置 p＝把 token p 當輸入時選到的 expert（全域 logical id）；每段最後一個位置從沒 forward 過，填 -1",
            "source": "帳本 routing，arcforge.tito.ledger.assemble_routing（後一筆 append 蓋前一筆）"},
    }
    sizes = {"media_npz_bytes": media_bytes, "media_u8_bytes": sum(r["u8_bytes"] for r in records.values())}
    for split in SPLITS:
        for name in ("input_ids.i32", "loss_mask.u8", "mrope.i32", "index.jsonl",
                     *(("routing.i16",) if args.routing else ())):
            sizes[f"{split}/{name}"] = (out / split / name).stat().st_size
    timings["total_seconds"] = time.time() - t_start
    report = {
        "ledger_stats": dict(stats), "candidates": len(cands), "checks": dict(sorted(checks.items())),
        "media": {"unique": len(records), "images": sum(r["modality"] == "image" for r in records.values()),
                  "videos": sum(r["modality"] == "video" for r in records.values()),
                  "refs": sum(len(c["media"]) for c in cands),
                  "device": args.device,
                  "bicubic_emulation": None if args.device == "cuda" else {
                      "resized_by_bicubic": len(resized),
                      "bicubic_sum": {k: sum(r[k] for r in resized) for k in RISK_KEYS},
                      "bicubic_media_with": {k: sum(bool(r[k]) for r in resized) for k in RISK_KEYS[2:]}}},
        "splits": splits, "sizes": sizes, "timings": timings,
        "slowest": {"media": {"sha256": slow_media["sha256"], "modality": slow_media["modality"],
                              "grid_thw": slow_media["grid_thw"], "seconds": slow_media["seconds"]},
                    "segment": {"key": slow_seg[1], "seconds": slow_seg[0]}},
        "join": {"joined": sum(c["task"] is not None for c in cands),
                 "harness": dict(collections.Counter(c["harness"] for c in cands))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    # ---- 5. read-back: the reader opens each split, reads the few longest rows and the parity set; ids / mask must
    # equal the ledger; timed
    from tito_mm_data import TitoMMDataset

    t0 = time.time()
    worst = 0.0
    parity_keys = {p["key"] for p in manifest["parity_set"]}
    n_read = 0
    for split in SPLITS:
        ds = TitoMMDataset(out, split, routing=args.routing)
        order = list(np.argsort(-ds.lengths)[:3]) + [i for i, r in enumerate(ds.index) if r["key"] in parity_keys]
        for i in order:
            t1 = time.time()
            item = ds[int(i)]
            worst = max(worst, time.time() - t1)
            if not np.array_equal(item["input_ids"], segs[item["key"]]):
                raise RuntimeError(f"{item['key']}: read-back input_ids differ from the ledger")
            c = by_key[item["key"]]
            if not np.array_equal(item["loss_mask"], loss_mask_from_spans(len(segs[item["key"]]), c["gen_spans"]) == 1):
                raise RuntimeError(f"{item['key']}: read-back loss_mask differs from the gen spans")
            if args.routing and not np.array_equal(
                    item["routing"], assemble_routing(Path(c["ledger_dir"]), c["key"], c["routing_refs"], c["length"])):
                raise RuntimeError(f"{item['key']}: read-back routing differs from the ledger")
            n_read += 1
    report["readback"] = {"samples": n_read, "max_seconds": worst, "seconds": time.time() - t0}
    if ref is not None:
        t0 = time.time()
        with ctx.Pool(args.workers) as pool:
            report["compare_to"] = compare_to_reference(out, ref, pool)
        report["compare_to"]["seconds"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"checks": report["checks"], "splits": splits, "media": report["media"],
                      "sizes_gb": sum(sizes.values()) / 1e9, "timings": timings,
                      "slowest": report["slowest"], "readback": report["readback"],
                      "compare_to": {k: v for k, v in report.get("compare_to", {}).items() if k != "top"}},
                     indent=1, ensure_ascii=False), flush=True)
    if shutil.disk_usage(out).free < 50e9:
        print("warning: less than 50 GB left on /data after output", flush=True)


if __name__ == "__main__":
    main()
