"""ViT position-embedding interpolation exactly as the deployed SGLang computes it.

FN's ViT in SGLang is ``Qwen3VLMoeVisionModel``. Its default (eager) forward -- TITO collection and every
deployment script leave ``SGLANG_VIT_ENABLE_CUDA_GRAPH`` unset -- calls ``fast_pos_embed_interpolate_from_list``
(fewer than 6 media per ViT call) or ``fast_pos_embed_interpolate_vectorized`` (6 or more). Both resample the
learned 48x48 table at ``torch.linspace(0, side - 1, n)`` (align_corners=True, the same geometry as HF), cast
the four bilinear weights to the embedding dtype and multiply-accumulate in bf16.
``--enable-precise-embedding-interpolation`` is read only on the CUDA-graph path (half-pixel when off), which
deployments do not take. HF transformers 5.16.1 keeps the weights in fp32, so it differs by bf16 rounding.

The HF vision forward accepts precomputed ``interp_indices``/``interp_weights`` (shape (patches, 4), in the
patch order it consumes), which is how training gets SGLang's numbers. probes/vision_interp_parity.py checks
this port against both SGLang functions; test_vision_interp.py checks the patch order against HF's own.
"""

import torch


def sglang_vit_interp(grid_thw, side, merge, dtype, device):
    """(indices[P, 4] int64, weights[P, 4] dtype) for media in order, P = sum t*h*w."""
    all_indices, all_weights = [], []
    for t, h, w in grid_thw:
        if h % merge or w % merge:
            raise ValueError(f"grid {(t, h, w)} is not a multiple of spatial merge {merge}")
        # The same ops in the same order as SGLang's loop version, so the weights round identically.
        h_idxs = torch.linspace(0, side - 1, h, dtype=torch.float32, device=device)
        w_idxs = torch.linspace(0, side - 1, w, dtype=torch.float32, device=device)
        h_floor = h_idxs.to(torch.long)
        w_floor = w_idxs.to(torch.long)
        h_ceil = torch.clamp(h_floor + 1, max=side - 1)
        w_ceil = torch.clamp(w_floor + 1, max=side - 1)
        dh, dw = torch.meshgrid(h_idxs - h_floor, w_idxs - w_floor, indexing="ij")
        h_floor, w_floor = torch.meshgrid(h_floor, w_floor, indexing="ij")
        h_ceil, w_ceil = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1 - dh - w01
        indices = torch.stack([h_floor * side + w_floor, h_floor * side + w_ceil,
                               h_ceil * side + w_floor, h_ceil * side + w_ceil], dim=-1).reshape(-1, 4)
        weights = torch.stack([w00, w01, w10, w11], dim=-1).reshape(-1, 4).to(dtype)
        order = torch.arange(h * w, device=device).view(h // merge, merge, w // merge, merge)
        order = order.permute(0, 2, 1, 3).flatten()
        all_indices.append(indices[order].repeat(t, 1))
        all_weights.append(weights[order].repeat(t, 1))
    return torch.cat(all_indices), torch.cat(all_weights)
