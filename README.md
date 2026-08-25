# ComfyUI-MinimaxH3-Latent-Upres

**Upscale long, high-resolution MiniMax H3 video on any GPU — in a single node.**
A fork of [bbaudio-2025/Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale) with high-VRAM optimization, memory-safety fixes, and ComfyUI ≥ 0.33.0 compatibility.

> Based on [Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale) (MIT).
> The underlying mechanics originate from [Comfyui-MiniMax-H3-LatentSplit](https://github.com/bbaudio-2025/Comfyui-MiniMax-H3-LatentSplit) and [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler).

---

## What it does

Re-samples (enhances / upscales) an already-denoised MiniMax H3 AV latent through a full auto pipeline:

```
input AV latent
  → temporal split (outer loop)
  →   latent upscale  (H3 3D upscaler, per chunk, video only)
  →   spatial split   (inner loop)
  →   per-tile diffusion sampling
  →   spatial stitch
  → temporal stitch
  → output AV latent
```

MiniMax H3 video is a nested latent bundling 24-channel video **and** 32-channel audio in one tensor. Standard upscale nodes don't understand this structure. This node keeps the audio track intact through every stage while the video is tiled, re-sampled with the original conditioning, and stitched back.

---

## Why this version

### 1. `keep_models_resident` — high-VRAM fast path (new)

The upstream unloads the 3D upscaler to CPU after **every** temporal chunk and offloads the diffusion model during each upscale. On cards with plenty of VRAM, that GPU↔CPU ping-pong is wasted transfer time.

New boolean input on **MMH3 Latent Upscale with Model Params**:

| Setting | Behavior | Recommended for |
|---|---|---|
| **OFF** (default) | Original per-chunk offload/reload | 12–16 GB cards |
| **ON** | Upscaler stays on GPU for the full pass; diffusion model not unloaded; upscaler unloaded once at the end | 24 GB+ cards |

### 2. LRU model cache

Upstream keeps every loaded upscaler checkpoint in an unbounded dict for the whole session. This fork caps the cache at 3 entries with LRU eviction, so loading several checkpoints can't silently eat your memory.

### 3. Tile progress bar

The spatial tiling loop now reports per-tile progress (`Tile N/M (row,col)`) through ComfyUI's progress bar — no more silent multi-tile waits.

### 4. ComfyUI ≥ 0.33.0 compatibility (crash fix)

Upstream crashes on ComfyUI 0.33.0 with:

```
TypeError: ProgressBar.update_absolute() got an unexpected keyword argument 'comment'
```

This fork detects the new API at runtime and falls back to the two-argument form. Works on both old and new ComfyUI versions.

### 5. Minor hardening

- `torch.cuda.empty_cache()` only called when CUDA is actually available.

---

## Nodes

| Node | Role |
|---|---|
| **MMH3 Ultimate Upscale** | Main node. Runs the whole loop. Inputs: `latent`, `conditioning`, `model`, `noise`, `sampler`, `sigmas`, optional `negative` + `cfg`, and three optional param inputs. |
| **MMH3 Temporal Split Params** | `chunk_length` (px frames, multiple of 17), `temporal_overlap` (multiple of 17), `anchor_strength`. |
| **MMH3 Spatial Split Params** | `tile_width` / `tile_height` (px, multiple of 32), overlaps, fade widths, `min_tile_size`, `overlap_mode`, `overlap_blend`. |
| **MMH3 Latent Upscale with Model Params** | H3 3D model upscaler: `model_name`, `width`, `height`, `device`, `precision`, **`keep_models_resident`** (new). |
| **MMH3 Latent Upscale Params** | Model-free interpolation: `method`, `width`, `height`. |

> `negative` is **optional** and MiniMax H3 does not use negative conditioning — leave it unconnected.

### Typical workflow

1. Generate an H3 AV latent with MiniMax H3 (video + audio in one latent).
2. (Optional) `MMH3 Temporal Split Params` → connect to `temporal_split_param`.
3. (Optional) `MMH3 Latent Upscale with Model Params` **or** `MMH3 Latent Upscale Params` → connect to `latent_upscale_param`.
4. (Optional) `MMH3 Spatial Split Params` → connect to `spatial_split_param`.
5. Feed `latent`, `conditioning`, `model`, `noise`, `sampler`, `sigmas` into `MMH3 Ultimate Upscale`.
6. Decode the output latent with the H3 VAE.

> The `width`/`height` you set for upscaling must match the **conditioning's generation size** (the size the video was conditioned at, after upscale).

---

## Install

1. Place this folder into `ComfyUI/custom_nodes/`.
2. Put your H3 latent upscaler checkpoint(s) into `ComfyUI/models/latent_upscale_models/` (e.g. `minimax_h3_latent_upscaler_3d_*.safetensors`). Model-free interpolation mode needs no checkpoint.
3. **Restart ComfyUI fully.** After updating any custom node, kill the old ComfyUI process and start fresh — Python keeps the old module in memory until the process exits.

---

## Design details (inherited from upstream)

- **Temporal consistency** — at each chunk seam (except the first), the frame-0 keyframe is pinned to the previous chunk's re-sampled boundary frame; overlapping chunks are cross-faded.
- **Spatial seam integrity** — overlap strips shared with already-stitched tiles are pre-filled and frozen via `noise_mask`, so re-sampling can only change free interior regions. Bands are blended with configurable mode (`linear` / `smoothstep` / `overwrite` / `midpoint`) and ownership (`earlier` / `later`).
- **Audio preserved** — the audio branch is carried through unchanged on every chunk and stitch; never re-sampled.
- **Optional stages** — leave any param input unconnected to skip that stage (no upscale / single chunk / whole-chunk sampling).

## Limits

- Single-video latent only (batch 1).
- No resume if a run is interrupted.
- Uniform resolution across all chunks.
- The 3D upscaler only upscales (effective scale ≥ 1.0).

## Verified environment

- ComfyUI v0.33.0, Python 3.12 (conda-forge), PyTorch 2.13.0+cu130, NVIDIA RTX PRO 6000 Blackwell (96 GB).
- Also compatible with older ComfyUI versions (pre-0.33.0) via the progress-bar fallback.

---

## Credits

- Upstream: [bbaudio-2025/Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale) (MIT)
- Temporal/spatial split, anchor & append mechanics: [Comfyui-MiniMax-H3-LatentSplit](https://github.com/bbaudio-2025/Comfyui-MiniMax-H3-LatentSplit)
- H3 3D upscaler checkpoints & inference: [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)

## License

MIT (see LICENSE).
