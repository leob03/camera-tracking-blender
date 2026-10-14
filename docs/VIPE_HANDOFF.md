# Reviewed ViPE source and Crafty handoff

## Source to mirror

- Branch: `study-transfer/2026-10-01`
- Commit: `a5d30f0978afec9f8262622425f994b5ba573a1f`
- Parent: `44ad630bc4e01666ace74cbccdacddc4de2fc0be`
- Upstream base: `95a8816947602ddc26fcb7a80bea4f9313059578`

The 2026-10-14 snapshot commits the exact eight modified and four new source/test files present in the reviewed study checkout. It changes no inference behavior relative to that working tree. All previously recorded worker source hashes still match. Mirror this commit and its ancestry into the team's ViPE fork. The mirror itself is managed by the integration team; this update did not publish ViPE to a new remote.

The pending changes provide disk-backed frame and video-depth storage, streaming artifact export, camera-stage preservation before optional work, CPU depth alignment, cached visualization reads, a frozen-cache export utility and regression tests. See [the receipt and file hashes](../provenance/vipe-source-snapshot.json).

## Is the ViPE folder sufficient?

**It contains the reviewed ViPE source and configs. It is not the complete ComfyUI worker deployment.** The current worker requires these separately packaged components:

| Component | Required when | How it is supplied |
| --- | --- | --- |
| Reviewed ViPE source, including root `configs/` | Always | Fork at the commit above; `vipe_root` must point to this checkout |
| ViPE Python environment and compiled extensions | Always | Rebuild for the chosen Python, Torch/CUDA and worker GPU architectures; configure `vipe_python` |
| `comfyui_vipe_remote/` node pack | Always | This repository, installed under ComfyUI custom nodes; includes `worker.py`, `dense_stage.py`, `evidence_stage.py`, `log_tail.py`, and `camera_evidence/` |
| Model weights and tokenizer/config files | According to enabled features | DVC/model volume via `torch_home`, `hf_home`, `cotracker_checkpoint`; [exact inventory](MODELS.md) |
| CoTracker source | Review witnesses enabled | Local checkout via `cotracker_repository`; commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` |
| FFmpeg and FFprobe | Always | Configured executable paths in the worker image |
| ComfyUI fork, node configuration and writable job directories | Always | Existing Crafty deployment; `VIPE_WORKER_CONFIG` points to its worker JSON |
| Blender add-on and cross asset | Artist client | Separate Blender add-on package; no Blender installation on the inference worker |

The required evidence code is already included in `comfyui_vipe_remote/camera_evidence/`. The worker does not need the study's separate `camera-evidence/` checkout, review portal, Nuke installation, old footage/results, or workstation support scripts. The local FFmpeg wrapper is a development path choice; the production config should use its own working FFmpeg binary.

The development `.venv` resolves to an environment outside the ViPE folder. Rebuilding from the effective pins and recompiling extensions is the correct way to remove that dependency; copying or mirroring `.venv` is insufficient. Keep the whole source checkout available because the current worker composes Hydra configuration from `<vipe_root>/configs`, even if the Python package is installed.

## Offline build details

- **zstandard:** `DiskFrameStore` imports it at runtime. The current `uv.lock` includes `zstandard==0.25.0` transitively through `viser`; it is not listed directly in `pyproject.toml`. Keep it in any minimized production dependency set. The preflight imports it explicitly.
- **Eigen:** `vipe/ext/specs.py` downloads Eigen 3.4.0 when `csrc/include/eigen3/Eigen` is absent. That generated include directory is Git-ignored. For a build without network access, stage those headers before compiling, or supply system Eigen headers with the appropriate include path and `USE_SYSTEM_EIGEN=1`. The loader's download source is `https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz`. A fork mirror alone does not package these generated headers.
- **Native extensions:** build them for the actual container Python, Torch/CUDA and GPU architectures. The study's prebuilt sm_75 extension is not the production build artifact.
- **Model caches:** preserve the specified cache layout, auxiliary files and Hugging Face refs. The worker sets offline Hugging Face flags; missing Torch/gdown files can still trigger download attempts. See [model staging](MODELS.md#dvc-and-container-paths).
- **Runtime directories:** input, output, scratch and evidence cache must be writable. Use worker-local scratch for `VIPE_FRAME_CACHE_DIR`. Cancellation or a killed worker may leave scratch files; model storage and job storage have separate lifecycles.

No dependency or inference changes were made as part of this source snapshot.

## Validation and timing

The snapshot's 12 Python files parsed successfully. Four disk/depth regression tests passed, including lossless frame storage, array overlap/stack parity, CPU/GPU alignment parity and disk/memory temporal-depth parity. The streaming export test also passed, comparing camera, depth, mask and decoded RGB outputs with the existing exporter. Its first attempt failed because the inherited `/usr/bin/ffmpeg` path did not exist; rerunning with the worker's configured `IMAGEIO_FFMPEG_EXE` passed.

No full shot or production-container test was rerun for this commit. Geronimo separately reported successful camera, witnesses, dense reconstruction and cancellation on his dev worker, with rebuilt Python 3.10.16 / Torch 2.9.0+cu128 extensions. That is integration-team evidence, not a test repeated by this snapshot audit.

The supplied message mentions both approximately 35 seconds and a job summary of 100 seconds for a 262-frame plate with 26,137 points. These should remain separate observations until the job ID, feature settings and timing boundaries are matched. Source resolution alone does not establish inference resolution; include `max_dimension` and witness/dense settings when comparing runs.
