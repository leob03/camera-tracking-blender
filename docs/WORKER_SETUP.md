# ComfyUI worker setup

## Prerequisites

- Linux NVIDIA/CUDA worker, FFmpeg and FFprobe.
- A working installation of your approved ComfyUI fork. The local test used ComfyUI 0.37.2 and frontend 1.52.7.
- A **separate, working reviewed ViPE environment**. Tested: Python 3.10, Torch 2.9.0+cu128, torchvision 0.24.0+cu128, ViPE 1.2.0 with its compiled extensions.
- Cached model assets for each enabled stage; see [offline preparation](OFFLINE.md).

### Required ViPE compatibility

The tested checkout reports commit `44ad630bc4e01666ace74cbccdacddc4de2fc0be` on the internal restored study branch, based on upstream `95a8816947602ddc26fcb7a80bea4f9313059578`. This is a modified source tree; the commit alone does not certify every runtime file. Selected file hashes are in `provenance/worker-runtime.json`.

Obtain the reviewed ViPE source/environment from the study handoff. It includes stage snapshots (`save_slam_stage_artifacts`, initial/pre/post-BA intrinsics), stock sparse-track configuration, disk-backed frame/depth caches, bounded TrackAnything encoder batching, and local model-loading fixes. **Do not assume `pip install nvidia-vipe` or a fresh upstream clone reproduces this setup.** The ViPE fork is not republished by this add-on repository. Publishing or replacing that fork is a separate integration decision.

Run the preflight in the ViPE environment after filling in the configuration:

```bash
/opt/vipe/.venv/bin/python scripts/check_worker.py --config /opt/camera-tracking/worker-config.json
```

Preflight checks interfaces and configured files. A passing preflight is not an inference or offline-model test.

## Install the node

```bash
python scripts/build_packages.py
```

Extract `dist/comfyui_vipe_remote-0.3.1.zip` so that ComfyUI has `custom_nodes/comfyui_vipe_remote/__init__.py`. Copy `worker-config.example.json` to a private `worker-config.json` and set the actual paths. Use worker-local SSD scratch for `work_directory` and `evidence_cache_directory`.

Set `VIPE_WORKER_CONFIG` in the ComfyUI server environment, then restart ComfyUI:

```bash
export VIPE_WORKER_CONFIG=/opt/camera-tracking/worker-config.json
/path/to/comfy-python /opt/ComfyUI/main.py --listen 127.0.0.1 --port 8189 --cpu --cache-none
```

Use your normal fork launch arguments for base/input/output/user directories and node allowlists. The example runs the controller on CPU; the ViPE child process still uses CUDA. It is not necessary to install ViPE model packages in ComfyUI's Python environment. This node itself uses the standard library plus ComfyUI's existing aiohttp and APIs.

Confirm `GET /vipe/capabilities`, open the supplied workflow in the browser, upload a movie, and run it. Try 48 frames at 640 pixels for connectivity, then a full shot at 1280 for review. Explicitly test witness off/on and dense reconstruction off/on.

## Fork requirements

Core ComfyUI source and requirements were not changed for this integration. The node uses `/prompt`, `/history`, `/view`, the prompt queue, progress reporting and model unloading. The Blender client additionally depends on this fork accepting its supplied prompt UUID and supporting **targeted** `POST /api/jobs/{id}/cancel`.

If your fork lacks targeted cancellation, adapt that operation in the shared bridge before rollout. Do not replace it with a global interrupt that could stop another artist's job. Route `/vipe/upload`, `/vipe/jobs/*`, `/vipe/capabilities` and normal Comfy endpoints through the existing authenticated gateway. Configure its upload size and timeout for video files. This node pack does not implement a production authentication service.

## Runtime behavior

One prompt starts a worker process group. Camera solve, optional witnesses and optional dense depth run sequentially in separate child processes. ComfyUI models are unloaded before the worker starts. Cancellation terminates the entire job process group. Low scratch space (2 GiB) or low available RAM (1.5 GiB) stops a job with an error.

The witness cache is keyed by source SHA256 and checks recipe/checkpoint metadata; it stores independent observations, never a replacement camera. Fresh evidence currently processes the full source video (maximum 1500 source frames), even when the solve selects a shorter range. Restricting the input movie itself shortens that workload.

Dense reconstruction reuses the prepared plate and actual solve masks, computes stock adaptive video depth, then samples up to 12 views. Depth processing is capped at a 1280-pixel long edge. It rejects dynamic/sky pixels, dilates exclusions, rejects depth edges and creates unmerged surfaces. The earlier 4K environment study used a higher depth resolution; visual quality is not guaranteed identical.

Jobs keep logs, request/configuration artifacts, camera/native-map outputs and stage timings on local scratch. There is no automatic disk retention policy yet. Stop jobs before manually cleaning their directories. Completed downloads can be reimported without the server; service restarts do not guarantee queued/running jobs survive.
