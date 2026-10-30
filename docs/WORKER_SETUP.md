# ComfyUI worker setup

## Prerequisites

- Linux NVIDIA/CUDA worker, FFmpeg and FFprobe.
- A working installation of your approved ComfyUI fork. The local test used ComfyUI 0.37.2 and frontend 1.52.7.
- A **separate, working reviewed ViPE environment**. Tested: Python 3.10, Torch 2.9.0+cu128, torchvision 0.24.0+cu128, ViPE 1.2.0 with its compiled extensions.
- Cached model assets for each enabled stage; see [offline preparation](OFFLINE.md).

### Required ViPE compatibility

Pin ViPE to **`a5d30f0978afec9f8262622425f994b5ba573a1f`** on `study-transfer/2026-10-01`, based on upstream `95a8816947602ddc26fcb7a80bea4f9313059578`. This commits the exact previously reviewed state: parent `44ad630` plus eight modified and four new files. All previously recorded worker source hashes match. See [the source and dependency handoff](VIPE_HANDOFF.md), `provenance/vipe-source-snapshot.json` and `provenance/worker-runtime.json`.

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

Extract `dist/comfyui_vipe_remote-0.5.0.zip` so that ComfyUI has `custom_nodes/comfyui_vipe_remote/__init__.py`. Copy `worker-config.example.json` to a private `worker-config.json` and set the actual paths. Use worker-local SSD scratch for `work_directory` and `evidence_cache_directory`.

Set `VIPE_WORKER_CONFIG` in the ComfyUI server environment, then restart ComfyUI:

```bash
export VIPE_WORKER_CONFIG=/opt/camera-tracking/worker-config.json
/path/to/comfy-python /opt/ComfyUI/main.py --listen 127.0.0.1 --port 8189 --cpu --cache-none
```

Use your normal fork launch arguments for base/input/output/user directories and node allowlists. The example runs the controller on CPU; the ViPE child process still uses CUDA. It is not necessary to install ViPE model packages in ComfyUI's Python environment. This node itself uses the standard library plus ComfyUI's existing aiohttp and APIs.

Optional Gaussian splat training adds no Python packages or models; it needs a native LichtFeld build and one worker JSON field. See [Gaussian splats → Worker setup and production deployment](GAUSSIAN_SPLATS.md#worker-setup-and-production-deployment).

Confirm `GET /vipe/capabilities`, open the supplied workflow in the browser, upload a movie, and run it. Try 48 frames at 640 pixels for connectivity, then a full shot at 1280 for review. Explicitly test witness off/on and dense reconstruction off/on.

## Fork requirements

Core ComfyUI source and requirements were not changed for this integration. The node uses `/prompt`, `/history`, `/view`, the prompt queue, progress reporting and model unloading. The Blender client additionally depends on this fork accepting its supplied prompt UUID and supporting **targeted** `POST /api/jobs/{id}/cancel`.

If your fork lacks targeted cancellation, adapt that operation in the shared bridge before rollout. Do not replace it with a global interrupt that could stop another artist's job. Route `/vipe/upload`, `/vipe/jobs/*`, `/vipe/capabilities` and normal Comfy endpoints through the existing authenticated gateway. Configure its upload size and timeout for video files. This node pack does not implement a production authentication service.

## Runtime behavior

One prompt starts a worker process group. Camera solve, optional witnesses and optional dense depth run sequentially in separate child processes. ComfyUI models are unloaded before the worker starts. Cancellation terminates the entire job process group. Low scratch space (2 GiB) or low available RAM (1.5 GiB) stops a job with an error.

The witness cache is keyed by source SHA256 and checks recipe/checkpoint metadata; it stores independent observations, never a replacement camera. Fresh evidence currently processes the full source video (maximum 1500 source frames), even when the solve selects a shorter range. Restricting the input movie itself shortens that workload.

Dense reconstruction reuses the prepared plate and actual solve masks, computes stock adaptive video depth, then samples up to 12 views. Depth processing is capped at a 1280-pixel long edge. It rejects dynamic/sky pixels, dilates exclusions, rejects depth edges and creates unmerged surfaces. The earlier 4K environment study used a higher depth resolution; visual quality is not guaranteed identical.

Jobs keep logs, request/configuration artifacts, camera/native-map outputs and stage timings on local scratch. There is no automatic disk retention policy yet. Stop jobs before manually cleaning their directories. Completed downloads can be reimported without the server; service restarts do not guarantee queued/running jobs survive.

## Production tuning (node pack 0.4.0)

The defaults keep the reviewed behaviour of the 16 GB development card. On larger workers, adjust these optional `worker-config.json` keys and compare timings:

| Key | Default | Effect |
| --- | --- | --- |
| `cpu_threads` | 2 | OMP/OpenBLAS/MKL and Torch CPU threads for every stage |
| `aot_encoder_batch_size` | 1 | TrackAnything DeAOT encoder batch (`VIPE_TRACK_ANYTHING_AOT_ENCODER_BATCH_SIZE`); larger uses more VRAM |
| `depth_alignment_device` | `cpu` | Depth alignment device for the original video-depth mode (`cuda` on large GPUs) |
| `async_prefetch` / `prefetch_queue_size` | false / 1 | Overlap decoding and annotation with SLAM; uses more GPU memory |

## What changed in 0.4.0

- **Content-addressed uploads.** Uploads are stored as `input/vipe/<sha256><suffix>`. `GET /vipe/uploads/{sha256}?suffix=.mp4` reports an existing plate so clients skip the upload. Older UUID-named uploads remain valid but are not deduplicated.
- **Review jobs.** `ViPEReviewJob(parent_job_id, request_id)` copies the parent's bundle and runs only the witness stage against the frozen camera. Witnesses are now off by default for new solves.
- **Fast dense mode (default).** `dense_mode=fast` computes depth only for the 12 sampled views: PriorDA completes the projected native SLAM map, or UniDepth is used when map coverage is poor (the same rule as the stock processor). `dense_mode=video` keeps the original full-video Video Depth Anything path.
- **Lens distortion (experimental).** `lens_distortion=true` solves the unified (MEI) model: pinhole GeoCalib initialises the focal length, `xi` starts at 0 and bundle adjustment refines both with the generic (non-fused) solver. The bundle camera is the pinhole camera of the undistorted plate; `camera.lens_distortion` holds the model. Not yet combined with dense reconstruction.
- **Image-sequence plates.** A `.zip` upload containing `frames/000000.jpg…` and `plate.json` (built by the Blender add-on from an EXR/DPX/TIFF sequence) is rebuilt into the worker movie. `plate_width/plate_height` make the camera report in original plate pixels.
- **Square pixels.** Intrinsics keep the horizontal field of view with `fx = fy`, so rounding the processing size to even pixels no longer creates a pixel aspect in Blender.
- **Status detail.** `GET /vipe/jobs/{id}` adds `detail` (latest progress bar, CoTracker window or dense view) from an 8 KiB log tail.

## What changed in 0.4.1

- **Camera solve reuse.** Every camera solve keeps its reusable state in the job directory (`plate.mp4`, `native_points.npz`, `native_slam_map.pt`, `depth_context.npz`, bit-packed `solve_masks.zip`, `camera_manifest.json`) and is registered in `<work_directory>/solve_index.json`, keyed by the plate checksum, every solve-affecting setting and `SOLVE_RECIPE`. A later job with the same key hard-links that state, resamples the cloud for its own point limit and runs only its optional stages. Bump `SOLVE_RECIPE` in `worker.py` whenever the camera stage changes. Deleting a job directory simply makes its solve unavailable for reuse.
- **Job extensions.** `ViPEReviewJob` accepts optional `review_witnesses` (default true), `dense_environment` and `dense_mode`, so review witnesses and dense geometry can each be added to a finished job. Dense extensions need the solve state above (jobs solved with 0.4.1 or later).
- Solve masks are always captured (bit-packed, fast compression) so dense can be added later; the dense stage still reads the older format.
