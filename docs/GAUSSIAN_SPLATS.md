# Gaussian environments from ViPE

Blender add-on **0.6.0** and ComfyUI node pack **0.5.0** add optional Gaussian training to a completed dense environment. Camera solving, witnesses and dense reconstruction work without the Gaussian trainer installed.

## Artist workflow

1. Install `dist/vipe_remote-0.6.0.zip` and restart Blender. The worker also needs the updated node pack and configured trainer below.
2. Solve a shot with **Exclude dynamic categories** enabled. Enable **Dense environment reconstruction**, or use **Results → Add Dense Environment** after solving.
3. Inspect the camera and dense geometry. In the **Gaussian Splat** sub-panel (below Results), choose **Quick preview** or **Quality** and click **Train Gaussian Environment**.
4. Follow **Open Live Log**, or the terminal used to launch Blender. Cancel and resume monitoring use the existing job controls.
5. When complete, click **Show Gaussian Splat** to load it under the job's **ViPE alignment**, or open **Camera preview**, **Nearby view**, **Masks** and **Output folder**. The downloaded Gaussian asset is `splat/scene.ply` beside this extension job's `result.zip`.

This stage does not repeat tracking or modify the camera, markers, scene alignment or dense geometry. It trains static appearance; the moving subjects are excluded from the image loss. It is never enabled automatically.

| Preset | Steps | Training long edge | Frames | Gaussian limit | Time (16 GB Quadro RTX 5000) |
| --- | --- | --- | --- | --- | --- |
| Quick preview | 10,000 | 1,536 px | every solved frame | 1,000,000 | about 4–5 min |
| Quality | 30,000 | 2,048 px | every solved frame | 3,000,000 | about 30–35 min for 300 frames |

- Shorter runs scale LichtFeld's whole 30k-step schedule (densification, SH growth, learning-rate decay) with `--steps-scaler`; plain `--iter` would stop the schedule halfway and leave densification unfinished.
- Every selected frame trains; none are held out. Shots longer than 400 frames are subsampled evenly. More views reduce floaters and smears; they do not slow each step.
- Image-sequence (EXR) plates are solved from a 1280 px proxy. **Quality** uploads a second, 2048 px proxy of exactly the solved frames (content-addressed and cached like the solve proxy; about 1 MB per frame). Movie plates are already on the worker at full size, so nothing extra is uploaded. Training never upscales the decoded frames.
- **Custom settings** exposes steps, long edge, frame step and Gaussian limit. About 250 MB of PLY per million Gaussians.

## Viewing the splat in Blender

**Show Gaussian Splat** imports `splat/scene.ply` into a **06 Gaussian splat** collection under the job, parented to the **ViPE alignment** object, so it follows Set Floor / Origin / Scale like the rest of the job. It uses, in order:

1. **Blender 5.3+ native splats** (point clouds of type 3D Gaussian Splats; Workbench, EEVEE and Cycles). Not yet tested here: 5.3 is in beta.
2. **KIRI 3DGS Render**, if installed: 4.1.x on Blender 4.4, 5.1.0 on Blender 5.1/5.2 (5.x needs Blender 5.1+). Tested on Blender 4.4 + KIRI 4.1.3 and Blender 5.2.3 + KIRI 5.1.0.
3. Otherwise the button reports what to install; the rendered previews still work.

The add-on compares the imported positions with the PLY and undoes any axis conversion an importer applied, so the splat sits on the dense samples whichever importer ran.

KIRI notes from testing (Ford 2.5M Gaussians, Linux):
- KIRI imports in its point-editing display (bare vertices, black in Solid view). **Show Gaussian Splat** switches the splat to KIRI's camera-facing display (*Enable Camera Updates*) and the viewport to Material Preview. EEVEE compiles KIRI's shaders the first time (30–60 s); the plate shows until then. On big splats, *Show As Point Cloud* in the 3DGS Render tab is lighter.
- Import takes about a minute; the viewport and EEVEE renders are approximate (EEVEE quads; grainy with dithered alpha, smoother but streaky with KIRI's blended HQ mode) and clearly below LichtFeld's own renders.
- KIRI's own GPU renderer (Render panel) failed on Linux (its shader directory is only set by other code paths) and, once its shaders were loaded, used more than 14 GB of RAM before it was stopped. Treat it as untested for production-size splats.
- KIRI's proxy object stores the whole Gaussian array in the `.blend` (about 240 MB per million Gaussians).

## Worker setup

Keep the current ViPE worker environment. Install the new node pack and restart ComfyUI when its queue is empty. Add this optional field to the worker JSON:

```json
{
  "lichtfeld_executable": "/opt/lichtfeld/launch.sh"
}
```

The configured file must be executable and accept LichtFeld CLI arguments. A wrapper can set library paths and then `exec` the trainer. An empty value leaves ordinary tracking available and reports Gaussian training as unavailable.

Tested trainer: [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio), source revision `c72a0d8554a33371ccd97df1e9fbe480d0981fdd` (0.5.3 development snapshot), an existing Linux CUDA 12.8 build on a Quadro RTX 5000, 16 GiB. The initial prototype targets a Linux worker; Windows Blender remains a network client. Other GPU architectures need a compatible trainer build.

Required CLI features include `--headless`, `--strategy mrnf`, `--mask-mode ignore`, `--centralize off`, `--min-track-length 0`, `--no-download`, PLY export, and `--render-camera-path/--render-load/--render-output`. Check the installed build’s `--help`; availability in another release is not assumed.

No Python libraries or models were added to ComfyUI for this stage. Dataset preparation runs in the existing ViPE Python environment using NumPy, SciPy and OpenCV. Training and preview rendering run in the separately installed native LichtFeld executable. Prepare that executable, CUDA/runtime libraries and its required assets in the online container build. This repository does not redistribute the trainer binary. LichtFeld carries GPL-3.0-or-later notices; retain its license and source/build provenance when packaging it.

Every trainer invocation uses `--no-download`; training also uses `--safe-mode`. LPIPS is optional and was unavailable in the offline test, so it was skipped. No additional inference checkpoint is required by this tested training path. The existing camera/dense models remain required for producing the parent job.

Keep the parent source movie, dense result and `solve_masks.zip` available. The new job follows explicit parent job IDs to find retained masks. It rejects missing masks, masking-disabled solves, distorted cameras and changed source bytes. Currently only pinhole solves are supported; undistortion is a future addition.

Use work **and** output filesystems with room for datasets, checkpoints, PLY and ZIP copies. The stage requires at least 5 GiB free scratch at startup and retains the existing worker disk/RAM guards. One million Gaussians produced a roughly 229 MB compressed review bundle in the test. The result budget is 500 MiB and the importer limit is 512 MiB. Cancellation stops the worker process group, including LichtFeld. Intermediate datasets and training outputs are retained for diagnosis; delete only an explicitly reviewed job directory after keeping its result.

### ComfyUI directly

Load `workflows/ViPE Train Gaussian Environment.json`. Paste the ID of a completed job whose result includes dense geometry. Leave `request_id` empty for a fresh ID on every Run. A dense extension has its own ID; the original camera job may not contain the later dense result.

For DCC bridge integration, `GET /vipe/capabilities` advertises `splat_jobs: "ViPESplatJob"` when the optional trainer is configured. Submit this single-node API prompt:

```json
{
  "1": {
    "class_type": "ViPESplatJob",
    "inputs": {
      "parent_job_id": "<completed-dense-job-uuid>",
      "request_id": "<new-job-uuid>",
      "splat_preset": "quality"
    }
  }
}
```

Omitted fields take the preset's values; `splat_iterations`, `splat_max_dimension`, `splat_frame_step` and `splat_max_gaussians` override them, and `splat_training_video` / `splat_training_sha256` name an optional uploaded training plate (a movie or a proxy-frame ZIP of exactly the solved frames). Use the same prompt ID, log, cancellation, history and download handling as tracking. Bundle version 4 adds five declared, checksummed files under `splat/`. Older add-ons must be upgraded to read it. The native trainer is never run on the artist machine.

## What the stage does

1. Extract source images for the solved frame selection and scale the final per-frame intrinsics to their dimensions.
2. Convert the final ViPE camera-to-world poses to COLMAP-format camera records. **COLMAP is only an interchange format here; no COLMAP camera solve runs.** Training keeps the supplied cameras fixed.
3. Create image-loss masks from the retained TrackAnything exclusions. Dynamic exclusions are slightly dilated; sky remains eligible for appearance learning. Excluded pixels are ignored in the loss, not painted into the training image.
4. Project dense samples into the other retained depth views. Keep samples agreeing with at least two views within 6% depth, reject sufficiently contradictory free-space samples, deduplicate spatially and cap them at 400,000, then add the post-BA native map. Occluded points behind an observed surface are not treated as free-space errors.
5. Train LichtFeld’s MRNF strategy with those points as initialization, image masks and fixed cameras. **Depth initializes and filters the geometry; this prototype does not apply a depth loss during training.**
6. Export the Gaussian PLY and render two previews: the solved trajectory and the same trajectory displaced sideways by 2.5% of the median dense depth. Verify that `camera.npz` remains byte-identical, then package the outputs.

Training preserves separate `fx/fy`, principal points and animated intrinsics. LichtFeld’s cinematic preview interface uses a centered camera and vertical field of view, with an integer output frame rate; previews are therefore approximate when a solve has off-center or strongly unequal intrinsics. The exact training camera records are retained in the worker dataset.

### Coordinates

Dataset cameras and points use the ViPE world (OpenCV axes). LichtFeld flips world Y and Z internally, but the exported PLY is back in the ViPE world: parent it under the job's **ViPE alignment** object with an identity transform, exactly like the dense samples. (Verified on the Ford and skateboard_alley splats against their seed points; an extra 180° X rotation puts the splat upside down.) Only LichtFeld's render camera paths use the flipped world and OpenGL camera axes: position = F·C, rotation = F·R·F with F = diag(1, −1, −1).

## Preset A/B on skateboard_alley — 2026-10-27

Same ViPE solve and 12-view dense result; all runs score the same 14 held-out source frames (0, 24, …, 312), masked to static pixels and compared at 1536 px. Holding frames out is for this comparison only; the presets train on every frame.

| Run | Views | Training size | Steps | Gaussians | Masked PSNR | SSIM | Trainer time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| First prototype (every 3rd frame, 15k steps of a 30k schedule) | 92 | 1536 × 810 | 15,000 | 1.0M (limit) | 32.71 dB | 0.957 | 274 s |
| **Quick preview** | 304 | 1536 × 810 | 10,000 (scaled schedule) | 1.0M | 34.79 dB | 0.963 | 250 s |
| **Quality** | 304 | 2048 × 1080 | 30,000 | 2.08M | 36.44 dB | 0.969 | 1,934 s |

Quality at its 15k checkpoint scored 33.7 dB (LichtFeld, 2048 px): the second half of the schedule is where most of the gain comes from, which is why Quick scales the schedule instead of stopping early. Quality's PLY is 515 MB.

## First skateboard_alley result — 2026-10-26

- Reused the existing 318-frame, 24 fps solve and 12-view dense reconstruction; no new camera or dense inference was needed.
- 106 training-dataset images at 1536 × 810; 14 held out of photometric fitting and 92 used for training.
- 493,722 dense samples → 446,044 passing depth consistency → 150,000 initialization points.
- 15,000 iterations, one million final Gaussians, approximately **264 seconds of trainer runtime** on the local Quadro RTX 5000. This excludes the earlier camera/dense solve, preparation, rendering, transfer and Blender import.
- The corridor remains recognizable along the moving camera and a small sideways offset. Skateboarders are largely removed. Floor smears, residual board/shadow artifacts and late left-wall distortions remain. The saved masks exclude part of that wall late in the shot.
- Preserved this first result for visual review. No parameter search or score-driven tuning was performed.

A separate 100-iteration real ComfyUI job verified submission, training, both previews, download and version-4 import. A cancellation test terminated the worker, stage and trainer processes. The actual Blender training operator also passed a second API test, including live logs, download and attachment. Blender 4.4 imported and reattached the full-quality result without duplicating geometry or changing the camera. Synthetic tests cover 180° camera rotations, anisotropic image scaling, packed masks, depth consistency and offline CLI settings.

The held-out images are held out only from appearance fitting: ViPE estimated cameras and depths using the full shot. These are not fully unseen reconstruction views. There is no matched COLMAP baseline yet, so this experiment does not establish that ViPE is superior to COLMAP. Nerfstudio and gsplat can also consume external cameras; we selected LichtFeld because its existing native build supplied the needed training, masks, PLY export and preview rendering.

## Next improvements to review together

- Repair the late-shot wall exclusion and include moving skateboards and shadows where appropriate. Compare saved masks and previews before increasing iterations.
- Add confidence-weighted depth supervision and stronger consistency filtering if floaters persist; avoid forcing uncertain depth onto the trainer.
- Add lens undistortion, more careful exposure handling and representative keyframe selection.
- Run a controlled comparison using the same images, masks, renderer and budget with ViPE versus alternative camera/point initialization.
- Add an explicit Blender Gaussian renderer adapter after choosing the renderer artists will use.
