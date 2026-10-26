# Gaussian environments from ViPE

Blender add-on **0.6.0** and ComfyUI node pack **0.5.0** add optional Gaussian training to a completed dense environment. Camera solving, witnesses and dense reconstruction work without the Gaussian trainer installed.

## Artist workflow

1. Install `dist/vipe_remote-0.6.0.zip` and restart Blender. The worker also needs the updated node pack and configured trainer below.
2. Solve a shot with **Exclude dynamic categories** enabled. Enable **Dense environment reconstruction**, or use **Results → Add Dense Environment** after solving.
3. Inspect the camera and dense geometry. In **Results → Gaussian environment**, choose the training settings and click **Train Gaussian Environment**.
4. Follow **Open Live Log**, or the terminal used to launch Blender. Cancel and resume monitoring use the existing job controls.
5. When complete, use **Camera preview**, **Nearby view**, **Masks**, and **Output folder**. The downloaded Gaussian asset is `splat/scene.ply` beside this extension job’s `result.zip`.

This stage does not repeat tracking or modify the camera, markers, scene alignment or dense geometry. It trains static appearance; the moving subjects are excluded from the image loss. It is never enabled automatically.

| Setting | Default | Effect |
| --- | --- | --- |
| Training iterations | 15,000 | More optimization steps; additional steps cannot repair a bad camera or missing observations |
| Training long edge | 1,536 px | Maximum image dimension; higher values use more memory |
| Use every Nth frame | 3 | Samples the solved selection; at least 12 sampled frames are required |
| Gaussian limit | 1,000,000 | Upper limit on the trained Gaussian count; affects memory, transfer size and detail |

The PLY contains Gaussian parameters, including opacity, covariance and spherical harmonics. It is not the native ViPE point cloud. This add-on stores its path and opens rendered previews; **it does not implement live Gaussian rendering in Blender**. Importing the PLY as ordinary mesh vertices will not reproduce the trained appearance. A compatible Gaussian renderer is a separate integration.

The files are stored with the downloaded job, not embedded in the `.blend`. Keep that folder when moving the scene. Existing scenes saved before 0.6.0 may need their dense result reimported to record its exact worker job ID, especially if witnesses were added afterwards.

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
      "splat_iterations": 15000,
      "splat_max_dimension": 1536,
      "splat_frame_step": 3,
      "splat_max_gaussians": 1000000
    }
  }
}
```

Use the same prompt ID, log, cancellation, history and download handling as tracking. Bundle version 4 adds five declared, checksummed files under `splat/`. Older add-ons must be upgraded to read it. The native trainer is never run on the artist machine.

## What the stage does

1. Extract source images for the solved frame selection and scale the final per-frame intrinsics to their dimensions.
2. Convert the final ViPE camera-to-world poses to COLMAP-format camera records. **COLMAP is only an interchange format here; no COLMAP camera solve runs.** Training keeps the supplied cameras fixed.
3. Create image-loss masks from the retained TrackAnything exclusions. Dynamic exclusions are slightly dilated; sky remains eligible for appearance learning. Excluded pixels are ignored in the loss, not painted into the training image.
4. Project dense samples into the other retained depth views. Keep samples agreeing with at least two views within 6% depth, reject sufficiently contradictory free-space samples, deduplicate spatially and cap the seeds at 150,000. Occluded points behind an observed surface are not treated as free-space errors.
5. Train LichtFeld’s MRNF strategy with those points as initialization, image masks and fixed cameras. **Depth initializes and filters the geometry; this prototype does not apply a depth loss during training.**
6. Export the Gaussian PLY and render two previews: the solved trajectory and the same trajectory displaced sideways by 2.5% of the median dense depth. Verify that `camera.npz` remains byte-identical, then package the outputs.

Training preserves separate `fx/fy`, principal points and animated intrinsics. LichtFeld’s cinematic preview interface uses a centered camera and vertical field of view, with an integer output frame rate; previews are therefore approximate when a solve has off-center or strongly unequal intrinsics. The exact training camera records are retained in the worker dataset.

### Coordinates

Dataset cameras and points use the ViPE world. The tested LichtFeld COLMAP importer/exporter flips world Y and Z. For a Gaussian renderer under the Blender scene’s existing **ViPE alignment** object, apply a 180° X rotation to the imported Gaussian asset; leave the alignment object’s existing transform and scale in effect. Apply the transform to the Gaussian object, including covariance orientation, not just its point centers. The bundle records this convention. Preview paths account for both this world transform and OpenGL camera axes.

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
