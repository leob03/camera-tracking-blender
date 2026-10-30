# Blender artist guide

## Install

1. Build the packages, or obtain the built add-on ZIP from the project maintainer.
2. In Blender 5.2 LTS or 4.4, open Preferences → Add-ons → Install from Disk. Select `vipe_remote-0.6.0.zip` and enable **ViPE Remote Tracking**.
3. In the 3D View sidebar (N), open **ViPE Remote → Worker**.
4. Enter the worker URL and click **Check Connection**. The main panel shows *Worker connected*, *Worker check failed* or *Worker not checked* for the current URL; the refresh button beside it checks again. `127.0.0.1` refers to the computer running Blender. Use your worker address, gateway or SSH tunnel for a different computer.

No pip or CUDA installation is required in Blender. The optional Bearer token is session-only and intended for an existing gateway.

## Panel layout

The **ViPE Remote Tracking** panel holds the connection indicator, **Submit Tracking Job** / **Cancel Job** (**Stop Download** while downloading), the status, a progress bar and the elapsed time. While a job runs, a checklist marks each stage done, current or pending: Upload, Queue, Prepare, Camera solve, Review witnesses and Dense (when enabled), Package, Download, Import. Live worker detail (for example a SLAM pass percentage) appears under the current stage.

Sub-panels, which can be collapsed or reordered, all start closed: **Worker**, **Plate**, **Lens**, **Solve**, **Outputs**, **Results** and **Gaussian Splat** (only when the selected job is in the scene), **Jobs** and **Logs**. Settings are locked while a request runs.

## Submit

Choose a local video, select a frame range, and click **Submit Tracking Job**. Keep **Import when ready** enabled for automatic import. Upload and polling run in a background thread; Blender scene edits run on the main thread. A plate the worker already has is not uploaded again.

- **Plate:** choosing a video reads its resolution, frame rate, frame count and duration once. First source frame is 1. Last frame 0 means the end; a nonzero last frame is inclusive. The line below shows the selected frames and duration; **Full Clip** resets the range to 1–0. **Blender start frame** is the timeline frame of the first selected source frame, normally 1001.
- **Lens → Estimate and refine:** GeoCalib initializes the lens and ViPE refines it. **Estimate lens distortion (experimental)** adds a one-parameter unified (MEI) distortion and undistort/redistort data; it is available only in this mode and not together with dense reconstruction.
- **Lens → Known focal length:** `focal_mm / sensor_mm × image_width` gives horizontal focal length in pixels. This mode locks the lens with a centered principal point.
- **Filmback:** choose a camera preset (ARRI ALEXA 35/LF/Mini, RED V-RAPTOR 8K VV, full frame, Super 35, Micro Four Thirds) or type **Sensor width (mm)**. The width sets the imported camera's sensor and the known-focal conversion; it does not change the solve.
- **Solve:** processing long edge defaults to 1280 pixels. This changes inference resolution; the imported camera matches the original movie dimensions. **Exclude dynamic categories** applies TrackAnything exclusions to the camera solve. Keep it enabled unless deliberately testing unmasked input.
- **Outputs → Review witnesses:** off by default. Adds CoTracker/DeepLab observations and the original four supported review crosses, at about 2.5–3× total time. It can also be added later from **Results**. It does not affect TrackAnything solve masking.
- **Outputs → Dense environment reconstruction:** off by default. **Dense depth** chooses *Sampled views (fast)* (about 1–2 minutes extra) or *Full video depth (slow, original)*. This is separate from the native cloud.
- **Cloud point limit:** limits native points exported/displayed, not SLAM computation or dense environment samples. Native points are reconstruction hypotheses, not surveyed points.
- **Scene scale:** scales the entire imported result together. ViPE scale is not surveyed.

Resubmitting a plate never re-uploads it, and the worker reuses an earlier camera solve whenever the plate and the solve settings (frame range, processing size, masking, lens) are unchanged; only newly requested outputs run. Image-sequence proxies are cached in the download folder, so resubmitting the same frames does not read the plates again. After an import, Timeline, Dope Sheet and Graph Editors are framed to the imported frames.

## Results

For the selected job: frames, focal length in mm at the filmback and in pixels, lens refinement, distortion if solved, native points, review error (median/p90) when witnesses exist, and stage timings. With witnesses, the camera carries an animated `vipe_reprojection_px` property: open the Graph Editor and select the camera. The horizontal axis is the timeline frame; the vertical value is, for that frame, the median distance in plate pixels between where independent CoTracker tracks see a static point and where the solved camera projects it (held-out observations only). Lower is better; isolated spikes point at frames worth checking. Values scale with plate resolution, so compare frames within a shot rather than across shots.

- **Run Review Witnesses** and **Add Dense Environment** run only that stage on the worker, reusing the job's solved camera, and merge the result into the existing collection. They need the original worker job record (not a saved bundle). Dense needs a job solved with node pack 0.4.1 or later.
- **Show / Hide Dense Environment** toggles the dense geometry. Show switches to Material Preview and frames it; Hide returns to Solid shading and the camera view.
- **Export to Nuke…** writes a Nuke script and its data to a chosen folder and reports any warnings.
- **Scene alignment** moves only the job's alignment parent, so the camera and points stay consistent. Select markers in Object Mode, or vertices (for example native cloud points) in Edit Mode. **Set Floor** fits a plane through 3+ points and makes it Z = 0 with its centroid at the origin and the camera above. **Set Origin** moves the centroid of the selection to the origin. **Set Scale** scales about the origin so two points are the entered distance apart. **Set X Axis** rotates about Z so the direction from the other point to the active point is +X. **Reset Alignment** restores the imported transform. All are undoable.

## Gaussian splats

After a dense environment, the **Gaussian Splat** sub-panel trains a static splat on the worker (**Quick preview** or **Quality**). **Show / Hide Gaussian Splat** draws it in the viewport on the solve, hiding the dense environment and point cloud meanwhile; **Camera preview** and **Nearby view** play the trainer's renders; **Open in KIRI 3DGS Render** hands it to KIRI for editing and relighting. See [Gaussian splats](GAUSSIAN_SPLATS.md).

Use constant-frame-rate square-pixel movies with orientation baked into the pixels. Check initial/final lens values in the panel and full manifest text block.

## Review and scene organization

New imports use Solid shading, material colors, the camera view and the original movie background. Each job has an alignment parent and independently controlled collections:

1. Camera and source plate.
2. Static tracks: independent triangulated landmarks when witnesses are enabled; native-map markers otherwise.
3. Review crosses: the artist-authored asset at supported very-near foreground, foreground, midground and background positions. Missing support is reported rather than invented.
4. Native point cloud: hidden by default. Reveal it in the Outliner; Material Preview shows its colors.
5. Dense environment, if requested: hidden initially for camera playback. Click **Show Dense Environment** to reveal colored depth samples, switch to Material Preview and frame the geometry. Textured per-view surfaces are also included, individually hidden, with packed textures.

The dense view does not change the camera. Return to camera view with Numpad 0; hide the environment and return to Solid for lightweight tracking review.

### Seeing the plate with the markers

Camera background images are viewport references only: Rendered shading draws the world over them and renders never include them, which used to give a black render. Imports therefore set up the scene for review:

- **Film → Transparent** is enabled, so the plate stays visible behind geometry in Rendered and Material Preview views.
- The **compositor** puts the plate (undistorted when a lens model was solved) under the render, so F12 and Render Animation show the plate with the markers. It is only set up when the scene has no compositor setup of its own; an existing setup is left untouched.
- **Static tracks** get bright yellow crosses that render (the black empties remain for selection and the alignment tools). **Review crosses** render too. **Hide / Show Markers in Renders** in Results removes them from renders, for example before rendering CG for comp; the plate stays composited.
- The scene view transform is set to **Standard**, so the plate looks the same in the viewport and in renders.

The native cloud and dense geometry stay excluded from renders.

The movie remains an external local file. Dense texture images and cross meshes are stored in the saved Blender scene. Save a `.blend` to preserve the imported result.

**Jobs:** **Resume Job** reconnects without submitting again. **Read Job Log** loads a bounded log excerpt into a Blender text block. **Saved Bundle** imports a downloaded ZIP: select the matching original movie first, or clear Video plate for geometry-only import. Reimporting the same job preserves existing artist edits. **Recent jobs** lists the 40 newest `job.json` records in the download folder after **Refresh** (it never scans on redraw); **Load** selects one like Resume would, **Import** imports its downloaded result.

## Interpreting witnesses

CoTracker tracks image features independently of ViPE, but its 3D landmarks are triangulated using the solved camera. Conditioning and placement filter the available landmarks, so good-looking crosses are useful spot checks, not proof that the whole trajectory is correct. Inspect varied depths and the full shot. Cross orientation is a display choice; the anchored origin is the tracked point.

## Live job logs (add-on 0.4.1)

New submissions and **Resume Job** automatically follow worker output about once per second. **Echo logs to terminal** is enabled by default: launch Blender from a terminal to see the worker's stdout/stderr alongside Blender messages. On Windows, Blender's Window → Toggle System Console provides a console view.

In **Logs**, click **Open Live Log** while a job is running to open its log in a Text Editor beside the viewport. Clicking again reuses that editor. Disable **Follow live output** to pause editor updates and scroll through a message; the job and terminal output continue. Enable it again to catch up, including after the job finishes. Each job has its own `ViPE Log <job-id>` text block.

The editor retains the latest 262,144 characters. Resume starts with recent worker output, and bounded catch-up may omit older bytes; the display labels any omission. The complete underlying log stays at `<work_directory>/<job-id>/worker.log` on the worker. **Read Job Log** still fetches a recent snapshot for completed jobs.

Log polling opens only the selected job file, reads at most 32 KiB per request and closes it immediately. There are no storage scans or additional GPU processes. A log endpoint failure does not fail the tracking job. The worker writes Python output unbuffered; CoTracker reports each tracking window. Other stages can still be quiet between their own messages.

Install both updated packages and restart ComfyUI when its queue is idle. The incremental endpoint requires node pack 0.3.1; older workers retain the manual snapshot endpoint. Restart Blender to load the updated add-on.
