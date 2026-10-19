# Changes

## Blender add-on 0.5.0 / ComfyUI node pack 0.4.0

Workflow
- Plates already on the worker are not uploaded again: the add-on hashes locally and the worker stores uploads by SHA256.
- Review witnesses are off by default and can be added later to a finished job (**Run Review Witnesses**) without solving again; the result merges into the existing scene.
- Dense environment defaults to a fast mode that computes depth only for the 12 kept views (the original full-video mode remains available).
- EXR (and other image-sequence) plates: the add-on builds a tone-mapped proxy at the processing size (OpenImageIO + OpenColorIO in Blender), uploads it as a frame archive and keeps it locally for fast background playback. The camera is reported in original plate pixels and the timeline starts at the sequence's first frame.
- Live stage detail (SLAM pass percentage, CoTracker window, dense view) in Blender's progress.
- Worker settings for larger GPUs: CPU threads, TrackAnything batch size, depth alignment device, prefetching.

Camera quality and handoff
- Imported cameras use the real filmback (sensor presets), so the focal length reads in real millimetres; the constant solved lens is no longer keyed on every frame.
- Square-pixel plates get one focal length (horizontal field of view preserved); rounding the processing size no longer creates a pixel aspect.
- Experimental lens distortion: unified (MEI) model refined in bundle adjustment. Blender gets a pinhole camera for the undistorted plate plus a fitted polynomial lens on the movie clip with an undistorted background.
- Review error: held-out reprojection residuals of the independent landmarks are summarised in the panel and animated on the camera (`vipe_reprojection_px`) for the Graph Editor.
- Nuke export: camera `.nk` (Camera, Read, point cloud, ScanlineRender), `.chan`, Alembic, and undistort/redistort STMaps when distortion is solved.
- Scene alignment tools: Set Floor, Set Origin, Set Scale, Set X Axis, Reset.

UI
- Sidebar reorganised into sub-panels with plate information, stage checklist, results summary and job history.

## Blender add-on 0.4.1 / ComfyUI node pack 0.3.1

- Running jobs stream bounded incremental worker output to Blender's launch terminal and a per-job Text Editor.
- Open Live Log works while tracking. Follow live output can pause the editor for inspection and catch up later.
- Worker Python output is unbuffered. UTF-8 boundaries, log replacement/truncation, final errors and connection cleanup are handled without changing the solve.
- Server and display buffers are bounded; logging errors do not fail tracking. The existing plain-text snapshot endpoint remains compatible.

## Blender add-on 0.4.0 / ComfyUI node pack 0.3.0

- Independent review witnesses are optional and enabled by default. Camera-only jobs skip CoTracker/DeepLab evidence entirely.
- Dense environment reconstruction is optional and disabled by default. It uses post-SLAM video depth, actual solve exclusions and frozen final cameras.
- Version-3 bundles carry validated per-view geometry and textures. Blender imports a separate environment collection and provides Show Dense Environment. Existing version-1 and version-2 bundles remain supported.
- Worker receipts separate camera, witness and dense-stage timing. Matched full-shot benchmarks are included in the handoff.
- New imports retain Solid shading, source movie background, hidden native cloud and the original artist witness asset/placement method when enabled.

Earlier local packages and saved scenes are retained on the development workstation. This is the first public repository handoff.
