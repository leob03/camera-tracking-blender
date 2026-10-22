# Nuke export

`vipe_remote.nuke_export.export_nuke(context, collection, directory)` writes a Nuke package for one imported job collection and returns `{'nk': path, 'files': [paths], 'warnings': [str]}`. User errors (no job collection, no folder, missing camera or manifest text) raise `ValueError`. A failed Alembic export becomes a warning. The exporter runs on Blender's main thread: it samples every frame with `scene.frame_set`, then restores the frame, selection, active object and visibility. Expect about 5 s extra for a 4K lens model (STMaps).

## Inputs on the job collection

`vipe_camera`, `vipe_start_frame`, `vipe_job_id`, `vipe_manifest_text` (source `name/width/height/fps_num/fps_den`, selection `count/first_source_frame`) and these optional properties:

- `vipe_plate_path`: a movie, or an image sequence with `####` or `%04d`.
- `vipe_plate_first_frame`: for sequences, the file number of source frame 0. If it is missing, the exporter assumes the file numbers match the timeline and warns.
- `vipe_distortion`: MEI JSON in original-plate OpenCV pixels (pixel centers on integers, v down). `width`/`height` default to the manifest size.

The camera is sampled from its evaluated world matrix, so edits to the alignment parent are baked in. The native cloud is the mesh with `vipe_role == 'native_cloud'`, read at the start frame.

## Files

| File | Contents |
|---|---|
| `<label>_track.nk` | Node text without a Root node, for File > Import Script or paste. The nodes are:<br>• `ViPE_Info` StickyNote<br>• `ViPE_Plate` Read (or a black Constant)<br>• if there is a lens model: `ViPE_UndistortMap` Read + `ViPE_Undistort` STMap<br>• `ViPE_Camera` Camera2<br>• `ViPE_NativeCloud` BakedPointCloud (at most 50,000 points, linear `Color`)<br>• `ViPE_Scene` + `ViPE_Render` ScanlineRender (bg is the undistorted plate)<br>• if there is a lens model: `ViPE_RedistortMap` Read + `ViPE_Redistort` STMap |
| `<label>_camera.chan` | `frame tx ty tz rx ry rz vfov` (degrees, ZXY, vertical FOV from focal and vaperture) |
| `<label>_camera.abc` | Blender Alembic export of only the camera and raw cloud vertices. It is flattened to world space, uses the job frame range and Alembic's Y-up conversion, and the point display modifier is disabled. |
| `stmap_undistort.exr`, `stmap_redistort.exr` | W×H, FLOAT R,G,B (B = 0), ZIP compression. Written by `exr_writer.py`. |
| `README.txt` | Compositor notes |

Paths in the script are absolute. The STMap Reads use `raw true`.

## Conventions

- **World:** Blender (x, y, z) → Nuke (x, z, −y). The camera axes agree (−Z forward, +Y up), so `M_nuke = C4 · M_world`. Scale is removed before decomposing, with a warning for non-uniform or mirrored scale.
- **Rotation:** `rot_order ZXY`, R = Ry·Rx·Rz (Z is applied first). Uses `Matrix.to_euler('ZXY', previous)` for frame-to-frame continuity. Values are in degrees, with `{curve x<start> …}` keys on consecutive frames.
- **Filmback:**
  - `focal` = lens (mm). It is written as a curve only if it is animated.
  - `haperture` = the horizontal sensor size. For VERTICAL or AUTO portrait fit, it is derived from the fitted size.
  - `vaperture = haperture·H/W`. Square pixels are assumed, with a warning otherwise.
- **Window translate:** `win_translate = 2·shift·fit_size/haperture`, which is `(2·shift_x, 2·shift_y)` for horizontal fit. It is written only when non-zero. This assumes Nuke's projection is `ndc = focal/(haperture/2)·(x, y)/(−z) − win_translate`, with the frame spanning x ∈ [−1, 1] and y ∈ [−H/W, H/W]. The test confirms that this model reproduces Blender's `world_to_camera_view` to 1e-6 frame units with shift applied. If Nuke instead normalizes y by half the *vertical* aperture, use `win_translate.y = 2·shift_y·W/H`.
- **Plate timing:**
  - Movie: `first 1`, `last first_source+count`, `frame_mode "start at"`, `frame start−first_source`. Movie frame 1 is source frame 0.
  - Sequence: file frames `plate_first+first_source …`, with `frame_mode offset` when the file numbers differ from the timeline.
- **STMap:** `R = (u+0.5)/W` and `G = (H−v−0.5)/H`, where (u, v) are OpenCV pixels. Rows in the EXR are stored top-first. Invalid rays are written as −1 (black) and reported.
- **MEI lens:** projection is `u = fx·X/(Z+ξ|P|)+cx` (same for v). Unprojection is:
  - `m = ((u−cx)/fx, (v−cy)/fy)`, `q = √(1+(1−ξ²)|m|²)`, `k = (ξ+q)/(1+|m|²)`
  - ray = `(k·mx, k·my, k−ξ)`

  The undistorted pinhole uses `f_u = undistorted_focal_px`, the same cx/cy and the plate resolution.
  - Undistort map: pinhole pixel → MEI pixel.
  - Redistort map: MEI pixel → pinhole pixel.

  A warning is given if the Blender camera's focal or principal point disagrees with the lens model. There is no overscan, so the redistorted CG is black where the pinhole frame does not cover the plate corners.

## Validation (Blender 4.2.3, no Nuke)

```
<blender> -b --factory-startup --python-exit-code 1 --python tests/blender_nuke_export.py -- /tmp/out
<blender>/4.2/python/bin/python3.11 tests/test_exr_writer.py     # any Python with NumPy; adds OpenImageIO when present
```

`tests/blender_nuke_export.py` builds a synthetic importer-style job: a rotated and scaled alignment parent, 20 keyed frames, a colored cloud with a display modifier, and MEI ξ = 0.08. It then checks:

- **Camera from the `.nk`:** rebuilds every frame's Nuke camera (Ry·Rx·Rz), which matches `C4 · matrix_world` to 4e-7.
- **Projection:** the projection model above matches Blender.
- **`.chan`:** the rows match the curves.
- **Point cloud:** BakedPointCloud points and colors match.
- **EXRs:** both load at W×H in Blender and OpenImageIO 2.5 with identical floats, and match an independent MEI evaluation.
- **Lens round trip:** redistort∘undistort through bilinear map sampling is ≤ 1.2e-4 px for ξ ∈ {0, 0.05, 0.08, 0.2}.
- **Alembic:** re-imported camera and points match the scene.
- **Restore:** selection and visibility are restored.

## Unverified without Nuke

- Whether Nuke accepts the `.nk` syntax as written: no `version` line, `set cut_paste_input`, quoted knob values, and `frame_mode`/`frame`, `origset`, `raw`, `note_font_size` knobs.
- **STMap input order.** The script assumes input 0 = `src` and input 1 = `stmap`. If the result is a UV gradient or wrong, swap the inputs with Shift+X.
- **ScanlineRender input order:** assumed 0 = bg, 1 = obj/scn, 2 = cam.
- **BakedPointCloud** `serializePoints`/`serializeNormals`/`serializeColors` as `"N x y z …"`, with N = point count. The normals point to the first camera. Use the Alembic file if the node does not load. ScanlineRender may not draw points at all; review the cloud in the 3D viewer.
- The meaning of ZXY in Nuke (assumed Z applied first, R = Ry·Rx·Rz), plus the `win_translate` units and sign. Both are derived from documentation, not from a Nuke render.
- The `.chan` vfov column on import, and how Nuke's ReadGeo/Camera reads Blender's Alembic camera (aperture and film offset).
- Whether the STMaps line up in Nuke against a real plate. They are only checked numerically against the formulas above.
