# Result bundle contract

`result.zip` is checksummed data. The importer never executes or generally extracts archive members. NumPy pickle loading is disabled. The outer archive and expanded NPZ arrays have bounded size; unexpected/duplicate members, corrupt hashes, invalid transforms and geometry indices are rejected.

- **Version 1:** `result.json`, `camera.npz`, `cloud.npz`.
- **Version 2:** adds `audit_landmarks.npz` for independent witnesses.
- **Version 3:** adds `environment/view_NNNNNN.npz` and `environment/plate_NNNNNN.png` for up to 12 declared views. Witness data is optional.

All binary files are listed in the manifest with SHA256 and byte length. The manifest contains source SHA256, original dimensions, rational FPS, frame selection, settings, processing resolution, original-pixel lens stages and optional-stage receipts.

## Coordinates

Camera arrays are OpenCV camera-to-world matrices, `[fx,fy,cx,cy]` in original source pixels, and zero-based original source-frame indices. Source selection UI is one-based/inclusive. Blender import converts camera axes and applies a common world-alignment parent; no geometry is independently rescaled.

Native `cloud.npz` contains world-space `xyz`, RGB in 0–1 and original point IDs. It is the post-BA native map sampled by ID, not the post-SLAM dense environment.

Audit arrays contain triangulated landmark positions, observation frame indices/roles, 2D source observations and conditioning metadata. They do not replace camera arrays.

Each dense view declares its selected-frame index, original-source frame index, processing dimensions, intrinsics, final camera transform, stride and array counts. NPZ data contains world-space points, RGB colors, positive depth, pixel positions, UVs and quad indices. Texture UVs use pixel centers; projection checks use the corresponding integer pixel-coordinate convention. Textures are packed into Blender on import. Dense geometry uses the same world frame as the native map; the camera hash is checked before/after the depth stage.

## Timing

`result.json.timings_seconds` records subprocess stage walls and time before packaging. The worker's local `TIMINGS.json` additionally records total worker wall time including packaging. These exclude queue wait, upload/download and Blender import. Witness receipts distinguish fresh evidence from cached observations.
