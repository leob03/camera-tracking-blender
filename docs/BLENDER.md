# Blender artist guide

## Install

1. Build the packages, or obtain the built add-on ZIP from the project maintainer.
2. In Blender 4.2, open Preferences → Add-ons → Install from Disk. Select `vipe_remote-0.4.0.zip` and enable **ViPE Remote Tracking**.
3. In the 3D View sidebar (N), open **ViPE Remote → Worker connection**.
4. Enter the worker URL and click **Check Connection**. `127.0.0.1` refers to the computer running Blender. Use your worker address, gateway or SSH tunnel for a different computer.

No pip or CUDA installation is required in Blender. The optional Bearer token is session-only and intended for an existing gateway.

## Submit

Choose a local video, select a frame range, and click **Submit Tracking Job**. Keep **Import when ready** enabled for automatic import. Upload and polling run in a background thread; Blender scene edits run on the main thread.

- First source frame is 1. Last frame 0 means the end; a nonzero last frame is inclusive.
- Processing long edge defaults to 1280 pixels. This changes inference resolution; the imported camera matches the original movie dimensions.
- **Estimate and refine:** GeoCalib initializes the lens and ViPE refines it.
- **Known focal length:** `focal_mm / sensor_mm × image_width` gives horizontal focal length in pixels. This mode locks the lens with a centered principal point.
- **Exclude dynamic categories:** TrackAnything exclusions for the camera solve. Keep enabled unless deliberately testing unmasked input.
- **Independent review witnesses:** on by default. Adds CoTracker/DeepLab observations and the original four supported review crosses. Disable for faster camera-only jobs. Disabling it does not remove TrackAnything solve masking.
- **Dense environment reconstruction:** off by default. Adds video-depth inference and static surface accumulation. This is separate from the native cloud.
- **Cloud point limit:** limits native points exported/displayed, not SLAM computation or dense environment samples.
- **Blender start frame:** timeline frame corresponding to the first selected source frame, normally 1001.
- **Scene scale:** scales the entire imported result together. ViPE scale is not surveyed.

Use constant-frame-rate square-pixel movies with orientation baked into the pixels. Check initial/final lens values in the panel and full manifest text block.

## Review and scene organization

New imports use Solid shading, material colors, the camera view and the original movie background. Each job has an alignment parent and independently controlled collections:

1. Camera and source plate.
2. Static tracks: independent triangulated landmarks when witnesses are enabled; native-map markers otherwise.
3. Review crosses: the artist-authored asset at supported very-near foreground, foreground, midground and background positions. Missing support is reported rather than invented.
4. Native point cloud: hidden by default. Reveal it in the Outliner; Material Preview shows its colors.
5. Dense environment, if requested: hidden initially for camera playback. Click **Show Dense Environment** to reveal colored depth samples, switch to Material Preview and frame the geometry. Textured per-view surfaces are also included, individually hidden, with packed textures.

The dense view does not change the camera. Return to camera view with Numpad 0; hide the environment and return to Solid for lightweight tracking review. Geometry and witnesses are excluded from final renders by default; enable their render visibility deliberately if needed.

The movie remains an external local file. Dense texture images and cross meshes are stored in the saved Blender scene. Save a `.blend` to preserve the imported result.

**Resume Job** reconnects without submitting again. **Read Job Log** loads a bounded log excerpt into a Blender text block. **Saved Bundle** imports a downloaded ZIP: select the matching original movie first, or clear Video plate for geometry-only import. Reimporting the same job preserves existing artist edits.

## Interpreting witnesses

CoTracker tracks image features independently of ViPE, but its 3D landmarks are triangulated using the solved camera. Conditioning and placement filter the available landmarks, so good-looking crosses are useful spot checks, not proof that the whole trajectory is correct. Inspect varied depths and the full shot. Cross orientation is a display choice; the anchored origin is the tracked point.
