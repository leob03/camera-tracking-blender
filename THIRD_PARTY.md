# Provenance and licenses

This repository contains original integration code, the original artist-authored cross asset, and camera-evidence helpers retained from the earlier camera-tracking study. `comfyui_vipe_remote/camera_evidence/SOURCE_RECEIPT.json` records the source helper hashes. `vipe_remote/artist_witnesses.py` preserves the study placement method.

External components are not bundled: NVIDIA ViPE and its dependencies, ComfyUI and the production fork, CoTracker, GeoCalib, TrackAnything/SAM/DeAOT/GroundingDINO, UniDepth, DeepLab, Video Depth Anything and PriorDA. Their code and model licenses/notices must accompany the approved worker image. Consult those exact source/model revisions before distributing a deployment.

The artist cross is `vipe_remote/assets/axis_shaped_cross.glb`, SHA256 `5dcb91287de03aa2e203327e9f3087ea315cba16c525c9bfe616b8fb9bd5ac01`. It is included at the project owner's request. This public handoff does not assign a new license to that asset or to the project's original code; the project owner has not selected an overall license yet. Existing notices in vendored files remain in place.

## Optional Gaussian trainer

[LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) runs as a separately installed executable. Its source carries GPL-3.0-or-later notices. No trainer code, binaries or weights are included in these packages. See [Gaussian worker setup](docs/GAUSSIAN_SPLATS.md) for the tested revision and CLI requirements.

## Viewport splat renderer

`vipe_remote/splat_viewer.py` follows the projection and spherical-harmonic evaluation of the reference 3D Gaussian
Splatting renderer (Kerbl et al., 2023) as implemented in KIRI Engine's
[3DGS Render for Blender](https://github.com/Kiri-Innovation/3dgs-render-blender-addon) (Apache License 2.0), whose
shaders were the reference for ours. No KIRI files are bundled; the data path (PLY loading, texture layout, sorting)
is original. KIRI 3DGS Render itself is optional and installed separately.

