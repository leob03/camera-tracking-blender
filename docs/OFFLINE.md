# Offline deployment preparation

The local prototype reused installed model caches. It did not build or certify an offline production image. `provenance/offline-dependencies.json` records selected effective versions and the 37 packages added to the local Comfy environment; it is **not a complete lock file or wheelhouse**. Those 37 packages mostly satisfy the fork's existing requirements, rather than new ViPE-node requirements.

Prepare in an Internet-connected build environment:

1. The approved ComfyUI fork, frontend static assets, this custom node pack, and pinned server dependencies.
2. The reviewed ViPE source and its own complete Python environment, including native C++/CUDA extensions built for the chosen Python, Torch/CUDA and GPU architectures. Do not copy the development Comfy environment's `.pth` link to another host's packages.
3. FFmpeg, FFprobe and runtime libraries.
4. Complete resolved model caches, including configuration/tokenizer/backbone files, licenses and checksums. Do not leave symlinks pointing outside the image.
5. Writable container-local input, output, scratch and cache paths; read-only packaged model directories where supported.

See the [exact model inventory and DVC layout](MODELS.md) and [checksum manifest](../provenance/model-manifest.json).

## Model groups

| Stage | Required assets |
| --- | --- |
| Camera | GeoCalib pinhole (auto lens), DROID, UniDepth V2 ViT-L (backbone included in the pinned full checkpoint) |
| Solve masking | SAM ViT-B, DeAOT, GroundingDINO Swin-T, text encoder/tokenizer assets |
| Optional witnesses | CoTracker3 repository and scaled-offline checkpoint; torchvision DeepLabV3 MobileNet V3 Large weights |
| Optional dense environment | Video Depth Anything small (`vits`), PriorDA and UniDepth assets used by `adaptive_unidepth-l_svda` |

The manifest records the current reviewed recipe; recheck loaders when changing model or source revisions. Witness generation requires its DeepLab checkpoint to exist locally and fails clearly if missing. Worker processes set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; these flags alone do not disable Torch Hub/gdown network fallbacks.

Run the built image with external network access blocked, while allowing internal Blender/Comfy traffic. Exercise auto and fixed lenses, masking, both witness settings, dense reconstruction, cancellation, and download/import. Pin/checksum the exact assets used by these tests. Model and upstream repository licenses remain separate from this integration code.

No Blender or Nuke installation is required on the inference worker. Artist machines need only Blender and access to the Comfy service. Windows client testing and production container validation remain rollout tasks.

The current witness receipt labels the CoTracker3 scaled-offline dependency `CC-BY-NC-4.0` / `internal_research_noncommercial_dependency`. This prototype option should be reviewed by the team before commercial deployment; a public integration repository does not change the model license. Witnesses can be disabled independently.

UniDepth, used by the core solve, also carries a CC-BY-NC-4.0 notice. See the [production license handoff](MODELS.md#production-license-handoff); disabling witnesses alone does not remove this consideration.
