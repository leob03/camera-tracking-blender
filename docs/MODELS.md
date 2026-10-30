# Models for the offline ComfyUI worker

This is the model handoff for the current camera-tracking prototype, audited against the reviewed worker on 2026-10-13. **All models run on the GPU worker. Blender clients need none of these weights.**

Use the [machine-readable manifest](../provenance/model-manifest.json) for exact download URLs, SHA256 checksums, byte counts, Hugging Face revisions and destination paths. It contains **12 weight files plus 5 configuration/tokenizer files**. The complete set is **4.34 GB / 4.04 GiB**, excluding code, Python/CUDA environments, model notices and job scratch space.

| Enabled features | Unique model/config bytes (GiB) |
| --- | ---: |
| Camera + automatic lens + masking | 3.068 |
| Above + review crosses (current defaults) | 3.204 |
| Above + dense environment (all features) | 4.040 |

Optional Gaussian splat training adds **no model files**; it needs the native LichtFeld trainer described in [Gaussian splats](GAUSSIAN_SPLATS.md#worker-setup-and-production-deployment).

## Downloads by stage

Download only the listed variants. Direct weight links follow below; pinned Hugging Face auxiliary-file links are in the manifest. Google Drive links are the file IDs used by the reviewed ViPE loaders; use gdown or the Drive download UI in the connected preparation environment and verify the resulting checksum.

### Camera solve — always

| Asset / download | Filename | Size (MiB) |
| --- | --- | ---: |
| [DROID-SLAM network](https://drive.google.com/uc?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh) | `droid.pth` | 15.3 |
| [UniDepth V2 ViT-L/14](https://huggingface.co/lpiccinelli/unidepth-v2-vitl14/resolve/52b349b514bd8b47642f67ac78cb7b5dc5c51dd9/model.safetensors) | `model.safetensors` | 1349.8 |

### Automatic lens — default

| Asset / download | Filename | Size (MiB) |
| --- | --- | ---: |
| [GeoCalib pinhole](https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar) | `pinhole.tar` | 110.7 |

### TrackAnything solve masking — default on

| Asset / download | Filename | Size (MiB) |
| --- | --- | ---: |
| [SAM ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | `sam_vit_b_01ec64.pth` | 357.7 |
| [DeAOT R50](https://drive.google.com/uc?id=1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ) | `R50_DeAOTL_PRE_YTB_DAV.pth` | 225.6 |
| [GroundingDINO Swin-T](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth) | `groundingdino_swint_ogc.pth` | 661.8 |
| [BERT base uncased](https://huggingface.co/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/model.safetensors) | `model.safetensors` | 420.0 |

### Independent review crosses — optional, default on

| Asset / download | Filename | Size (MiB) |
| --- | --- | ---: |
| [DeepLabV3 MobileNetV3 Large](https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth) | `deeplabv3_mobilenet_v3_large-fc3c493d.pth` | 42.3 |
| [CoTracker3 scaled offline](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth) | `scaled_offline.pth` | 97.2 |

### Dense environment — optional, default off

| Asset / download | Filename | Size (MiB) |
| --- | --- | ---: |
| [Video Depth Anything Small](https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth) | `video_depth_anything_vits.pth` | 111.0 |
| [PriorDA base: Depth Anything V2 ViT-B](https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/25b3bfcf85209fb015e3b15259f4df8b6f13bb16/depth_anything_v2_vitb.pth) | `depth_anything_v2_vitb.pth` | 371.9 |
| [PriorDA ViT-B](https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/25b3bfcf85209fb015e3b15259f4df8b6f13bb16/prior_depth_anything_vitb.pth) | `prior_depth_anything_vitb.pth` | 372.5 |

Dense reconstruction reuses UniDepth and the solve masks. Both PriorDA files are required: the base ViT-B checkpoint and the conditioned PriorDA checkpoint. The frozen and conditioned base models use the same base checkpoint, so warehouse that file once.

Review crosses use CoTracker plus independent DeepLab masks. Turning this option off removes both dependencies. The camera solve itself does not use them. Keep the CoTracker source available at the configured `cotracker_repository`, pinned to `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` from [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker). The worker imports its local predictor; it does not need to clone that repo during inference.

### Hugging Face auxiliary files and exact revisions

| Loader repository ID | Revision | Files to retain |
| --- | --- | --- |
| `lpiccinelli/unidepth-v2-vitl14` | `52b349b514bd8b47642f67ac78cb7b5dc5c51dd9` | `config.json`, `model.safetensors` |
| `bert-base-uncased` | `86b5e0934494bd15c9632b12f734a8a67f723594` | `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt` |
| `Rain729/Prior-Depth-Anything` | `25b3bfcf85209fb015e3b15259f4df8b6f13bb16` | `depth_anything_v2_vitb.pth`, `prior_depth_anything_vitb.pth` |

The BERT Hub page redirects to [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594), but the current loader requests **`bert-base-uncased`**. Preserve its cache directory name **`models--bert-base-uncased`**. A cache populated only under `models--google-bert--bert-base-uncased` will not satisfy that lookup offline.

The pinned UniDepth config sets `pixel_encoder.pretrained` to `null`; its full checkpoint supplies the DINOv2 backbone weights. This recipe requires no separate DINOv2 checkpoint. Full DeepLab weights likewise supply the MobileNet backbone. DA3, BootsTAPIR, SAM ViT-H, and the large Video Depth Anything variant are not used by this worker recipe.

## DVC and container paths

The inspected Comfy fork uses per-file `.dvc` descriptors and model symlinks into a DVC content store. The ViPE worker currently reads `torch_home`, `hf_home` and `cotracker_checkpoint` from its own JSON config. It does **not** discover these through ComfyUI's normal model-folder registry.

A proposed layout that fits the existing worker without changing its loaders is a DVC-managed `models/vipe/` tree, mounted at `/models/vipe` inside the container. These are deployment suggestions; this audit did not add DVC objects or change the fork.

```json
{
  "torch_home": "/models/vipe/torch",
  "hf_home": "/models/vipe/huggingface",
  "cotracker_checkpoint": "/models/vipe/cotracker/scaled_offline.pth",
  "cotracker_repository": "/opt/camera-tracking/vendor/cotracker"
}
```

Merge these keys into the complete [worker configuration](../worker-config.example.json). Set `VIPE_WORKER_CONFIG` to that file. The manifest's `destination` fields are relative to `/models/vipe`.

### Torch files

```text
torch/hub/droid_slam/droid.pth
torch/hub/geocalib/pinhole.tar
torch/hub/sam/sam_vit_b_01ec64.pth
torch/hub/aot/R50_DeAOTL_PRE_YTB_DAV.pth
torch/hub/checkpoints/groundingdino_swint_ogc.pth
torch/hub/checkpoints/deeplabv3_mobilenet_v3_large-fc3c493d.pth
torch/hub/checkpoints/video_depth_anything_vits.pth
cotracker/scaled_offline.pth
```

GeoCalib downloads as `geocalib-pinhole.tar`, but **store it as `pinhole.tar`** at the path above; this is the loader's explicit cache filename. Do not extract the checkpoint archive.

### Hugging Face cache

For each entry in the manifest's `hf_snapshots` array, package:

```text
huggingface/hub/<cache-directory>/refs/main
huggingface/hub/<cache-directory>/snapshots/<revision>/<listed-files>
```

`refs/main` must contain the exact revision string from the manifest. It is needed because the current loaders request their default revision at runtime. Two valid packaging choices:

1. Materialize the listed snapshot files as regular files, and warehouse them plus `refs/main` through DVC. This avoids carrying Hugging Face's extra blob symlink layer.
2. Preserve the existing `blobs/`, `snapshots/` relative symlinks and `refs/main` together. The manifest records each snapshot file's `hf_blob_name` for this layout.

Do not ship snapshot symlinks without their targets. DVC checkout symlinks also need targets accessible **inside the container**; a link to a host-only `/mnt/share/cache/dvc/...` path will be broken unless that store is mounted at the matching path. Materializing the small selected model tree during image assembly is another option. There is no need to copy a whole shared Hugging Face or DVC cache.

## Build-time preparation and validation

1. In the connected preparation environment, download the selected manifest files, verify SHA256, construct the cache paths and three `refs/main` files, and add the assets to the team's DVC warehouse. Pin the DVC descriptors in the production source revision.
2. During image assembly or artifact staging, retrieve those DVC objects and make them available at the configured paths. Offline inference must not depend on running `dvc pull` against an unreachable remote.
3. Include the reviewed ViPE code/environment, compiled extensions, CoTracker source if enabled, and FFmpeg/FFprobe. See [worker setup](WORKER_SETUP.md) and [offline dependencies](OFFLINE.md); weights alone are insufficient.
4. Test the assembled worker with external networking blocked and all intended feature combinations. The worker already sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Torch Hub and gdown can still attempt downloads when a checkpoint is missing, so these flags alone are not proof of offline readiness.

This audit read selected source files, existing checksum receipts and five small auxiliary files. It did not download model binaries, rehash the large weights, run inference or certify an offline container. Existing recorded weight hashes identify the study's assets; validate downloaded bytes against them during staging. URLs containing `main` are checksum-bound by this manifest.

## Production license handoff

The [UniDepth license](https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE) and [CoTracker3 model card](https://huggingface.co/facebook/cotracker3) specify **CC-BY-NC-4.0**. UniDepth is used by the core solve, so disabling optional crosses does not remove that issue. Confirm the team's permission for commercial production before rollout. The vendored Segment-and-Track-Anything files also carry AGPL-3.0 notices. Preserve the applicable model/code notices and review the exact approved versions; this model inventory is not a license clearance.
