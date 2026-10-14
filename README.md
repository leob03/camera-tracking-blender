# Camera Tracking for Blender through ComfyUI

Select a video in Blender, submit it to a ComfyUI GPU worker, and import the solved camera, native point cloud and 3D markers. The artist machine does not need CUDA or model dependencies.

The prototype runs the reviewed ViPE camera pipeline. Two extra outputs are selectable:

| Output | Default | Purpose |
| --- | --- | --- |
| Camera + native post-BA map | Always | Solved camera, source movie background and native reconstruction points |
| Independent review witnesses | On | CoTracker observations triangulated through the frozen camera, with the original artist cross assets |
| Dense environment reconstruction | Off | Post-SLAM video depth accumulated into colored samples and textured per-view surfaces |

The witness stage assesses the camera; it does not improve the solve. Neither optional stage reruns bundle adjustment. Dense geometry is an accumulation of views, not a watertight or merged model.

## Start here

- **Artist:** [Blender installation and use](docs/BLENDER.md)
- **Worker developer:** [ComfyUI and ViPE setup](docs/WORKER_SETUP.md)
- **ViPE fork mirror:** [Committed source pin and external dependencies](docs/VIPE_HANDOFF.md)
- **DCC bridge developer:** [Integration boundaries and API](docs/BRIDGE_HANDOFF.md)
- **Models / DVC handoff:** [Exact weights, download links and container paths](docs/MODELS.md)
- **Offline deployment:** [Dependencies and model preparation](docs/OFFLINE.md)
- **Validation and measured performance:** [Test results](docs/VALIDATION.md)
- **Data conventions:** [Bundle format](docs/BUNDLE.md)

Build the installable ZIPs with a normal Python 3 interpreter:

```bash
python scripts/build_packages.py
```

Install `dist/vipe_remote-0.4.1.zip` in Blender. The separate `dist/comfyui_vipe_remote-0.3.1.zip` belongs on the ComfyUI worker. Open `workflows/ViPE Camera and Native Cloud.json` to test directly in ComfyUI.

## Scope

Tested on Blender 4.2.3 LTS and a Linux CUDA worker. Windows artist-machine testing remains to be done. This repository contains the Blender adapter and ComfyUI node pack, not the model weights, ViPE checkout, ComfyUI fork, production container or shot media. The worker currently requires the reviewed ViPE fork described in the setup guide; a clean upstream ViPE checkout is not a verified replacement.

The existing transport and importer remain separate. No changes have been made to the team's generic DCC bridge. Maya, image sequences, distortion handling, production shot IDs and a farm job browser are outside this prototype.

See [third-party provenance](THIRD_PARTY.md). An overall redistribution license for the project's original code and artist asset has not yet been selected.
