# Changes

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
