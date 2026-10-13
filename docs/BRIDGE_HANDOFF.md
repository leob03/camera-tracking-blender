# Generic DCC bridge handoff

The current Blender add-on can serve as a Blender adapter. This handoff preserves the working prototype; it does not introduce a competing generic bridge or change your bridge contracts.

## Existing boundaries

- `vipe_remote/client.py`: standard-library HTTP client with no Blender or GPU imports. Upload, submit, poll, cancel, download and persisted job receipts.
- `vipe_remote/addon.py`: artist controls, background network thread, main-thread scene import and reconnect UI.
- `vipe_remote/bundle.py`: data-only archive validation independent of Blender.
- `vipe_remote/importer.py`, `artist_witnesses.py`, `environment.py`: Blender-specific scene construction.
- `comfyui_vipe_remote`: server-side upload/status routes and queued worker node. Launches the separately configured inference Python.

A future adapter can replace `Client` operations with the shared bridge while preserving validated bundle parsing and Blender import. Agree on job IDs, progress/cancellation semantics, source identity and download ownership before changing those interfaces.

## Current transport

| Operation | Endpoint |
| --- | --- |
| Capabilities | `GET /vipe/capabilities` |
| Upload movie bytes | `POST /vipe/upload?filename=plate.mp4` |
| Submit `ViPECameraTrack` | `POST /prompt` with supplied UUID |
| Poll queue/history | `GET /queue`, `GET /history/{id}` |
| Stage status | `GET /vipe/jobs/{id}` |
| Cancel one job | `POST /api/jobs/{id}/cancel` (fork-specific) |
| Bounded log | `GET /vipe/jobs/{id}/log` |
| Bundle download | `GET /view?filename=result.zip&subfolder=vipe/{id}&type=output` |

Upload returns a worker-relative video path, SHA256 and byte count. Submit includes that identity, source basename, UUID and settings. The source SHA256 is verified before inference and before using a selected local movie as the camera background. Output descriptors include a bundle SHA256. Neither artist filesystem paths nor server executable paths are accepted as uploaded job settings.

The node inputs are `frame_start`, `frame_end`, `max_dimension`, `masking`, `mask_phrases`, `lens_mode`, `focal_mm`, `sensor_mm`, `max_points`, plus optional `review_witnesses=true` and `dense_environment=false`. Import-only settings (timeline start, marker count, scene scale) stay in Blender. The workflow's browser node wraps the same API node.

A saved receipt contains job ID, server, local source/download paths, settings and import preferences. Bearer tokens are not persisted in receipts. Treat source paths and logs as internal production data; the public repository contains neither.

## Behaviors to retain during bridge integration

- Blender stays responsive; only its main thread modifies scene data.
- Cancel targets this job and its subprocesses, not the whole server.
- Reconnect uses the original job ID; it does not silently submit duplicate inference.
- Results are checksummed data, not downloaded Python/Blender scripts.
- Coordinate and source-frame mappings remain explicit.
- The original artist movie stays local; inference uses an uploaded copy.
- Optional review/reconstruction stages do not alter the solved camera.

The next bridge-specific changes should be discussed with the project owner before implementation.

## Live logs

`GET /vipe/jobs/{id}/log?offset=N&generation=G` returns bounded incremental JSON chunks. Offset `-1` attaches to recent output. Response fields: base64-encoded raw `data`, `offset`, `next_offset`, `size`, `generation`, `reset`, `skipped_bytes`, and `exists`. Preserve the byte cursor and generation between requests; decode UTF-8 incrementally across chunks. A missing queued job log returns `exists=false`.

Each response reads at most 32 KiB, with bounded catch-up and explicit omission/reset notices. `Client.wait(..., log=callback)` follows logs alongside job status and drains the final output on completion/error. Log transport/display failures remain independent of job success. Calling the endpoint without `offset` preserves the earlier plain-text tail response. Capabilities advertise `live_logs: "byte_cursor_v1"`.
