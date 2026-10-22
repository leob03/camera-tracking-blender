# Validation and performance

## Matched witness timing tests

Both complete shots used identical stock settings within each pair: 1280-pixel long edge, automatic lens, TrackAnything masking, 30,000 exported native points, dense environment off. Runs were serial on the same Quadro RTX 5000 (16 GB). Each enabled run used a new, isolated CoTracker evidence cache. Model weights were already present.

| Shot | Camera only | Camera + fresh witnesses | Added wall time | Total multiplier |
| --- | ---: | ---: | ---: | ---: |
| skateboard | 128.03 s | 320.15 s | +192.12 s | 2.50× |
| skateboard_alley | 124.03 s | 364.08 s | +240.05 s | 2.94× |

`skateboard` contains 262 frames; `skateboard_alley` contains 318. These are worker wall timings, including subprocess startup and packaging, excluding upload/download, queue wait and Blender import. One observation per condition is not a latency distribution. Detailed receipts are in `provenance/benchmarks.json`.

The isolated witness subprocess took 193.35 seconds on skateboard and 240.53 seconds on skateboard_alley (see exact receipts). Its independent observations are reusable; a previous full-skateboard cached-evidence reconstruction took about 29 seconds. That cached figure is not the first-run overhead.

The old approximately 126-second skateboard result included local transfer and Blender import but omitted witnesses and dense video depth. It was a different measurement scope; use the matched table above for the new option's cost.

## Verification

- Camera projection/import regressions: PASS in Blender 4.2.3 LTS. Maximum observed real-camera projection discrepancy 0.000411 pixels. Checks cover off-center intrinsics, unequal focal axes, source offsets, fractional FPS, world scale, repeated import, and invalid bundles.
- Dense geometry CPU tests: PASS for calibrated unprojection, mask holes, nonfinite depth and depth discontinuity rejection.
- Synthetic dense bundle import: PASS, packed textures and default visibility verified; maximum projection discrepancy 0.000026 pixels. Invalid face indices rejected.
- Real dense-only Comfy API and Blender import: PASS on a 32-frame selection from a 48-frame source, 640-pixel processing edge. Twelve views, 519,734 samples; maximum checked projection discrepancy 0.000092 pixels. Camera SHA256 unchanged, textures packed, Show Dense Environment control and malformed face rejection verified. Camera solve took 35.08 seconds; dense stage 41.86 seconds; local upload through download 78.36 seconds. Combined witnesses-plus-dense API/import: PASS on the same short selection, with cached independent observations; 519,753 dense samples and original artist-cross import. Both stages kept the camera unchanged. Packaged add-on 0.4.0 installation, server connection and legacy version-2 witness import: PASS in isolated Blender preferences. Full-shot `skateboard_alley` dense API/import: PASS, all 318 frames at 1280 pixels; 12 sampled views and 493,722 points. Camera solve 129.40 seconds, dense stage 192.37 seconds, total worker 321.82 seconds. Witnesses disabled for this reconstruction test. Maximum checked projection discrepancy 0.000286 pixels; source movie, packed textures, hidden defaults and Show Dense Environment verified. This checks projection consistency, not real-world geometry accuracy.

## Reproduce on your worker

```bash
python scripts/submit_video.py /path/to/plate.mp4 --url http://127.0.0.1:8189 --no-witnesses
python scripts/submit_video.py /path/to/plate.mp4 --url http://127.0.0.1:8189
python scripts/submit_video.py /path/to/plate.mp4 --url http://127.0.0.1:8189 --no-witnesses --dense
```

Then validate a returned bundle with `python tests/check_bundle.py result.zip` (NumPy required), or import in Blender:

```bash
blender -b --factory-startup --python-exit-code 1 --python tests/blender_import.py -- result.zip /path/to/original.mp4
```

Run `python tests/test_surface_geometry.py` with NumPy and OpenCV for the CPU geometry checks. To measure fresh witness work, use a separate evidence-cache directory; do not delete a cache another job is using. Stage timings are in the result manifest and the local worker `TIMINGS.json`.

## Remaining rollout checks

Actual Windows artist-machine operation, an Internet-blocked production image, generic DCC-bridge integration, multi-user scheduling/authentication policy and broader shot quality review have not been validated here. Numerical checks verify implementation and projection consistency; they do not choose a solve or certify camera quality.

## Live log validation (0.4.1 / 0.3.1)

Six CPU tests cover incremental appends, split UTF-8 characters, missing logs, truncation/replacement, bounded catch-up, final failure output, nonfatal endpoint/display errors, detach semantics, repeated real HTTP polling and display-buffer limits. Blender 4.2.3 checks verify live Text Editor updates, one-click splitting/reuse, paused inspection and resuming after completion. These tests require no model inference.

Run `python tests/test_job_logs.py` and `blender -b --factory-startup --python-exit-code 1 --python tests/blender_live_logs.py`.

The CoTracker wrapper now reports decoding, mask generation, model loading and each tracking window. An AST comparison confirms that removing these print statements restores the previous wrapper exactly; no tracking calculations or selection thresholds changed.

## Add-on 0.5.0 / node pack 0.4.0 (2026-10-19, Quadro RTX 5000 workstation)

Measured on the local worker, one job at a time. The full A/B campaign report (pinhole vs. experimental distortion on three shots, with held-out review residuals) is written to `artifacts/ab-2026-10-19/full/REPORT.md` on the development workstation.

| Check | Before | After |
| --- | ---: | ---: |
| `desert_cars` (1059 frames, 1280 px) dense stage | 585.7 s (video depth for every frame, 12 kept) | 29.0 s (12 sampled views), 412,185 points |
| `desert_cars` camera + dense worker wall | 962 s | 391 s |
| Resubmitting an existing plate | full upload | local hash + 1 request (`Plate already on worker`) |
| Review witnesses added to a finished 48-frame job (cached evidence) | full resubmission | 15 s review-only job, merged into the existing scene |
| Synthetic ACEScg EXR sequence (48 frames, 2048×1080) | not supported | 7.9 MB proxy upload, camera in plate pixels, timeline 1001 |

Short-clip distortion smoke test (`skateboard_alley` frames 1–48, 640 px): MEI solved `xi = 0.63` with a 25% shorter centre focal than the pinhole solve, while held-out review residuals were not better (3.25 vs 3.16 px median). On short, small-motion clips focal length and distortion are not separable; keep the option experimental and judge it on full shots.

Full-shot A/B (1280 px, held-out review residuals in plate pixels, same independent CoTracker evidence for both solves of a plate):

| Shot | Frames | Pinhole median / p90 | Distortion median / p90 | Solved xi | Camera solve time |
| --- | ---: | --- | --- | ---: | --- |
| skateboard_alley | 318 | 2.168 / 4.552 | 2.159 / 4.528 | 0.644 (centre focal 2690 vs 3703 px) | 121 → 155 s |
| skateboard | 262 | 1.207 / 2.791 | 1.189 / 2.782 | 0.119 | 125 → 163 s |
| desert_cars | 1059 | 0.287 / 0.559 | 0.288 / 0.562 | −0.018 | 343 → 376 s |

No shot improved by more than 1.5%. On `skateboard_alley` the distortion solve fits the evidence equally well with a 27% shorter focal length, so focal length and distortion are not separable from this footage even over the full shot. The option stays experimental and off by default; a future refinement should constrain distortion (lens grid, prior, or long-track bundle adjustment) rather than trust this one-parameter estimate. Review witnesses with cached evidence took 25–36 s on the 262–318-frame shots and 121 s on the 1059-frame shot.

Tests (Blender 4.2.3 LTS and 4.4.0, background, isolated user folders): `blender_ui_register.py`, `blender_scene_tools.py`, `blender_live_logs.py`, `blender_nuke_export.py`, `blender_import_features.py` (real 0.4.0 bundles: filmback/constant lens/square pixels, review merge, distortion import, image-sequence import), `blender_plate_proxy.py`; CPU: `test_lens_math.py`, `test_job_logs.py`, `test_exr_writer.py`. Nuke itself was not available: the `.nk` syntax, STMap input order, BakedPointCloud serialization and `win_translate` units remain to be confirmed in Nuke (see NUKE.md).

## Add-on 0.5.1 / node pack 0.4.1: solve reuse and job extensions (2026-10-22)

`skateboard_alley` frames 1–48 at 640 px, then the 96-frame synthetic EXR proxy, one job at a time on the same worker:

| Job | Worker wall | What ran |
| --- | ---: | --- |
| Fresh camera solve | 37.2 s | camera solve |
| Same plate and settings, dense on | 23.9 s | solve reused (0.2 s), dense only |
| Add Dense Environment to the first job | 26.9 s | dense only, merged into the existing collection |
| Run Review Witnesses on the first job | 12.9 s | witnesses only (cached evidence) |
| Same plate, point limit 5,000 | 0.1 s | solve reused, cloud resampled to exactly 5,000 points |
| EXR proxy resubmitted | — | upload and solve reused (identical archive bytes) |

Blender 4.2.3 and 4.4.0: `blender_extension_merge.py` (review then dense merged into one job; witnesses kept, dense hidden, summary and manifest combined, Show/Hide toggle), `blender_ui_register.py` (Add Dense flow, toggle labels), `blender_plate_proxy.py` (byte-identical rebuild, cache hit on resubmit), plus the earlier suite; 0.4.0 bundles still import.

## Add-on 0.5.2: plate and markers in Rendered view and renders (2026-10-22)

Blender 4.4.0 GUI screenshots (off-screen GPU display) of an image-sequence job: Solid and Rendered camera views show the proxy plate with the static-track and review crosses; an F12 render shows the plate composited under the markers. Frame mapping verified numerically by rendering through the compositor and matching each frame to its proxy file (sequences starting at 1001 and 0977, first/middle/last frames). `blender_review_render.py` passes on 4.2.3 and 4.4.0 (film, view transform, plate compositor, existing compositor preserved, marker geometry, render toggle).
