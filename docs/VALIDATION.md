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
