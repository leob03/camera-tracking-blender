"""Run the stock ViPE camera pipeline in its own installed Python environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from fractions import Fraction
import shutil
import subprocess
import signal
import threading
import time
import sys
import zipfile


# Bump when the camera stage changes, so older solves are no longer reused.
SOLVE_RECIPE = 2
# Kept by every camera solve so later jobs can add review witnesses or dense geometry without solving again.
SOLVE_STATE = ['plate.mp4', 'native_points.npz', 'native_slam_map.pt', 'depth_context.npz', 'solve_masks.zip', 'camera_manifest.json']


def solve_key(request):
    """Identity of a camera solve: plate checksum plus every setting that changes the solve."""
    s = request['settings']
    data = {k: s.get(k) for k in ('frame_start', 'frame_end', 'max_dimension', 'masking', 'lens_mode', 'lens_distortion', 'plate_width', 'plate_height')}
    data['mask_phrases'] = ','.join(x.strip() for x in s['mask_phrases'].split(',') if x.strip()) if s.get('masking') else ''
    if s.get('lens_mode') == 'fixed':
        data.update(focal_mm=s['focal_mm'], sensor_mm=s['sensor_mm'])
    data.update(lens_distortion=bool(s.get('lens_distortion')), plate_width=s.get('plate_width') or 0, plate_height=s.get('plate_height') or 0)
    data.update(source=request.get('identity_sha256', request['source_sha256']), recipe=SOLVE_RECIPE)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def write_cloud(output, xyz, rgb, max_points):
    import numpy as np
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1)
    ids = np.flatnonzero(valid)
    eligible = len(ids)
    if len(ids) > max_points:
        ids = ids[np.linspace(0, len(ids) - 1, max_points, dtype=int)]
    np.savez_compressed(output / "cloud.npz", xyz=xyz[ids].astype(np.float32), rgb=np.clip(rgb[ids], 0, 1).astype(np.float32), ids=ids.astype(np.int64))
    return len(ids), eligible


def solve_index(cfg):
    return Path(cfg['work_directory']) / 'solve_index.json'


def register_solve(request, cfg, work):
    if not all((work / name).is_file() for name in SOLVE_STATE):
        return
    path = solve_index(cfg)
    try:
        index = json.loads(path.read_text())
    except (OSError, ValueError):
        index = {}
    index[solve_key(request)] = request['job_id']
    temp = path.with_name(path.name + '.' + request['job_id'] + '.tmp')
    temp.write_text(json.dumps(index, indent=1))
    temp.replace(path)


def reuse_solve(request, cfg, work, output):
    """Copy a matching earlier camera solve into this job; return its job ID, or None."""
    import numpy as np
    try:
        parent = json.loads(solve_index(cfg).read_text()).get(solve_key(request))
    except (OSError, ValueError):
        return None
    if not parent or parent == request['job_id']:
        return None
    parent_work = Path(cfg['work_directory']) / parent
    try:
        parent_output = Path(json.loads((parent_work / 'request.json').read_text())['output'])
    except (OSError, ValueError, KeyError):
        return None
    if not all((parent_work / name).is_file() for name in SOLVE_STATE) or not (parent_output / 'camera.npz').is_file():
        return None
    output.mkdir(parents=True, exist_ok=True)
    for name in SOLVE_STATE + ['camera_distortion.json', 'pipeline.yaml']:
        if (parent_work / name).is_file():
            link_or_copy(parent_work / name, work / name)
    link_or_copy(parent_output / 'camera.npz', output / 'camera.npz')
    with np.load(work / 'native_points.npz', allow_pickle=False) as points:
        count, eligible = write_cloud(output, points['xyz'], points['rgb'].astype(np.float32), request['settings']['max_points'])
    manifest = json.loads((work / 'camera_manifest.json').read_text())
    manifest.update(job_id=request['job_id'], settings=request['settings'], camera_reused_from=parent)
    manifest['source']['name'] = request['source_name']
    manifest['cloud'].update(point_count=count, eligible_point_count=eligible)
    manifest['files'] = {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size} for name in ["camera.npz", "cloud.npz"]}
    (output / 'result.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return parent


def write_status(work, state, message, progress):
    data = {"state": state, "message": message, "progress": progress, "updated": time.time()}
    temp = work / "status.tmp"
    temp.write_text(json.dumps(data), encoding="utf-8")
    temp.replace(work / "status.json")
    print(f"[ViPE] {state}: {message}", flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decoded_video_info(data):
    """Use displayed frames, respecting container edit lists, for indexing and timing."""
    info = dict(data["streams"][0])
    frames = data.get("frames", [])
    if not frames:
        raise ValueError("The video contains no decodable display frames.")
    info["reported_nb_frames"] = info.get("nb_frames", "N/A")
    info["reported_avg_frame_rate"] = info.get("avg_frame_rate", "0/0")
    info["nb_frames"] = str(len(frames))
    info["frame_count_method"] = "decoded_display_frames"
    info["frame_rate_method"] = "average_rate_normalization"
    rates = {}
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = Fraction(info.get(key, "0/0"))
            if rate > 0:
                rates[key] = rate
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    rate = rates.get("avg_frame_rate", rates.get("r_frame_rate"))
    if rate is None:
        raise ValueError("The video has no usable frame rate.")
    # r_frame_rate can itself be misleading for variable-rate clips. Use it only
    # when every displayed-frame interval agrees, allowing timestamp quantization.
    try:
        ticks = [int(frame["best_effort_timestamp"]) for frame in frames]
        time_base = Fraction(info["time_base"])
        deltas = [b - a for a, b in zip(ticks, ticks[1:])]
        candidate = rates.get("r_frame_rate")
        if candidate and time_base > 0 and deltas and min(deltas) > 0:
            expected_ticks = 1 / (candidate * time_base)
            if (len(deltas) >= 2 and all(abs(delta - expected_ticks) <= 1 for delta in deltas)
                    and abs(sum(deltas) - len(deltas) * expected_ticks) <= 1):
                rate = candidate
                info["frame_rate_method"] = "decoded_constant_cadence"
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        pass
    info["avg_frame_rate"] = str(rate)
    return info


def probe_video(path, ffprobe, *, decode_frames=False):
    command = [ffprobe, "-v", "error", "-threads", "2", "-select_streams", "v:0", "-show_streams"]
    if decode_frames:
        command += ["-show_frames", "-show_entries", "stream:frame=best_effort_timestamp"]
    command += ["-of", "json", str(path)]
    data = json.loads(subprocess.check_output(command))
    if not data.get("streams"):
        raise ValueError("The source has no video stream.")
    if decode_frames:
        return decoded_video_info(data)
    info = data["streams"][0]
    if info.get("nb_frames", "N/A") == "N/A":
        count = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-threads", "2", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)]))
        info["nb_frames"] = count["streams"][0]["nb_read_frames"]
    return info


def prepare_plate(source, plate, cfg, first, end, fps, width, height):
    """Retain one output frame per selected display frame, with exact rational timing."""
    fps = Fraction(fps)
    time_base = f"{fps.denominator}/{fps.numerator}"
    filters = f"trim=start_frame={first}:end_frame={end},settb=expr={time_base},setpts=N,scale={width}:{height}:flags=area"
    subprocess.run([cfg["ffmpeg"], "-v", "error", "-threads", "2", "-noautorotate", "-i", str(source), "-map", "0:v:0", "-vf", filters, "-an", "-fps_mode", "passthrough", "-enc_time_base", time_base, "-frames:v", str(end - first), "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(plate)], check=True)
    prepared = probe_video(plate, cfg["ffprobe"], decode_frames=True)
    if int(prepared["nb_frames"]) != end - first:
        raise ValueError(f"Video preparation produced {prepared['nb_frames']} decoded frames; expected {end - first} for source frames {first + 1}–{end}.")
    if Fraction(prepared["avg_frame_rate"]) != fps:
        raise ValueError(f"Video preparation changed the frame rate from {fps} to {prepared['avg_frame_rate']} fps.")
    return prepared


def solve(request_path, config_path):
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from vipe.pipeline import make_pipeline
    from vipe.streams.raw_mp4_stream import RawMp4Stream
    from vipe.streams.base import ProcessedVideoStream, AssignAttributesProcessor, FrameAttribute
    from vipe.utils.cameras import CameraType
    from vipe.utils.io import ArtifactPath, save_slam_stage_artifacts
    from vipe.utils.logging import configure_logging

    request = json.loads(request_path.read_text())
    cfg = json.loads(config_path.read_text())
    work = request_path.parent
    source = Path(request["source"])
    output = Path(request["output"])
    settings = request["settings"]
    write_status(work, "running", "Preparing video", 5)
    if digest(source) != request["source_sha256"]:
        raise ValueError("Uploaded video checksum changed; upload the plate again.")
    info = probe_video(source, cfg["ffprobe"], decode_frames=True)
    print(f"[ViPE] Video: {info['nb_frames']} decoded display frames at {info['avg_frame_rate']} fps; container reports {info['reported_nb_frames']} frames at {info['reported_avg_frame_rate']} fps.", flush=True)
    width, height = int(info["width"]), int(info["height"])
    fps = Fraction(info["avg_frame_rate"])
    if fps <= 0:
        raise ValueError("The video has no usable frame rate.")
    if info.get("sample_aspect_ratio", "1:1") not in ("1:1", "0:1", "N/A"):
        raise ValueError("This prototype requires a square-pixel video plate.")
    if any(int(x.get("rotation", 0)) % 360 for x in info.get("side_data_list", [])):
        raise ValueError("Bake the video's rotation metadata into the pixels before submitting.")
    total = int(info["nb_frames"])
    first = settings["frame_start"] - 1
    end = total if settings["frame_end"] == 0 else settings["frame_end"]
    if not 0 <= first < end <= total:
        raise ValueError(f"Frame range must lie within the video's {total} frames (end 0 means all).")
    count = end - first
    if count < 2:
        raise ValueError("Camera tracking needs at least two frames.")
    maximum = settings["max_dimension"]
    scale = min(1.0, maximum / max(width, height)) if maximum else 1.0
    process_w = max(2, round(width * scale / 2) * 2)
    process_h = max(2, round(height * scale / 2) * 2)
    # An uploaded proxy can stand in for a larger plate; report the camera in plate pixels.
    plate_w, plate_h = settings.get("plate_width") or width, settings.get("plate_height") or height
    if abs(plate_w / plate_h - width / height) > 0.01 * width / height:
        raise ValueError("The uploaded proxy and the declared plate size have different aspect ratios.")
    sx, sy = plate_w / process_w, plate_h / process_h
    # Square-pixel plates: keep the horizontal field of view and one focal length, so even-pixel
    # rounding of the processing size never invents a pixel aspect.
    kscale = np.array([sx, sx, sx, sy])
    plate = work / "plate.mp4"
    prepare_plate(source, plate, cfg, first, end, fps, process_w, process_h)
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    if not torch.cuda.is_available():
        raise RuntimeError("The ViPE worker has no usable CUDA GPU.")
    torch.cuda.set_per_process_memory_fraction(cfg.get("gpu_memory_fraction", 0.9))
    with initialize_config_dir(version_base=None, config_dir=str(Path(cfg["vipe_root"]) / "configs")):
        configuration = compose(config_name="default", overrides=["pipeline=default", "streams=raw_mp4_stream"])
    pipeline_cfg = configuration.pipeline
    pipeline_cfg.output.path = str(work / "vipe")
    pipeline_cfg.output.save_artifacts = False
    pipeline_cfg.output.save_slam_map = False
    pipeline_cfg.output.save_viz = False
    pipeline_cfg.init.async_prefetch = bool(cfg.get("async_prefetch", False))
    pipeline_cfg.init.prefetch_queue_size = int(cfg.get("prefetch_queue_size", 1))
    pipeline_cfg.slam.buffer = max(128, min(count + 64, cfg.get("slam_buffer_limit", 1024)))
    pipeline_cfg.slam.sparse_tracks.name = "dummy"
    pipeline_cfg.post.depth_align_model = None
    if settings["masking"]:
        pipeline_cfg.init.instance.phrases = [x.strip() for x in settings["mask_phrases"].split(",") if x.strip()]
    else:
        pipeline_cfg.init.instance = None
    stream = RawMp4Stream(plate)
    if settings["lens_mode"] == "fixed":
        fx_original = settings["focal_mm"] / settings["sensor_mm"] * width
        intrinsics = torch.tensor([fx_original * process_w / width, fx_original * process_h / height, process_w / 2, process_h / 2], device="cuda", dtype=torch.float32)
        stream = ProcessedVideoStream(stream, [AssignAttributesProcessor({FrameAttribute.INTRINSICS: [intrinsics] * count, FrameAttribute.CAMERA_TYPE: [CameraType.PINHOLE] * count})])
        pipeline_cfg.init.intrinsics = "gt"
        pipeline_cfg.slam.optimize_intrinsics = False
    elif settings.get("lens_distortion"):
        # The offline model set has only pinhole GeoCalib weights: start the unified (MEI)
        # model from that focal length with xi = 0 and let bundle adjustment refine both.
        import numpy as _np
        from vipe.pipeline.processors import GeoCalibIntrinsicsProcessor
        write_status(work, "running", "Estimating initial lens", 12)
        fov_y = GeoCalibIntrinsicsProcessor(stream, camera_type=CameraType.PINHOLE).fov_y
        torch.cuda.empty_cache()
        focal = process_h / (2 * _np.tan(fov_y / 2))
        intrinsics = torch.tensor([focal, focal, process_w / 2, process_h / 2, 0.0], device="cuda", dtype=torch.float32)
        stream = ProcessedVideoStream(stream, [AssignAttributesProcessor({FrameAttribute.INTRINSICS: [intrinsics] * count, FrameAttribute.CAMERA_TYPE: [CameraType.MEI] * count})])
        pipeline_cfg.init.intrinsics = "gt"
        pipeline_cfg.init.camera_type = "mei"
        pipeline_cfg.slam.optimize_intrinsics = True
        pipeline_cfg.slam.ba.fused = False  # the fused solver is pinhole-only
    else:
        pipeline_cfg.init.intrinsics = "geocalib"
        pipeline_cfg.slam.optimize_intrinsics = True
    OmegaConf.save(pipeline_cfg, work / "pipeline.yaml", resolve=True)
    configure_logging()
    write_status(work, "running", "Tracking camera and reconstructing native points", 20)
    pipeline = make_pipeline(pipeline_cfg)
    pipeline.return_payload = True
    # Always keep the solve masks and depth context so dense geometry can be added later.
    from dense_stage import capture_masks
    mask_archive = capture_masks(pipeline, work)
    try:
        slam = pipeline.run(stream).payload
    finally:
        mask_archive.close()
    np.savez_compressed(work / 'depth_context.npz', poses=slam.get_view_trajectory(0).data.detach().cpu().numpy(), intrinsics=slam.intrinsics[0].detach().cpu().numpy())
    write_status(work, "running", "Saving frozen camera and native points", 65)
    poses = slam.get_view_trajectory(0).matrix().detach().cpu().numpy().astype(np.float64)
    solved = slam.intrinsics[0].detach().cpu().numpy().astype(np.float64)
    k = solved[:4] * kscale
    k[1] = k[0]
    distortion = None
    if settings.get("lens_distortion"):
        xi = float(solved[4])
        # The exported camera is the pinhole camera of the undistorted plate (unit slope at the centre).
        distortion = {"model": "mei", "fx": float(k[0]), "fy": float(k[1]), "cx": float(k[2]), "cy": float(k[3]), "xi": xi, "undistorted_focal_px": float(k[0] / (1 + xi)), "width": plate_w, "height": plate_h, "convention": "opencv_pixel_centers"}
        (work / "camera_distortion.json").write_text(json.dumps(distortion, indent=2))
        k = np.array([k[0] / (1 + xi), k[1] / (1 + xi), k[2], k[3]])
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all() or not np.isfinite(k).all() or np.any(k[:2] <= 0):
        raise ValueError("ViPE returned an incomplete or non-finite camera.")
    xyz = slam.slam_map.dense_disp_xyz.detach().cpu().numpy()
    rgb = slam.slam_map.dense_disp_rgb.detach().cpu().numpy()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "camera.npz", c2w=poses, intrinsics=np.repeat(k[None], count, axis=0), source_frame_indices=np.arange(first, end, dtype=np.int32))
    point_count, eligible = write_cloud(output, xyz, rgb, settings["max_points"])
    # All native points (not just the exported sample), so a reused solve can honour another point limit.
    np.savez(work / "native_points.npz", xyz=xyz.astype(np.float32), rgb=rgb.astype(np.float16))
    slam.slam_map.save(work / "native_slam_map.pt")
    save_slam_stage_artifacts(ArtifactPath(work / "vipe", "plate"), slam)
    stages = {}
    for name, values in [("initial", slam.initial_intrinsics), ("pre_global_ba", slam.pre_global_ba.intrinsics if slam.pre_global_ba else None), ("post_global_ba", slam.post_global_ba.intrinsics if slam.post_global_ba else None)]:
        if values is not None:
            raw = values.detach().cpu().numpy().reshape(-1)
            stages[name] = (raw[:4] * kscale).tolist() + raw[4:5].tolist()
    manifest = {"schema": "vipe-camera-bundle", "version": 1, "job_id": request["job_id"], "source": {"name": request["source_name"], "sha256": request.get("identity_sha256", request["source_sha256"]), **({"kind": "image_sequence", "first_frame": request["plate"]["first_frame"], "colorspace": request["plate"].get("colorspace", ""), "pattern": request["plate"].get("pattern", "")} if request.get("plate") else {}), "width": plate_w, "height": plate_h, "frame_count": total, "fps_num": fps.numerator, "fps_den": fps.denominator, **({"upload_width": width, "upload_height": height} if (plate_w, plate_h) != (width, height) else {})}, "selection": {"first_source_frame": first, "last_source_frame": end - 1, "count": count}, "camera": {"file": "camera.npz", "model": "PINHOLE_UNDISTORTED" if distortion else "PINHOLE", "convention": "opencv_camera_to_world", "intrinsics_units": "original_source_pixels", "distortion": [], **({"lens_distortion": distortion} if distortion else {}), "world_scale": "ViPE estimated scale; not surveyed"}, "cloud": {"file": "cloud.npz", "kind": "native_post_ba_slam_map", "point_count": point_count, "eligible_point_count": eligible, "filtering": "ViPE depth consistency and solve masks; no additional DeepLab filter", "sampling": "evenly spaced source point IDs; no score ranking"}, "settings": settings, "processing": {"width": process_w, "height": process_h, "dense_depth_computed": False, "backend": "stock", "camera_model": "mei" if distortion else "pinhole", "lens_stages_original_pixels": stages}, "files": {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size} for name in ["camera.npz", "cloud.npz"]}}
    manifest["processing"]["video_preparation"] = {key: info[key] for key in ("reported_nb_frames", "reported_avg_frame_rate", "frame_count_method", "frame_rate_method")}
    (output / "result.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (work / "camera_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(request_path, config_path):
    # Each GPU stage exits before the next starts. Neither optional stage changes the camera.
    started = time.monotonic()
    timings = {}
    common = ['--request', str(request_path), '--config', str(config_path)]
    request = json.loads(request_path.read_text())
    work, output = request_path.parent, Path(request['output'])
    cfg = json.loads(config_path.read_text())
    review_of = request.get('review_of')
    reused = None
    if not review_of:
        reused = reuse_solve(request, cfg, work, output)
        if reused:
            write_status(work, 'running', f'Reusing camera solve from job {reused[:8]}', 60)
            timings['camera_reused'] = time.monotonic() - started
    stages = [] if review_of or reused else [('camera_solve', Path(__file__), ['--solve-only'])]
    if review_of:
        write_status(work, 'running', 'Preparing review', 5)
    if request['settings'].get('review_witnesses', False):
        stages.append(('review_witnesses', Path(__file__).with_name('evidence_stage.py'), []))
    if request['settings'].get('dense_environment', False):
        stages.append(('dense_environment', Path(__file__).with_name('dense_stage.py'), []))
    if request['settings'].get('train_splat', False):
        stages.append(('gaussian_splat', Path(__file__).with_name('splat_stage.py'), []))
    for name, script, flags in stages:
        before = time.monotonic()
        subprocess.run([sys.executable, str(script), *flags, *common], check=True)
        timings[name] = time.monotonic() - before
        shutil.rmtree(work / 'cache', ignore_errors=True)
        if name == 'camera_solve':
            register_solve(request, cfg, work)
    write_status(work, 'running', 'Packaging tracking results', 95)
    manifest = json.loads((output / 'result.json').read_text())
    if review_of:
        manifest['job_id'], manifest['review_of'] = request['job_id'], review_of
        manifest['timings_seconds'] = {**manifest.get('timings_seconds', {}), **{'review_' + k if k == 'worker_before_packaging' else k: v for k, v in timings.items()}, 'review_worker_before_packaging': time.monotonic() - started}
    else:
        manifest['timings_seconds'] = {**timings, 'worker_before_packaging': time.monotonic() - started}
    (output / 'result.json').write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output / 'result.zip.tmp', 'w', zipfile.ZIP_STORED) as archive:
        for name in ['result.json', *manifest['files']]:
            archive.write(output / name, name)
    (output / 'result.zip.tmp').replace(output / 'result.zip')
    timings['worker_total'] = time.monotonic() - started
    (work / 'TIMINGS.json').write_text(json.dumps(timings, indent=2))
    write_status(work, 'complete', 'Tracking results ready', 100)
    print('VIPE_BUNDLE_COMPLETE', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument('--solve-only', action='store_true')
    args = parser.parse_args()
    parent_pid = os.getppid()
    def stop_if_parent_exits():
        while os.getppid() == parent_pid:
            time.sleep(1)
        write_status(args.request.parent, "failed", "ComfyUI stopped while the job was running; resubmit this job.", 0)
        if os.getpgrp() == os.getpid():
            os.killpg(os.getpid(), signal.SIGTERM)
        else:
            os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=stop_if_parent_exits, daemon=True).start()
    try:
        (solve if args.solve_only else run)(args.request, args.config)
    except Exception as exc:
        status_path = args.request.parent / "status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        if previous.get("state") != "failed":
            write_status(args.request.parent, "failed", str(exc)[-1500:], 0)
        raise
