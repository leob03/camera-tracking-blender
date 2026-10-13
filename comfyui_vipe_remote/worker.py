"""Run the stock ViPE camera pipeline in its own installed Python environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import signal
import threading
import time
import sys
import zipfile


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


def probe_video(path, ffprobe):
    command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path)]
    info = json.loads(subprocess.check_output(command))["streams"][0]
    if info.get("nb_frames", "N/A") == "N/A":
        count = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)]))
        info["nb_frames"] = count["streams"][0]["nb_read_frames"]
    return info


def solve(request_path, config_path):
    import numpy as np
    import torch
    from fractions import Fraction
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
    info = probe_video(source, cfg["ffprobe"])
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
    plate = work / "plate.mp4"
    subprocess.run([cfg["ffmpeg"], "-v", "error", "-threads", "2", "-noautorotate", "-i", str(source), "-vf", f"trim=start_frame={first}:end_frame={end},setpts=N/(({fps})*TB),scale={process_w}:{process_h}:flags=area", "-an", "-r", str(fps), "-frames:v", str(count), "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(plate)], check=True)
    prepared = probe_video(plate, cfg["ffprobe"])
    if int(prepared["nb_frames"]) != count:
        raise ValueError("Video preparation did not preserve the selected frame count.")
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
    pipeline_cfg.init.async_prefetch = False
    pipeline_cfg.init.prefetch_queue_size = 1
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
    else:
        pipeline_cfg.init.intrinsics = "geocalib"
        pipeline_cfg.slam.optimize_intrinsics = True
    OmegaConf.save(pipeline_cfg, work / "pipeline.yaml", resolve=True)
    configure_logging()
    write_status(work, "running", "Tracking camera and reconstructing native points", 20)
    pipeline = make_pipeline(pipeline_cfg)
    pipeline.return_payload = True
    mask_archive = None
    if settings.get('dense_environment', False):
        from dense_stage import capture_masks
        mask_archive = capture_masks(pipeline, work)
    try:
        slam = pipeline.run(stream).payload
    finally:
        if mask_archive is not None:
            mask_archive.close()
    if settings.get('dense_environment', False):
        np.savez_compressed(work / 'depth_context.npz', poses=slam.get_view_trajectory(0).data.detach().cpu().numpy(), intrinsics=slam.intrinsics[0].detach().cpu().numpy())
    write_status(work, "running", "Saving frozen camera and native points", 65)
    poses = slam.get_view_trajectory(0).matrix().detach().cpu().numpy().astype(np.float64)
    k = slam.intrinsics[0].detach().cpu().numpy()[:4].astype(np.float64)
    k *= np.array([width / process_w, height / process_h, width / process_w, height / process_h])
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all() or not np.isfinite(k).all() or np.any(k[:2] <= 0):
        raise ValueError("ViPE returned an incomplete or non-finite camera.")
    xyz = slam.slam_map.dense_disp_xyz.detach().cpu().numpy()
    rgb = slam.slam_map.dense_disp_rgb.detach().cpu().numpy()
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1)
    ids = np.flatnonzero(valid)
    eligible = len(ids)
    if len(ids) > settings["max_points"]:
        ids = ids[np.linspace(0, len(ids) - 1, settings["max_points"], dtype=int)]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "camera.npz", c2w=poses, intrinsics=np.repeat(k[None], count, axis=0), source_frame_indices=np.arange(first, end, dtype=np.int32))
    np.savez_compressed(output / "cloud.npz", xyz=xyz[ids].astype(np.float32), rgb=np.clip(rgb[ids], 0, 1).astype(np.float32), ids=ids.astype(np.int64))
    slam.slam_map.save(work / "native_slam_map.pt")
    save_slam_stage_artifacts(ArtifactPath(work / "vipe", "plate"), slam)
    stages = {}
    for name, values in [("initial", slam.initial_intrinsics), ("pre_global_ba", slam.pre_global_ba.intrinsics if slam.pre_global_ba else None), ("post_global_ba", slam.post_global_ba.intrinsics if slam.post_global_ba else None)]:
        if values is not None:
            vals = values.detach().cpu().numpy().reshape(-1)[:4] * np.array([width / process_w, height / process_h, width / process_w, height / process_h])
            stages[name] = vals.tolist()
    manifest = {"schema": "vipe-camera-bundle", "version": 1, "job_id": request["job_id"], "source": {"name": request["source_name"], "sha256": request["source_sha256"], "width": width, "height": height, "frame_count": total, "fps_num": fps.numerator, "fps_den": fps.denominator}, "selection": {"first_source_frame": first, "last_source_frame": end - 1, "count": count}, "camera": {"file": "camera.npz", "model": "PINHOLE", "convention": "opencv_camera_to_world", "intrinsics_units": "original_source_pixels", "distortion": [], "world_scale": "ViPE estimated scale; not surveyed"}, "cloud": {"file": "cloud.npz", "kind": "native_post_ba_slam_map", "point_count": len(ids), "eligible_point_count": eligible, "filtering": "ViPE depth consistency and solve masks; no additional DeepLab filter", "sampling": "evenly spaced source point IDs; no score ranking"}, "settings": settings, "processing": {"width": process_w, "height": process_h, "dense_depth_computed": False, "backend": "stock", "lens_stages_original_pixels": stages}, "files": {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size} for name in ["camera.npz", "cloud.npz"]}}
    (output / "result.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(request_path, config_path):
    # Each GPU stage exits before the next starts. Neither optional stage changes the camera.
    started = time.monotonic()
    timings = {}
    common = ['--request', str(request_path), '--config', str(config_path)]
    request = json.loads(request_path.read_text())
    work, output = request_path.parent, Path(request['output'])
    stages = [('camera_solve', Path(__file__), ['--solve-only'])]
    if request['settings'].get('review_witnesses', True):
        stages.append(('review_witnesses', Path(__file__).with_name('evidence_stage.py'), []))
    if request['settings'].get('dense_environment', False):
        stages.append(('dense_environment', Path(__file__).with_name('dense_stage.py'), []))
    for name, script, flags in stages:
        before = time.monotonic()
        subprocess.run([sys.executable, str(script), *flags, *common], check=True)
        timings[name] = time.monotonic() - before
        shutil.rmtree(work / 'cache', ignore_errors=True)
    write_status(work, 'running', 'Packaging tracking results', 95)
    manifest = json.loads((output / 'result.json').read_text())
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
