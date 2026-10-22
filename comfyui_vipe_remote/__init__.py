"""ComfyUI queue node and streamed plate upload for the ViPE camera worker."""
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

from aiohttp import web
import folder_paths
import comfy.model_management as model_management
from comfy.utils import ProgressBar
from server import PromptServer


def config_path():
    value = os.environ.get("VIPE_WORKER_CONFIG")
    if not value:
        raise RuntimeError("Set VIPE_WORKER_CONFIG to the worker configuration JSON and restart ComfyUI.")
    return Path(value)


def config():
    return json.loads(config_path().read_text(encoding="utf-8"))


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def job_id(value):
    return str(uuid.UUID(value))


def work_path(value):
    return Path(config()["work_directory"]) / job_id(value)


def save_status(work, state, message, progress=0):
    temp = work / "server-status.tmp"
    temp.write_text(json.dumps({"state": state, "message": message, "progress": progress, "updated": time.time()}))
    temp.replace(work / "status.json")


def resolve_video(value):
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Choose a video uploaded to this ComfyUI worker.")
    return path


FRAME_ARCHIVE_LIMIT = 20000


def prepare_frame_archive(archive_path, work, cfg):
    """Turn an uploaded proxy-frame ZIP (from an EXR/DPX sequence) into the worker's source movie."""
    import zipfile
    frames = work / "frames"
    frames.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if "plate.json" not in names or len(members) > FRAME_ARCHIVE_LIMIT + 1 or len(set(names)) != len(names):
            raise ValueError("The frame archive needs plate.json and at most 20000 frames.")
        plate = json.loads(archive.read("plate.json"))
        images = sorted(n for n in names if n != "plate.json")
        suffix = Path(images[0]).suffix.lower() if images else ""
        expected = [f"frames/{i:06d}{suffix}" for i in range(len(images))]
        if suffix not in (".jpg", ".png") or images != expected:
            raise ValueError("Frame archive members must be frames/000000.jpg (or .png) numbered without gaps.")
        if sum(m.file_size for m in members) > cfg.get("max_upload_bytes", 1024**3) * 2:
            raise ValueError("The frame archive expands beyond the upload limit.")
        for index, name in enumerate(images):
            (frames / f"{index:06d}{suffix}").write_bytes(archive.read(name))
    for key in ("width", "height", "fps_num", "fps_den", "first_frame"):
        if type(plate.get(key)) is not int or plate[key] < (0 if key == "first_frame" else 1):
            raise ValueError(f"Frame archive plate.json has an invalid {key}.")
    movie = work / "source.mp4"
    fps = f"{plate['fps_num']}/{plate['fps_den']}"
    subprocess.run([cfg["ffmpeg"], "-v", "error", "-threads", "2", "-framerate", fps, "-i", str(frames / f"%06d{suffix}"), "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-r", fps, str(movie)], check=True)
    shutil.rmtree(frames)
    return movie, plate


def validate_settings(values):
    if values["frame_start"] < 1 or values["frame_end"] < 0 or (values["frame_end"] and values["frame_end"] < values["frame_start"]):
        raise ValueError("Use source frames starting at 1; end 0 means the full video.")
    if values["lens_mode"] not in ("auto", "fixed"):
        raise ValueError("Unknown lens mode.")
    if any(not math.isfinite(values[k]) or values[k] <= 0 for k in ("focal_mm", "sensor_mm")):
        raise ValueError("Focal length and sensor width must be positive finite numbers.")
    if values["max_dimension"] < 0 or (values["max_dimension"] and values["max_dimension"] < 64):
        raise ValueError("Processing size must be at least 64 pixels, or 0 for original resolution.")
    if not 1 <= values["max_points"] <= 1000000:
        raise ValueError("Point limit must be between 1 and 1,000,000.")
    if values["masking"] and not values["mask_phrases"].strip():
        raise ValueError("Enter the object categories to exclude, or disable masking.")
    if values.get("lens_distortion") and values["lens_mode"] != "auto":
        raise ValueError("Lens distortion estimation requires the automatic lens mode.")
    if values.get("lens_distortion") and values.get("dense_environment"):
        raise ValueError("Dense environment reconstruction does not support lens distortion yet; disable one of them.")
    if values.get("dense_mode", "fast") not in ("fast", "video"):
        raise ValueError("Unknown dense mode.")
    if bool(values.get("plate_width")) != bool(values.get("plate_height")):
        raise ValueError("Declare both plate width and height, or neither.")


OPTIONAL_INPUTS = {"review_witnesses": ("BOOLEAN", {"default": False}), "dense_environment": ("BOOLEAN", {"default": False}), "dense_mode": (["fast", "video"], {"default": "fast"}), "lens_distortion": ("BOOLEAN", {"default": False}), "plate_width": ("INT", {"default": 0, "min": 0}), "plate_height": ("INT", {"default": 0, "min": 0})}
OPTIONAL_DEFAULTS = {name: spec[1]["default"] for name, spec in OPTIONAL_INPUTS.items()}


def worker_environment(cfg, work):
    env = dict(os.environ)
    threads = str(cfg.get("cpu_threads", 2))
    env.update({"PYTHONPATH": cfg["vipe_root"], "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "LD_LIBRARY_PATH": "", "OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "TORCH_HOME": cfg["torch_home"], "HF_HOME": cfg["hf_home"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VIPE_FRAME_CACHE_DIR": str(work / "cache"), "VIPE_TRACK_ANYTHING_AOT_ENCODER_BATCH_SIZE": str(cfg.get("aot_encoder_batch_size", 1)), "VIPE_DEPTH_ALIGNMENT_DEVICE": cfg.get("depth_alignment_device", "cpu"), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MPLCONFIGDIR": str(work / "matplotlib"), "TMPDIR": str(work / "tmp"), "QT_QPA_PLATFORM": "offscreen", "IMAGEIO_FFMPEG_EXE": cfg["ffmpeg"]})
    return env


def run_worker(identifier, request, output):
    """Run worker.py for one prepared job directory and return the ComfyUI result."""
    cfg = config()
    work = work_path(identifier)
    request_file = work / "request.json"
    request_file.write_text(json.dumps(request, indent=2), encoding="utf-8")
    model_management.unload_all_models()
    model_management.soft_empty_cache()
    save_status(work, "running", "Starting ViPE worker")
    progress = ProgressBar(100)
    process = None
    try:
        with (work / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([cfg["vipe_python"], str(Path(__file__).parent / "worker.py"), "--request", str(request_file), "--config", str(config_path())], cwd=cfg["vipe_root"], env=worker_environment(cfg, work), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                model_management.throw_exception_if_processing_interrupted()
                if shutil.disk_usage(work).free < cfg.get("min_free_disk_bytes", 2 * 1024**3):
                    raise RuntimeError("Worker stopped: local scratch has less than 2 GiB free. Free space and resubmit.")
                meminfo = Path("/proc/meminfo")
                if meminfo.exists():
                    available = next(int(x.split()[1]) for x in meminfo.read_text().splitlines() if x.startswith("MemAvailable:"))
                    if available < 1536 * 1024:
                        raise RuntimeError("Worker stopped: available system memory is below 1.5 GiB.")
                status = json.loads((work / "status.json").read_text())
                progress.update_absolute(status["progress"])
                time.sleep(0.5)
        if process.returncode:
            status = json.loads((work / "status.json").read_text())
            raise RuntimeError(status["message"] if status["state"] == "failed" else f"ViPE worker exited with code {process.returncode}; inspect the job log.")
        bundle = output / "result.zip"
        if not bundle.is_file():
            raise RuntimeError("Worker finished without a result bundle.")
        h = hashlib.sha256()
        with bundle.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        item = {"filename": "result.zip", "subfolder": f"vipe/{identifier}", "type": "output", "sha256": h.hexdigest(), "job_id": identifier}
        (work / "result_descriptor.json").write_text(json.dumps(item))
        progress.update_absolute(100)
        return {"ui": {"vipe_result": [item]}, "result": (f"vipe/{identifier}/result.zip",)}
    except model_management.InterruptProcessingException:
        save_status(work, "cancelled", "Job cancelled")
        raise
    except Exception as exc:
        save_status(work, "failed", str(exc)[-1500:])
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def new_job_directory(identifier):
    work = work_path(identifier)
    work.mkdir(parents=True, exist_ok=False)
    (work / "cache").mkdir()
    (work / "tmp").mkdir()
    return work


class ViPECameraTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("STRING", {"default": ""}), "source_name": ("STRING", {"default": "plate.mp4"}), "source_sha256": ("STRING", {"default": ""}), "request_id": ("STRING", {"default": ""}), "frame_start": ("INT", {"default": 1, "min": 1}), "frame_end": ("INT", {"default": 0, "min": 0}), "max_dimension": ("INT", {"default": 1280, "min": 0}), "masking": ("BOOLEAN", {"default": True}), "mask_phrases": ("STRING", {"default": "person,animal,vehicle,ball,balloon,gun,pet,car,bus"}), "lens_mode": (["auto", "fixed"],), "focal_mm": ("FLOAT", {"default": 35.0, "min": 0.1}), "sensor_mm": ("FLOAT", {"default": 36.0, "min": 0.1}), "max_points": ("INT", {"default": 30000, "min": 1, "max": 1000000})}, "optional": OPTIONAL_INPUTS}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_bundle",)
    FUNCTION = "track"
    OUTPUT_NODE = True
    CATEGORY = "Camera Tracking/ViPE"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def track(self, video, source_name, source_sha256, request_id, **settings):
        settings = {**OPTIONAL_DEFAULTS, **settings}
        validate_settings(settings)
        source = resolve_video(video)
        identifier = job_id(request_id)
        if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
            raise ValueError("The uploaded video needs a SHA256 checksum.")
        work = new_job_directory(identifier)
        output = Path(folder_paths.get_output_directory()) / "vipe" / identifier
        request = {"job_id": identifier, "source": str(source), "source_name": source_name, "source_sha256": source_sha256, "output": str(output), "settings": settings}
        if source.suffix == ".zip":
            if file_sha256(source) != source_sha256:
                raise ValueError("Uploaded frame archive checksum changed; upload the plate again.")
            save_status(work, "running", "Unpacking plate frames")
            movie, plate = prepare_frame_archive(source, work, config())
            # The bundle keeps the artist's archive identity; the worker verifies its derived movie.
            settings.update(plate_width=plate["width"], plate_height=plate["height"])
            request.update(source=str(movie), source_sha256=file_sha256(movie), identity_sha256=source_sha256, plate=plate)
        return run_worker(identifier, request, output)


class ViPEReviewJob:
    """Add review witnesses and/or dense geometry to an existing job without solving again."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"parent_job_id": ("STRING", {"default": ""}), "request_id": ("STRING", {"default": ""})}, "optional": {"review_witnesses": ("BOOLEAN", {"default": True}), "dense_environment": ("BOOLEAN", {"default": False}), "dense_mode": (["fast", "video"], {"default": "fast"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_bundle",)
    FUNCTION = "review"
    OUTPUT_NODE = True
    CATEGORY = "Camera Tracking/ViPE"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def review(self, parent_job_id, request_id, review_witnesses=True, dense_environment=False, dense_mode="fast"):
        parent, identifier = job_id(parent_job_id), job_id(request_id)
        parent_work = work_path(parent)
        parent_request = json.loads((parent_work / "request.json").read_text(encoding="utf-8"))
        parent_output = Path(parent_request["output"])
        manifest = json.loads((parent_output / "result.json").read_text(encoding="utf-8"))
        witnesses = review_witnesses and not manifest.get("witnesses")
        dense = dense_environment and not manifest.get("environment")
        if not (witnesses or dense):
            raise ValueError("This job already includes the requested outputs.")
        if dense and manifest["camera"].get("lens_distortion"):
            raise ValueError("Dense environment reconstruction does not support lens distortion yet.")
        if dense and not all((parent_work / name).is_file() for name in DENSE_STATE):
            raise ValueError("This job was solved before dense data was kept. Submit the plate again with Dense enabled; the upload is reused.")
        source = Path(parent_request["source"])
        if witnesses and not source.is_file():
            raise ValueError("The parent job's uploaded plate is no longer on this worker; submit the plate again.")
        work = new_job_directory(identifier)
        output = Path(folder_paths.get_output_directory()) / "vipe" / identifier
        output.mkdir(parents=True)
        for name in ["result.json", *manifest["files"]]:
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent_output / name, output / name)
        for name in ["camera_distortion.json", *(DENSE_STATE if dense else [])]:
            if (parent_work / name).is_file():
                try:
                    os.link(parent_work / name, work / name)
                except OSError:
                    shutil.copyfile(parent_work / name, work / name)
        settings = {**parent_request["settings"], "review_witnesses": witnesses, "dense_environment": dense, "dense_mode": dense_mode}
        request = {**parent_request, "job_id": identifier, "output": str(output), "settings": settings, "review_of": parent}
        return run_worker(identifier, request, output)


DENSE_STATE = ["plate.mp4", "native_slam_map.pt", "depth_context.npz", "solve_masks.zip"]


routes = PromptServer.instance.routes


@routes.get("/vipe/capabilities")
async def capabilities(request):
    cfg = config()
    return web.json_response({"protocol_version": 1, "live_logs": "byte_cursor_v1", "node": "ViPECameraTrack", "max_upload_bytes": cfg.get("max_upload_bytes", 1024**3), "outputs": ["camera", "native_cloud", "scene_markers", "independent_audit_landmarks", "artist_witnesses", "dense_environment"], "job_cancellation": "by_prompt_id", "dense_depth": True, "optional_stages": {"review_witnesses": False, "dense_environment": False}, "optional_inputs": sorted(OPTIONAL_INPUTS), "upload_reuse": "sha256_v1", "review_jobs": "ViPEReviewJob", "extend_jobs": ["review_witnesses", "dense_environment"], "solve_reuse": True, "status_detail": True, "frame_archives": "proxy_frames_v1"})


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".zip"}


def stored_upload(digest, suffix):
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or suffix not in VIDEO_SUFFIXES:
        raise web.HTTPBadRequest(text="Invalid upload identity.")
    return Path(folder_paths.get_input_directory()) / "vipe" / (digest + suffix)


@routes.get("/vipe/uploads/{digest}")
async def find_upload(request):
    # Content-addressed uploads let clients skip sending a plate the worker already has.
    path = stored_upload(request.match_info["digest"], request.query.get("suffix", ".mp4").lower())
    if not path.is_file():
        raise web.HTTPNotFound(text="Upload not found.")
    return web.json_response({"video": "vipe/" + path.name, "sha256": path.stem, "bytes": path.stat().st_size, "reused": True})


@routes.post("/vipe/upload")
async def upload_video(request):
    suffix = Path(request.query.get("filename", "")).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise web.HTTPBadRequest(text="Choose a supported video file.")
    maximum = config().get("max_upload_bytes", 1024**3)
    if request.content_length and request.content_length > maximum:
        raise web.HTTPRequestEntityTooLarge(max_size=maximum, actual_size=request.content_length)
    directory = Path(folder_paths.get_input_directory()) / "vipe"
    directory.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex + suffix
    path = directory / name
    temp = path.with_suffix(path.suffix + ".upload")
    h = hashlib.sha256()
    size = 0
    try:
        with temp.open("xb") as stream:
            async for chunk in request.content.iter_chunked(1024 * 1024):
                size += len(chunk)
                if size > maximum:
                    raise web.HTTPRequestEntityTooLarge(max_size=maximum, actual_size=size)
                if shutil.disk_usage(directory).free < 2 * 1024**3:
                    raise web.HTTPServiceUnavailable(text="Worker input disk is full.")
                h.update(chunk)
                stream.write(chunk)
        if not size:
            raise web.HTTPBadRequest(text="The video is empty.")
        path = stored_upload(h.hexdigest(), suffix)
        if path.is_file():
            temp.unlink()
        else:
            temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return web.json_response({"video": "vipe/" + path.name, "sha256": h.hexdigest(), "bytes": size})


PROGRESS = re.compile(r"([A-Za-z][\w ()/.-]{0,40}?):\s+(\d{1,3})%\|")
WINDOW = re.compile(r"CoTracker window (\d+): source frames \d+-(\d+)")
DENSE = re.compile(r"Dense view (\d+)/(\d+)")


def log_detail(path):
    """Summarize the latest progress bar from the bounded log tail."""
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - 8192))
            tail = stream.read(8192).decode("utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return ""
    found = []
    for pattern, render in [(PROGRESS, lambda m: f"{m.group(1).strip()} {m.group(2)}%"), (WINDOW, lambda m: f"CoTracker window {m.group(1)} (to frame {m.group(2)})"), (DENSE, lambda m: f"Dense view {m.group(1)}/{m.group(2)}")]:
        for match in pattern.finditer(tail):
            found.append((match.end(), render(match)))
    return max(found)[1] if found else ""


@routes.get("/vipe/jobs/{identifier}")
async def get_status(request):
    try:
        work = work_path(request.match_info["identifier"])
    except ValueError:
        raise web.HTTPBadRequest(text="Invalid job ID.")
    path = work / "status.json"
    if not path.exists():
        return web.json_response({"state": "queued", "message": "Waiting for worker", "progress": 0})
    status = json.loads(path.read_text())
    if status["state"] == "running":
        detail = log_detail(work / "worker.log")
        if detail:
            status["detail"] = detail
    descriptor = work / "result_descriptor.json"
    if status["state"] == "complete" and descriptor.exists():
        status["result"] = json.loads(descriptor.read_text())
    return web.json_response(status)


@routes.get("/vipe/jobs/{identifier}/log")
async def get_log(request):
    try:
        work = work_path(request.match_info["identifier"])
    except ValueError:
        raise web.HTTPBadRequest(text="Invalid job ID.")
    path = work / "worker.log"
    if "offset" in request.query:
        from .log_tail import read_log_chunk
        try:
            offset = int(request.query["offset"])
            chunk = read_log_chunk(path, offset, request.query.get("generation"))
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc))
        return web.json_response(chunk, headers={"Cache-Control": "no-store"})
    if not path.exists():
        return web.Response(text="The worker has not started yet.\n")
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 32768))
        text = stream.read(32768).decode("utf-8", errors="replace")
    return web.Response(text=text)


NODE_CLASS_MAPPINGS = {"ViPECameraTrack": ViPECameraTrack}
NODE_DISPLAY_NAME_MAPPINGS = {"ViPECameraTrack": "ViPE Camera + Native Points"}


class ViPEVideoInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("STRING", {"default": "", "tooltip": "Use Upload video to select a plate."}), "source_name": ("STRING", {"default": "plate.mp4"})}}

    RETURN_TYPES = ("VIPE_VIDEO",)
    RETURN_NAMES = ("plate",)
    FUNCTION = "load"
    CATEGORY = "Camera Tracking/ViPE"
    DESCRIPTION = "Upload a video and pass it to the ViPE camera solve."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load(self, video, source_name):
        source = resolve_video(video)
        return ({"video": video, "source_name": source_name or source.name, "sha256": file_sha256(source)},)


class ViPETrackVideo:
    @classmethod
    def INPUT_TYPES(cls):
        settings = ViPECameraTrack.INPUT_TYPES()["required"]
        for name in ("video", "source_name", "source_sha256", "request_id"):
            del settings[name]
        return {"required": {"plate": ("VIPE_VIDEO",), **settings}, "optional": ViPECameraTrack.INPUT_TYPES()["optional"]}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_bundle",)
    FUNCTION = "track"
    OUTPUT_NODE = True
    CATEGORY = "Camera Tracking/ViPE"
    DESCRIPTION = "Run the stock camera solve used by Blender. Every run saves a separate result."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def track(self, plate, **settings):
        identifier = str(uuid.uuid4())
        result = ViPECameraTrack().track(plate["video"], plate["source_name"], plate["sha256"], identifier, **settings)
        path = Path(folder_paths.get_output_directory()) / "vipe" / identifier / "result.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        lens = manifest["processing"]["lens_stages_original_pixels"]
        summary = f"{manifest['selection']['count']} camera frames; {manifest['cloud']['point_count']:,} native points."
        if manifest.get("witnesses"):
            summary += f"\nIndependent audit landmarks: {manifest['witnesses']['landmark_count']:,}; original artist-cross placement in Blender."
        if manifest.get("environment"):
            summary += f"\nDense environment: {manifest['environment']['point_count']:,} samples from {len(manifest['environment']['views'])} views."
        if lens.get("initial") and lens.get("post_global_ba"):
            summary += f"\nLens fx: {lens['initial'][0]:.1f} → {lens['post_global_ba'][0]:.1f} original-plate pixels."
        result["ui"]["vipe_summary"] = [summary]
        return result


NODE_CLASS_MAPPINGS.update({"ViPEVideoInput": ViPEVideoInput, "ViPETrackVideo": ViPETrackVideo, "ViPEReviewJob": ViPEReviewJob})
NODE_DISPLAY_NAME_MAPPINGS.update({"ViPEVideoInput": "ViPE · Upload Video", "ViPETrackVideo": "ViPE · Solve Camera + Native Cloud", "ViPEReviewJob": "ViPE · Add Review or Dense to Job"})
WEB_DIRECTORY = "./web"
