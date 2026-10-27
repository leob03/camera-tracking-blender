"""Small HTTP client; contains no Blender or GPU imports."""
import hashlib
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlencode, urlsplit
import uuid

from .bundle import SPLAT_LIMIT


class APIError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


class JobCancelled(RuntimeError):
    pass


class Detached(RuntimeError):
    pass


class Client:
    def __init__(self, base_url, token="", timeout=30):
        parsed = urlsplit(base_url.rstrip("/"))
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
            raise ValueError("Worker URL must be an http(s) address, without credentials or query parameters.")
        self.base_url = base_url.rstrip("/")
        self.scheme, self.host, self.port = parsed.scheme, parsed.hostname, parsed.port
        self.prefix = parsed.path.rstrip("/")
        self.token = token
        self.timeout = timeout

    def connection(self):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout)

    def headers(self):
        return {"Authorization": "Bearer " + self.token} if self.token else {}

    def request(self, method, path, data=None):
        body = json.dumps(data).encode("utf-8") if data is not None else None
        headers = self.headers()
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection = self.connection()
        try:
            connection.request(method, self.prefix + path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise APIError(response.status, "Worker JSON response exceeded 4 MiB.")
            if response.status >= 300:
                raise APIError(response.status, f"Worker HTTP {response.status}: {raw.decode('utf-8', errors='replace')[:1200]}")
            return json.loads(raw)
        finally:
            connection.close()

    def capabilities(self):
        result = self.request("GET", "/vipe/capabilities")
        if result.get("protocol_version") != 1:
            raise ValueError("This worker does not support ViPE bundle protocol 1.")
        return result

    def upload(self, path, progress=lambda *args: None, stop=lambda: False):
        path = Path(path)
        size = path.stat().st_size
        caps = self.capabilities()
        self.caps = caps
        if size <= 0 or size > caps["max_upload_bytes"]:
            raise ValueError(f"Video must be nonempty and smaller than {caps['max_upload_bytes'] / 1024**2:.0f} MiB.")
        if caps.get("upload_reuse") == "sha256_v1":
            # Hashing locally is much cheaper than sending a plate the worker already has.
            local = hashlib.sha256()
            done = 0
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    if stop():
                        raise JobCancelled("Upload stopped before job submission.")
                    local.update(chunk)
                    done += len(chunk)
                    progress("Checking plate on worker", done / size)
            try:
                found = self.request("GET", f"/vipe/uploads/{local.hexdigest()}?" + urlencode({"suffix": path.suffix.lower()}))
                if found.get("sha256") == local.hexdigest() and found.get("bytes") == size:
                    progress("Plate already on worker", 1)
                    return {**found, "reused": True}
            except APIError as exc:
                if exc.status != 404:
                    raise
        connection = self.connection()
        digest = hashlib.sha256()
        sent = 0
        try:
            connection.putrequest("POST", self.prefix + "/vipe/upload?" + urlencode({"filename": path.name}))
            for key, value in self.headers().items():
                connection.putheader(key, value)
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("Content-Length", str(size))
            connection.endheaders()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    if stop():
                        raise JobCancelled("Upload stopped before job submission.")
                    connection.send(chunk)
                    digest.update(chunk)
                    sent += len(chunk)
                    progress("Uploading video", sent / size)
            response = connection.getresponse()
            raw = response.read(65536)
            if response.status >= 300:
                raise APIError(response.status, raw.decode("utf-8", errors="replace")[:1200])
            result = json.loads(raw)
            if result["sha256"] != digest.hexdigest() or result["bytes"] != size:
                raise ValueError("Uploaded video checksum does not match the local plate.")
            return {**result, "reused": False}
        finally:
            connection.close()

    def submit(self, uploaded, source_name, settings, identifier):
        identifier = str(uuid.UUID(identifier))
        caps = getattr(self, "caps", None) or self.capabilities()
        # Older workers only know the original optional stages; never send them unknown inputs.
        known = set(caps.get("optional_inputs", ["review_witnesses", "dense_environment"]))
        required = {"frame_start", "frame_end", "max_dimension", "masking", "mask_phrases", "lens_mode", "focal_mm", "sensor_mm", "max_points"}
        unsupported = [k for k, v in settings.items() if k not in required and k not in known and v]
        if unsupported:
            raise ValueError("This worker does not support: " + ", ".join(sorted(unsupported)) + ". Update the ComfyUI node pack.")
        settings = {k: v for k, v in settings.items() if k in required or k in known}
        inputs = {**settings, "video": uploaded["video"], "source_name": source_name, "source_sha256": uploaded["sha256"], "request_id": identifier}
        workflow = {"1": {"class_type": "ViPECameraTrack", "inputs": inputs, "_meta": {"title": "ViPE camera and native points"}}}
        response = self.request("POST", "/prompt", {"prompt": workflow, "prompt_id": identifier, "client_id": "blender-vipe-" + identifier})
        if response.get("prompt_id") != identifier or response.get("node_errors"):
            raise ValueError(f"Worker rejected the camera workflow: {response}")
        return workflow

    def submit_review(self, parent_job_id, identifier, review_witnesses=True, dense_environment=False, dense_mode="fast"):
        """Add review witnesses and/or dense geometry to a finished job, reusing its camera solve."""
        identifier = str(uuid.UUID(identifier))
        caps = self.capabilities()
        if caps.get("review_jobs") != "ViPEReviewJob":
            raise ValueError("This worker cannot extend an existing job. Update the ComfyUI node pack.")
        inputs = {"parent_job_id": str(uuid.UUID(parent_job_id)), "request_id": identifier}
        if dense_environment:
            if "dense_environment" not in caps.get("extend_jobs", []):
                raise ValueError("This worker cannot add dense geometry to an existing job. Update the ComfyUI node pack.")
            inputs.update(review_witnesses=bool(review_witnesses), dense_environment=True, dense_mode=dense_mode)
        workflow = {"1": {"class_type": "ViPEReviewJob", "inputs": inputs, "_meta": {"title": "ViPE job extension"}}}
        response = self.request("POST", "/prompt", {"prompt": workflow, "prompt_id": identifier, "client_id": "blender-vipe-" + identifier})
        if response.get("prompt_id") != identifier or response.get("node_errors"):
            raise ValueError(f"Worker rejected the review workflow: {response}")
        return workflow

    def submit_splat(self, parent_job_id, identifier, **settings):
        identifier = str(uuid.UUID(identifier))
        if self.capabilities().get('splat_jobs') != 'ViPESplatJob':
            raise ValueError('This worker has no Gaussian trainer configured. Update the node pack and configure LichtFeld.')
        inputs = {'parent_job_id': str(uuid.UUID(parent_job_id)), 'request_id': identifier, **settings}
        workflow = {'1': {'class_type': 'ViPESplatJob', 'inputs': inputs}}
        response = self.request('POST', '/prompt', {'prompt': workflow, 'prompt_id': identifier, 'client_id': 'blender-vipe-' + identifier})
        if response.get('prompt_id') != identifier or response.get('node_errors'):
            raise ValueError(f'Worker rejected the splat workflow: {response}')
        return workflow

    def cancel(self, identifier):
        return self.request("POST", f"/api/jobs/{str(uuid.UUID(identifier))}/cancel", {})

    def wait(self, identifier, progress=lambda *args: None, cancel=lambda: False, detach=lambda: False, interval=2, log=None):
        from .job_logs import LogFollower
        follower = LogFollower(self, identifier, log) if log is not None else None
        try:
            return self._wait(identifier, progress, cancel, detach, min(interval, 1) if follower else interval, follower)
        finally:
            if follower is not None:
                follower.finish()

    def _wait(self, identifier, progress, cancel, detach, interval, follower):
        identifier = str(uuid.UUID(identifier))
        cancellation_sent = False
        while True:
            if detach():
                raise Detached("Monitoring stopped; the worker can continue. Use Resume Job later.")
            if cancel() and not cancellation_sent:
                response = self.cancel(identifier)
                cancellation_sent = True
                if response.get("cancelled"):
                    progress("Cancelling job", 0)
            if follower is not None:
                follower.poll()
            history = self.request("GET", f"/history/{identifier}").get(identifier)
            if history:
                for output in history.get("outputs", {}).values():
                    if output.get("vipe_result"):
                        item = output["vipe_result"][0]
                        if item.get("job_id") != identifier:
                            raise ValueError("Worker returned another job's result.")
                        return item
                state = history.get("status", {})
                if state.get("completed") or state.get("status_str") in ("error", "success"):
                    messages = state.get("messages", [])
                    for kind, data in reversed(messages):
                        if kind == "execution_interrupted":
                            raise JobCancelled("Job cancelled.")
                        if kind == "execution_error":
                            raise RuntimeError(data.get("exception_message", "ViPE worker failed."))
                    raise RuntimeError("Job finished without a camera result.")
            status = self.request("GET", f"/vipe/jobs/{identifier}")
            if status["state"] == "cancelled":
                raise JobCancelled(status["message"])
            if status["state"] == "failed":
                raise RuntimeError(status["message"])
            if status.get("result"):
                return status["result"]
            if status["state"] == "queued":
                try:
                    self.request("GET", f"/api/jobs/{identifier}")
                except APIError as exc:
                    if exc.status == 404:
                        if cancellation_sent:
                            raise JobCancelled("Queued job cancelled.")
                        raise RuntimeError("This worker no longer knows the job. Check the worker address or submit a new job.") from exc
                    raise
            progress(status["message"] + (" · " + status["detail"] if status.get("detail") else ""), status["progress"] / 100)
            deadline = time.monotonic() + interval
            while time.monotonic() < deadline:
                if detach():
                    raise Detached("Monitoring stopped; the worker can continue.")
                time.sleep(0.1)

    def download(self, item, destination, progress=lambda *args: None, stop=lambda: False):
        if item.get("type") != "output" or item.get("filename") != "result.zip":
            raise ValueError("Worker returned an unexpected artifact.")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".zip.part")
        connection = self.connection()
        digest = hashlib.sha256()
        try:
            query = urlencode({key: item[key] for key in ("filename", "subfolder", "type")})
            connection.request("GET", self.prefix + "/view?" + query, headers=self.headers())
            response = connection.getresponse()
            if response.status != 200:
                raise APIError(response.status, "Could not download the camera result.")
            total = int(response.getheader("Content-Length", "0"))
            received = 0
            with temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    if stop():
                        raise Detached("Download stopped; use Resume Job to retrieve it later.")
                    received += len(chunk)
                    if received > 512 * 1024**2 + SPLAT_LIMIT:
                        raise ValueError("The result exceeds the client download limit.")
                    stream.write(chunk)
                    digest.update(chunk)
                    progress("Downloading result", received / total if total else 0)
            if total and received != total:
                raise ValueError("Result download was incomplete.")
            if digest.hexdigest() != item["sha256"]:
                raise ValueError("Result checksum failed; use Resume Job to download it again.")
            temporary.replace(destination)
            return destination
        finally:
            connection.close()
            temporary.unlink(missing_ok=True)

    def log(self, identifier):
        connection = self.connection()
        try:
            connection.request("GET", self.prefix + f"/vipe/jobs/{str(uuid.UUID(identifier))}/log", headers=self.headers())
            response = connection.getresponse()
            if response.status != 200:
                raise APIError(response.status, "Could not read the job log.")
            return response.read(65536).decode("utf-8", errors="replace")
        finally:
            connection.close()

    def log_chunk(self, identifier, offset=-1, generation=None):
        query = {"offset": offset}
        if generation is not None:
            query["generation"] = generation
        # A slow log endpoint must not hold up cancellation for the normal 30 s timeout.
        log_client = Client(self.base_url, self.token, timeout=min(self.timeout, 2))
        try:
            return log_client.request("GET", f"/vipe/jobs/{str(uuid.UUID(identifier))}/log?" + urlencode(query))
        except json.JSONDecodeError as exc:
            raise ValueError("Update and restart the ViPE worker for live logs; Read Job Log still provides snapshots") from exc
