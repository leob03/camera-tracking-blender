"""Queue a splat-training extension of an existing dense ViPE result."""

import json
import os
import uuid
from pathlib import Path, PurePosixPath


class ViPESplatJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parent_job_id": ("STRING", {"default": ""}),
                "request_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "splat_preset": (["quick", "quality"], {"default": "quick"}),
                "splat_iterations": ("INT", {"default": 10000, "min": 100, "max": 60000}),
                "splat_max_dimension": ("INT", {"default": 1536, "min": 256, "max": 4096}),
                "splat_frame_step": ("INT", {"default": 1, "min": 1, "max": 20}),
                "splat_max_gaussians": ("INT", {"default": 1000000, "min": 50000, "max": 6000000}),
                # Optional higher-resolution plate of exactly the solved frames (EXR jobs upload a 1280 px solve proxy).
                "splat_training_video": ("STRING", {"default": ""}),
                "splat_training_sha256": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_bundle",)
    FUNCTION = "train"
    OUTPUT_NODE = True
    CATEGORY = "Camera Tracking/ViPE"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def train(self, parent_job_id, request_id, splat_training_video="", splat_training_sha256="", **settings):
        from . import (
            config,
            job_id,
            work_path,
            new_job_directory,
            run_worker,
            folder_paths,
            file_sha256,
            prepare_frame_archive,
            resolve_video,
        )
        from .worker import link_or_copy

        cfg = config()
        executable = Path(cfg.get("lichtfeld_executable", ""))
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(
                "This worker needs a configured lichtfeld_executable for splat training."
            )
        parent, identifier = (
            job_id(parent_job_id),
            job_id(request_id) if request_id.strip() else str(uuid.uuid4()),
        )
        parent_work = work_path(parent)
        previous = json.loads((parent_work / "request.json").read_text())
        parent_output = Path(previous["output"])
        manifest = json.loads((parent_output / "result.json").read_text())
        if not manifest.get("environment"):
            raise ValueError(
                "Add Dense Environment to this job before training a splat."
            )
        if manifest["camera"]["model"] != "PINHOLE":
            raise ValueError("Splat training currently requires a pinhole camera.")
        source = Path(previous["source"])
        if not source.is_file():
            raise ValueError("The source plate is no longer available on the worker.")
        masks = parent_work / "solve_masks.zip"
        # Witness-only extension jobs may keep masks on their parent. Follow only explicit IDs.
        ancestor = previous
        for _ in range(8):
            if masks.is_file():
                break
            if not ancestor.get("review_of"):
                break
            ancestor_work = work_path(ancestor["review_of"])
            masks = ancestor_work / "solve_masks.zip"
            ancestor = json.loads((ancestor_work / "request.json").read_text())
        if not masks.is_file():
            raise ValueError(
                "Retained solve masks are missing; resubmit the camera solve."
            )
        output = Path(folder_paths.get_output_directory()) / "vipe" / identifier
        work = new_job_directory(identifier)
        output.mkdir(parents=True, exist_ok=False)
        for name in ["result.json", *manifest["files"]]:
            if name.startswith("splat/"):
                continue
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid parent result filename.")
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            # JSON is rewritten by this job; never hardlink mutable metadata.
            if name == "result.json":
                destination.write_text((parent_output / name).read_text())
            else:
                link_or_copy(parent_output / name, destination)
        manifest.pop("splat", None)
        manifest["files"] = {
            k: v for k, v in manifest["files"].items() if not k.startswith("splat/")
        }
        manifest["version"] = 3
        (output / "result.json").write_text(json.dumps(manifest, indent=2))
        link_or_copy(masks, work / "solve_masks.zip")
        training = {}
        if splat_training_video.strip():
            video = resolve_video(splat_training_video)
            if file_sha256(video) != splat_training_sha256:
                raise ValueError("Uploaded training plate checksum changed; upload it again.")
            if video.suffix == ".zip":
                (work / "training").mkdir()
                video, plate = prepare_frame_archive(video, work / "training", config())
                if plate.get("frame_count", manifest["source"]["frame_count"]) != manifest["source"]["frame_count"]:
                    raise ValueError("The training plate has a different frame count from the solved plate.")
            training = {"training_source": str(video), "training_sha256": file_sha256(video)}
        request = {
            **previous,
            "job_id": identifier,
            "output": str(output),
            "review_of": parent,
            "splat_of": parent,
            **training,
            "settings": {
                **previous["settings"],
                **settings,
                "review_witnesses": False,
                "dense_environment": False,
                "train_splat": True,
            },
        }
        return run_worker(identifier, request, output)
