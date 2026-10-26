"""Materialize only the declared Gaussian outputs; never execute downloaded content."""

import hashlib
from pathlib import Path
import zipfile

SPLAT_FILES = {
    "splat/scene.ply",
    "splat/camera_preview.mp4",
    "splat/nearby_preview.mp4",
    "splat/masks_preview.jpg",
    "splat/dataset.json",
}


def materialize_splat(bundle, manifest):
    destination = Path(bundle).parent / "splat"
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(bundle) as z:
        for name in sorted(SPLAT_FILES):
            info = manifest["files"][name]
            target = destination / Path(name).name
            temporary = target.with_suffix(target.suffix + ".part")
            h = hashlib.sha256()
            count = 0
            try:
                with z.open(name) as src, temporary.open("wb") as dst:
                    for data in iter(lambda: src.read(1024 * 1024), b""):
                        count += len(data)
                        if count > info["bytes"]:
                            raise ValueError("Splat file exceeds declared size.")
                        dst.write(data)
                        h.update(data)
                if count != info["bytes"] or h.hexdigest() != info["sha256"]:
                    raise ValueError("Corrupt splat output: " + name)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


def check_splat_camera(collection, manifest):
    if collection.get("vipe_source_sha256") != manifest["source"]["sha256"]:
        raise ValueError("The splat belongs to a different plate.")
    import bpy
    import json

    report = bpy.data.texts.get(collection.get("vipe_manifest_text", ""))
    if report is not None:
        original = json.loads(report.as_string())
        if (
            original["files"]["camera.npz"]["sha256"]
            != manifest["splat"]["camera_sha256"]
        ):
            raise ValueError("The splat uses a different camera solve from this scene.")


def attach_splat(collection, bundle, manifest):
    check_splat_camera(collection, manifest)
    folder = materialize_splat(bundle, manifest)
    collection["vipe_splat_directory"] = str(folder.resolve())
    collection["vipe_splat_ply"] = str((folder / "scene.ply").resolve())
    collection["vipe_splat_job_id"] = manifest["job_id"]
    collection["vipe_splat_bundle"] = str(Path(bundle).resolve())
    collection["vipe_splat_coordinates"] = manifest["splat"]["coordinates"]
