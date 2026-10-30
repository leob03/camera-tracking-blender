"""Data transfer contract: workflow identity, fixed filenames and corrupt-file rejection."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vipe_remote.client import Client
from vipe_remote.splat import SPLAT_FILES, materialize_splat
from vipe_remote.bundle import load_bundle


class TransferTests(unittest.TestCase):
    def test_workflow_identity_and_capability(self):
        client = Client("http://127.0.0.1:1")
        identifier = "00000000-0000-4000-8000-000000000001"
        parent = "00000000-0000-4000-8000-000000000002"
        calls = []
        client.capabilities = lambda: dict(splat_jobs="ViPESplatJob")

        def request(method, path, body):
            calls.append((method, path, body))
            return {"prompt_id": identifier}

        client.request = request
        workflow = client.submit_splat(parent, identifier, splat_iterations=100)
        self.assertEqual(workflow["1"]["inputs"]["parent_job_id"], parent)
        self.assertEqual(calls[0][2]["prompt_id"], identifier)
        self.assertEqual(calls[0][2]["prompt"]["1"]["inputs"]["request_id"], identifier)
        client.capabilities = lambda: {}
        with self.assertRaisesRegex(ValueError, "no Gaussian trainer"):
            client.submit_splat(parent, identifier)

    def test_materialize_checks_hash_and_cleans_partial_file(self):
        raw = b"Gaussian output fixture"
        manifest = {
            "files": {
                name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                for name in SPLAT_FILES
            }
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.zip"
            with zipfile.ZipFile(path, "w") as z:
                for name in SPLAT_FILES:
                    z.writestr(name, raw)
            folder = materialize_splat(path, manifest)
            self.assertEqual((folder / "scene.ply").read_bytes(), raw)
            manifest["files"]["splat/scene.ply"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Corrupt splat"):
                materialize_splat(path, manifest)
            self.assertFalse((folder / "scene.ply.part").exists())

    def test_bundle_rejects_path_escape_and_changed_camera_before_arrays(self):
        files = {
            "camera.npz": {"sha256": "a" * 64, "bytes": 0},
            "cloud.npz": {"sha256": "b" * 64, "bytes": 0},
        }
        manifest = dict(
            schema="vipe-camera-bundle",
            version=4,
            camera=dict(convention="opencv_camera_to_world", model="PINHOLE"),
            selection=dict(count=12),
            environment=dict(
                views=[
                    dict(
                        frame_index=0,
                        file="environment/view_000000.npz",
                        texture="environment/plate_000000.png",
                        width=1,
                        height=1,
                    )
                ]
            ),
            files=files,
        )
        splat = dict(
            schema="vipe-gaussian-splat",
            version=1,
            file="splat/scene.ply",
            previews=["splat/camera_preview.mp4", "splat/nearby_preview.mp4"],
            mask_preview="splat/masks_preview.jpg",
            dataset_report="splat/dataset.json",
            camera_sha256="a" * 64,
            camera_unchanged=True,
        )
        for changes in (
            {"file": "../escape.ply"},
            {"camera_sha256": "c" * 64},
            {"camera_unchanged": False},
        ):
            manifest["splat"] = {**splat, **changes}
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "bad.zip"
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("result.json", json.dumps(manifest))
                    z.writestr("camera.npz", b"")
                    z.writestr("cloud.npz", b"")
                with self.assertRaisesRegex(ValueError, "Invalid Gaussian splat"):
                    load_bundle(path)


if __name__ == "__main__":
    unittest.main()
