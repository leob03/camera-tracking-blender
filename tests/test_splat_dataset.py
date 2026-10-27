"""Coordinate, mask and geometry regression tests; no GPU or model downloads."""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comfyui_vipe_remote"))
from splat_dataset import (
    colmap_pose,
    scaled_intrinsics,
    read_masks,
    filter_seeds,
    write_camera_path,
    frame_selection,
    training_size,
)
from splat_stage import scratch_bytes, settings_checked, trainer_command


class SplatDatasetTests(unittest.TestCase):
    def test_rotated_camera_roundtrip_and_projection(self):
        # Includes 180 degrees, where trace-only quaternion formulas become singular.
        for angle in (0, 90, 180, 179.999):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler("y", angle, degrees=True).as_matrix()
            pose[:3, 3] = [2, -1, 5]
            q, t = colmap_pose(pose)
            rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            local = np.array([0.2, -0.4, 3.0])
            world = pose[:3, :3] @ local + pose[:3, 3]
            np.testing.assert_allclose(rotation @ world + t, local, atol=1e-10)
        k = scaled_intrinsics([1200, 1150, 800, 450], (1600, 900), (800, 448))
        np.testing.assert_allclose(k, [600, 1150 * 448 / 900, 400, 224])

    def test_packed_and_plain_masks_agree(self):
        masks = {
            k: np.arange(35).reshape(5, 7) % (i + 2) == 0
            for i, k in enumerate(("valid", "sky", "excluded"))
        }
        plain, packed = io.BytesIO(), io.BytesIO()
        np.savez(plain, **masks)
        np.savez(
            packed,
            shape=np.array([5, 7]),
            **{k: np.packbits(v) for k, v in masks.items()},
        )
        for a, b in zip(read_masks(plain.getvalue()), read_masks(packed.getvalue())):
            np.testing.assert_array_equal(a, b)

    def test_depth_support_rejects_free_space_and_deduplicates(self):
        yy, xx = np.mgrid[:40, :40]
        pixels = np.stack([xx.ravel(), yy.ravel()], axis=1)
        pts = np.column_stack(
            [
                (pixels[:, 0] - 20) / 20,
                (pixels[:, 1] - 20) / 20,
                np.full(len(pixels), 2),
            ]
        )
        meta = dict(
            stride=1,
            width=40,
            height=40,
            c2w=np.eye(4).tolist(),
            intrinsics=[40, 40, 20, 20],
        )
        data = dict(pixels=pixels, depths=np.full(len(pixels), 2.0))
        bad = pts[:100] * 0.5
        points = np.concatenate([pts, pts, bad, [[np.nan, 0, 0]]])
        kept, _, report = filter_seeds(
            points, np.ones_like(points), [(meta, data), (meta, data)], 2000
        )
        self.assertEqual(len(kept), 1600)
        np.testing.assert_allclose(kept[:, 2], 2)
        self.assertEqual(report["consistent_points"], 3200)
        limited, _, _ = filter_seeds(
            points, np.ones_like(points), [(meta, data), (meta, data)], 500
        )
        self.assertEqual(len(limited), 500)

    def test_render_path_preserves_camera_frame_conventions(self):
        pose = np.eye(4)
        pose[:3, 3] = [1, 2, 3]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "path.json"
            write_camera_path(
                path, [pose], [[900, 1000, 960, 540]], 1920, 1080, 24, 0.1
            )
            key = json.loads(path.read_text())["keyframes"][0]
        np.testing.assert_allclose(key["position"], [1.1, -2, -3])
        np.testing.assert_allclose(key["rotation"], [1, 0, 0, 0])
        self.assertAlmostEqual(key["focal_length_mm"], 1000 / 1080 * 24)

    def test_settings_and_offline_command(self):
        for value in (True, 0, 60001, "15000"):
            with self.assertRaises(ValueError):
                settings_checked({"splat_iterations": value})
        command = trainer_command("/opt/trainer", "data", "output", {})
        self.assertIn("--no-download", command)
        self.assertEqual(command[command.index("--centralize") + 1], "off")
        self.assertNotIn("--optimize-poses", command)

    def test_presets_scale_the_whole_schedule(self):
        quick = settings_checked({})
        self.assertEqual((quick["splat_preset"], quick["splat_iterations"], quick["splat_frame_step"]), ("quick", 10000, 1))
        quality = settings_checked({"splat_preset": "quality"})
        self.assertEqual((quality["splat_iterations"], quality["splat_max_dimension"], quality["splat_max_gaussians"]), (30000, 2048, 3000000))
        with self.assertRaises(ValueError):
            settings_checked({"splat_preset": "ultra"})
        command = trainer_command("/opt/trainer", "data", "output", {"splat_iterations": 15000})
        # The schedule stays 30k-shaped and is scaled; plain --iter would stop it halfway through densification.
        self.assertEqual(command[command.index("--iter") + 1], "30000")
        self.assertEqual(float(command[command.index("--steps-scaler") + 1]), 0.5)
        self.assertNotIn("--test-every", command)  # artist runs train on every selected frame

    def test_disk_guard_scales_with_the_job(self):
        quick, quality = (scratch_bytes(settings_checked({"splat_preset": p})) for p in ("quick", "quality"))
        self.assertLess(quick, quality)
        self.assertLess(quality, 5 * 1024**3)  # no longer a fixed 5 GiB for every job
        self.assertGreater(quality, 3 * 3000000 * 248)  # PLY, its result copy and the checkpoint

    def test_frame_selection_and_training_size(self):
        np.testing.assert_array_equal(frame_selection(10, 1), np.arange(10))
        self.assertEqual(len(frame_selection(1000, 1)), 334)  # long shots are subsampled to at most 400 views
        self.assertLessEqual(len(frame_selection(401, 1)), 400)
        # Never upscale a 1280 px solve proxy; keep the plate aspect at even sizes.
        self.assertEqual(training_size((4608, 3164), (1280, 878), 2048), [1280, 878])
        self.assertEqual(training_size((4096, 2160), (4096, 2160), 2048), [2048, 1080])


if __name__ == "__main__":
    unittest.main()
