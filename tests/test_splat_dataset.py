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
)
from splat_stage import settings_checked, trainer_command


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


if __name__ == "__main__":
    unittest.main()
