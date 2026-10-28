"""Viewport splat data: PLY decoding, covariance, SH layout, draw order and display subset; numpy only."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location("splat_data", Path(__file__).resolve().parents[1] / "vipe_remote" / "splat_data.py")
splat_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splat_data)

NAMES = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", *[f"f_rest_{i}" for i in range(45)],
         "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def write_ply(path, rows):
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(rows)}", *[f"property float {n}" for n in NAMES], "end_header"]
    with open(path, "wb") as stream:
        stream.write(("\n".join(header) + "\n").encode())
        stream.write(np.asarray(rows, "<f4").tobytes())


class SplatViewDataTests(unittest.TestCase):
    def test_covariance_colour_and_padding(self):
        rng = np.random.default_rng(3)
        rows = rng.normal(size=(5, len(NAMES))).astype(np.float32)
        column = {n: i for i, n in enumerate(NAMES)}
        rows[:, column["f_rest_0"]:column["f_rest_44"] + 1] = np.arange(45)  # channel-major: R 0-14, G 15-29, B 30-44
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.ply"
            write_ply(path, rows)
            data = splat_data.load_gaussians(path, pad=8)
        self.assertEqual((data["count"], data["sh_degree"], data["base"].shape, data["sh"].shape), (5, 3, (8, 3, 4), (8, 12, 4)))
        for i, row in enumerate(rows):
            q = row[[column["rot_1"], column["rot_2"], column["rot_3"], column["rot_0"]]]  # scipy wants (x, y, z, w)
            m = Rotation.from_quat(q / np.linalg.norm(q)).as_matrix() * np.exp(row[[column["scale_0"], column["scale_1"], column["scale_2"]]])
            cov = m @ m.T
            texels = data["base"][i]
            np.testing.assert_allclose(texels[0], [*row[:3], 1 / (1 + np.exp(-row[column["opacity"]]))], rtol=1e-5)
            np.testing.assert_allclose([*texels[1], *texels[2][:2]], [cov[0, 0], cov[0, 1], cov[0, 2], cov[1, 1], cov[1, 2], cov[2, 2]], rtol=1e-4, atol=1e-6)
        sh = data["sh"][0].reshape(-1)
        np.testing.assert_allclose(sh[0:3], rows[0, [column["f_dc_0"], column["f_dc_1"], column["f_dc_2"]]])
        np.testing.assert_array_equal(sh[3:9], [0, 15, 30, 1, 16, 31])  # coefficient 1 RGB, then coefficient 2 RGB
        self.assertTrue((data["base"][5:] == 0).all())

    def test_draw_order_is_back_to_front(self):
        points = np.array([[0, 0, -1], [0, 0, -9], [0, 0, -4]], np.float32)  # in front of a camera looking down -Z
        order = splat_data.back_to_front(points, np.array([10, 11, 12]), (0, 0, 1, 0))
        np.testing.assert_array_equal(order, [11, 12, 10])

    def test_display_subset_keeps_the_most_visible(self):
        importance = np.array([0.1, 0.9, 0.5, 0.05, 0.7], np.float32)
        np.testing.assert_array_equal(splat_data.display_subset(importance, 60), [1, 2, 4])
        self.assertEqual(len(splat_data.display_subset(importance, 100)), 5)


if __name__ == "__main__":
    unittest.main()
