"""CPU checks for lens distortion and review-error math: python tests/test_lens_math.py (NumPy required)."""
import ast
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vipe_remote.lens_math import polynomial_from_mei, review_errors


def evidence_function(name):
    # evidence_stage imports worker-only modules; load just the pure function.
    tree = ast.parse((ROOT / 'comfyui_vipe_remote/evidence_stage.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'np': np}
    exec(compile(ast.Module([node], []), name, 'exec'), scope)
    return scope[name]


def mei_project(points, lens):
    r = np.linalg.norm(points, axis=1)
    d = points[:, 2] + lens['xi'] * r
    return np.stack((lens['fx'] * points[:, 0] / d + lens['cx'], lens['fy'] * points[:, 1] / d + lens['cy']), axis=1)


def test_undistort_matches_pinhole():
    undistort = evidence_function('undistort_pixels')
    rng = np.random.default_rng(1)
    for xi in (0.0, 0.05, 0.3):
        lens = dict(fx=3000., fy=3000., cx=2021., cy=1388., xi=xi, undistorted_focal_px=3000. / (1 + xi))
        points = np.column_stack((rng.uniform(-1.5, 1.5, 500), rng.uniform(-1, 1, 500), rng.uniform(2, 20, 500)))
        pinhole = np.stack((lens['undistorted_focal_px'] * points[:, 0] / points[:, 2] + lens['cx'], lens['undistorted_focal_px'] * points[:, 1] / points[:, 2] + lens['cy']), axis=1)
        error = np.abs(undistort(mei_project(points, lens), lens) - pinhole).max()
        assert error < 1e-6, (xi, error)


def test_polynomial_fit():
    for xi, limit in ((0.0, 1e-9), (0.05, 0.05), (0.1, 0.1)):
        lens = dict(fx=3000., fy=3000., cx=2021., cy=1388., xi=xi, undistorted_focal_px=3000. / (1 + xi))
        k, fit = polynomial_from_mei(lens, 4042, 2776)
        assert fit < limit, (xi, fit, k)
        if xi:
            assert abs(k[0] + xi / (2 * (1 + xi))) < 0.1 * xi, k


def test_review_errors():
    rng = np.random.default_rng(2)
    n = 30
    c2w = np.repeat(np.eye(4)[None], n, axis=0)
    c2w[:, 0, 3] = np.linspace(0, 1, n)
    intrinsics = np.repeat([[1000., 1000., 640., 360.]], n, axis=0)
    xyz = np.column_stack((rng.uniform(-2, 3, 40), rng.uniform(-1, 1, 40), rng.uniform(4, 10, 40)))
    frames, observed, offsets = [], [], [0]
    for point in xyz:
        for f in range(5, 15):
            local = point - c2w[f, :3, 3]
            frames.append(100 + f)
            observed.append([1000 * local[0] / local[2] + 640 + 0.5, 1000 * local[1] / local[2] + 360])
        offsets.append(len(frames))
    roles = np.tile([0, 1], len(frames) // 2)
    audit = dict(landmark_xyz=xyz, observation_offsets=np.array(offsets), observation_frames=np.array(frames), observed_xy=np.array(observed), observation_roles=roles)
    got_frames, errors = review_errors(dict(c2w=c2w, intrinsics=intrinsics), audit, 100, held_out=False)
    assert len(errors) == len(frames) and np.allclose(errors, 0.5) and got_frames.min() == 5 and got_frames.max() == 14
    held_frames, held = review_errors(dict(c2w=c2w, intrinsics=intrinsics), audit, 100)
    assert len(held) == len(frames) // 2 and np.allclose(held, 0.5) and np.all(np.isin(held_frames, [6, 8, 10, 12, 14]))


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('PASS', name)
