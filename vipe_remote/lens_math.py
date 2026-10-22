"""Lens and review-error math shared by Blender import and command-line checks; NumPy only."""
import numpy as np


def polynomial_from_mei(lens, width, height):
    """Least-squares Blender/libmv polynomial (k1, k2, k3) matching the unified model on this frame."""
    xi = lens['xi']
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float64)
    mx, my = (corners[:, 0] - lens['cx']) / lens['fx'], (corners[:, 1] - lens['cy']) / lens['fy']
    r2 = mx * mx + my * my
    factor = (xi + np.sqrt(np.maximum(1 + (1 - xi * xi) * r2, 0))) / (1 + r2)
    reach = float(np.max(np.sqrt(r2) * factor / np.maximum(factor - xi, 1e-9)))
    r = np.linspace(0, reach * 1.05, 256)
    ratio = (1 + xi) / (1 + xi * np.sqrt(1 + r * r)) * lens['fx'] / ((1 + xi) * lens['undistorted_focal_px'])
    basis = np.stack((r**2, r**4, r**6), axis=1)
    k, *_ = np.linalg.lstsq(basis, ratio - 1, rcond=None)
    fit_px = float(np.max(np.abs(basis @ k - (ratio - 1)) * r) * lens['undistorted_focal_px'])
    return [float(v) for v in k], fit_px


def review_errors(camera, audit, first_source_frame, held_out=True):
    """Per-observation reprojection residuals (plate pixels) of the independent landmarks.

    By default only held-out validation observations (role 1) are used: each landmark is
    triangulated from the other half, so these residuals are not fitted to themselves.
    """
    counts = np.diff(audit['observation_offsets'])
    xyz = np.repeat(audit['landmark_xyz'], counts, axis=0)
    offsets = audit['observation_frames'] - first_source_frame
    if held_out and 'observation_roles' in audit:
        keep = audit['observation_roles'] == 1
        xyz, offsets = xyz[keep], offsets[keep]
        audit = dict(audit, observed_xy=audit['observed_xy'][keep])
    poses = camera['c2w'][offsets]
    k = camera['intrinsics'][offsets]
    local = np.einsum('nji,nj->ni', poses[:, :3, :3], xyz - poses[:, :3, 3])
    valid = local[:, 2] > 1e-9
    uv = np.stack((k[:, 0] * local[:, 0] / np.where(valid, local[:, 2], 1) + k[:, 2], k[:, 1] * local[:, 1] / np.where(valid, local[:, 2], 1) + k[:, 3]), axis=1)
    errors = np.linalg.norm(uv - audit['observed_xy'], axis=1)
    return offsets[valid], errors[valid]
