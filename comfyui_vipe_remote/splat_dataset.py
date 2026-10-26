"""Export frozen ViPE cameras and static depth samples for LichtFeld training.

COLMAP is a file format here; no structure-from-motion or camera refinement runs.
"""

import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read_masks(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as d:
        if "shape" not in d.files:
            return tuple(d[k] for k in ("valid", "sky", "excluded"))
        shape = tuple(d["shape"])
        size = int(np.prod(shape))
        return tuple(
            np.unpackbits(d[k])[:size].reshape(shape).astype(bool)
            for k in ("valid", "sky", "excluded")
        )


def scaled_intrinsics(k, old_size, new_size):
    sx, sy = np.asarray(new_size) / np.asarray(old_size)
    return np.asarray(k) * [sx, sy, sx, sy]


def colmap_pose(c2w):
    w2c = np.linalg.inv(c2w)
    q = Rotation.from_matrix(w2c[:3, :3]).as_quat()  # xyzw -> wxyz
    return q[[3, 0, 1, 2]], w2c[:3, 3]


def write_camera_path(path, poses, intrinsics, width, height, fps, offset=0.0):
    # LichtFeld's COLMAP importer flips world Y/Z. Its path cameras use OpenGL axes.
    flip = np.diag([1.0, -1.0, -1.0])
    keys = []
    for i, (pose, k) in enumerate(zip(poses, intrinsics)):
        r = flip @ pose[:3, :3] @ flip
        q = Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]
        position = pose[:3, 3] + pose[:3, 0] * offset
        keys.append(
            dict(
                time=i / fps,
                position=(flip @ position).tolist(),
                rotation=q.tolist(),
                focal_length_mm=float(k[1] / height * 24),
                easing=0,
            )
        )
    Path(path).write_text(
        json.dumps(dict(version=1, clip_duration=len(keys) / fps, keyframes=keys))
    )


def filter_seeds(points, colors, views, max_points):
    """Reject unsupported/free-space samples and voxel-deduplicate in the ViPE world."""
    finite = np.isfinite(points).all(1) & np.isfinite(colors).all(1)
    points, colors = points[finite], colors[finite]
    support = np.zeros(len(points), np.uint16)
    contradict = np.zeros(len(points), np.uint16)
    for meta, a in views:
        stride = meta["stride"]
        grid = np.zeros(
            (
                (meta["height"] + stride - 1) // stride,
                (meta["width"] + stride - 1) // stride,
            ),
            np.float32,
        )
        pixels = a["pixels"]
        grid[pixels[:, 1] // stride, pixels[:, 0] // stride] = a["depths"]
        c = np.asarray(meta["c2w"])
        k = np.asarray(meta["intrinsics"])
        local = (points - c[:3, 3]) @ c[:3, :3]
        z = local[:, 2]
        uv = local[:, :2] / np.maximum(z[:, None], 1e-8) * k[:2] + k[2:]
        ij = np.rint(uv / stride).astype(np.int64)
        inside = (
            (z > 0)
            & (ij[:, 0] >= 0)
            & (ij[:, 0] < grid.shape[1])
            & (ij[:, 1] >= 0)
            & (ij[:, 1] < grid.shape[0])
        )
        ids = np.flatnonzero(inside)
        ref = grid[ij[ids, 1], ij[ids, 0]]
        known = ref > 0
        ids, ref = ids[known], ref[known]
        support[ids] += np.abs(z[ids] - ref) <= 0.06 * ref
        # Samples behind another observed surface are occluded, not contradicted.
        contradict[ids] += z[ids] < 0.90 * ref
    keep = (support >= min(2, len(views))) & (contradict <= support)
    if keep.sum() < 1000:
        raise ValueError(
            "Too few consistent static depth samples for splat initialization; inspect masks and depth."
        )
    points, colors = points[keep], np.clip(colors[keep], 0, 1)
    depth_scale = float(np.median(np.concatenate([a["depths"] for _, a in views])))
    voxel = max(depth_scale * 0.002, 1e-6)
    _, ids = np.unique(
        np.floor(points / voxel).astype(np.int64), axis=0, return_index=True
    )
    ids.sort()
    if len(ids) > max_points:
        ids = ids[np.linspace(0, len(ids) - 1, max_points, dtype=int)]
    return (
        points[ids],
        colors[ids],
        dict(
            input_points=int(len(finite)),
            consistent_points=int(keep.sum()),
            seed_points=len(ids),
            voxel_size=voxel,
            depth_scale=depth_scale,
            min_depth_support=min(2, len(views)),
        ),
    )


def prepare_dataset(result_dir, source, masks_path, out, settings, progress=print):
    result_dir, source, masks_path, out = map(
        Path, (result_dir, source, masks_path, out)
    )
    m = json.loads((result_dir / "result.json").read_text())
    if m["camera"]["model"] != "PINHOLE" or m["camera"].get("lens_distortion"):
        raise ValueError(
            "Splat training currently requires a pinhole solve. Undistortion support is not implemented."
        )
    views = m.get("environment", {}).get("views", [])
    if not views:
        raise ValueError("Add Dense Environment before training a splat.")
    if not masks_path.is_file():
        raise ValueError(
            "Retained solve masks are missing; resubmit the camera job with masking enabled."
        )
    if not m.get("settings", {}).get("masking", False):
        raise ValueError(
            "Static environment training requires a solve with dynamic-category masking enabled."
        )
    with np.load(result_dir / "camera.npz", allow_pickle=False) as d:
        poses = d["c2w"]
        ks = d["intrinsics"]
        source_indices = d["source_frame_indices"]
    if not np.isfinite(poses).all() or not np.isfinite(ks).all():
        raise ValueError("Non-finite camera data.")
    n = len(poses)
    step = settings.get("splat_frame_step", 3)
    selected = np.arange(0, n, step, dtype=int)
    if len(selected) < 12:
        raise ValueError(
            "Splat training needs at least 12 selected views; reduce the frame step."
        )
    original = (m["source"]["width"], m["source"]["height"])
    scale = min(1.0, settings.get("splat_max_dimension", 1536) / max(original))
    width, height = [max(2, round(x * scale / 2) * 2) for x in original]
    out.mkdir(parents=True, exist_ok=False)
    for name in ["images", "masks", "sparse/0"]:
        (out / name).mkdir(parents=True)
    cap = cv2.VideoCapture(str(source))
    wanted = {int(source_indices[i]): int(i) for i in selected}
    if not cap.isOpened():
        raise ValueError("Cannot open the source movie for splat training.")
    cam_lines = []
    image_lines = []
    preview = []
    mask_fractions = []
    seen = []
    try:
        with zipfile.ZipFile(masks_path) as masks:
            for source_index in range(int(source_indices[-1]) + 1):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(
                        "Source movie ended before the solved frame selection."
                    )
                if source_index not in wanted:
                    continue
                i = wanted[source_index]
                j = len(seen) + 1
                name = f"{j:06d}.jpg"
                rgb = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
                valid, sky, excluded = read_masks(masks.read(f"{i:06d}.npz"))
                # Retain sky for appearance; dynamic objects are excluded, never painted black.
                moving = (excluded & ~sky).astype(np.uint8)
                moving = cv2.resize(
                    moving, (width, height), interpolation=cv2.INTER_NEAREST
                )
                margin = max(3, round(width / 200) | 1)
                moving = cv2.dilate(moving, np.ones((margin, margin), np.uint8)).astype(
                    bool
                )
                keep = ~moving
                mask_fractions.append(float(moving.mean()))
                if keep.mean() < 0.1:
                    raise ValueError(f"Frame {i} has less than 10% static support.")
                assert cv2.imwrite(
                    str(out / "images" / name), rgb, [cv2.IMWRITE_JPEG_QUALITY, 96]
                )
                assert cv2.imwrite(
                    str(out / "masks" / f"{j:06d}.png"), keep.astype(np.uint8) * 255
                )
                k = scaled_intrinsics(ks[i], original, (width, height))
                q, t = colmap_pose(poses[i])
                cam_lines.append(
                    f"{j} PINHOLE {width} {height} " + " ".join(f"{x:.12g}" for x in k)
                )
                image_lines.extend(
                    [
                        f"{j} "
                        + " ".join(f"{x:.12g}" for x in [*q, *t])
                        + f" {j} {name}",
                        "",
                    ]
                )
                if len(preview) < 6 and (j == 1 or j % max(1, len(selected) // 6) == 0):
                    overlay = rgb.copy()
                    overlay[moving] = (
                        overlay[moving] * 0.3 + np.array([0, 0, 170])
                    ).astype(np.uint8)
                    preview.append(
                        cv2.resize(overlay, (512, round(height / width * 512)))
                    )
                seen.append(i)
                if j % 20 == 0:
                    progress(f"Prepared splat frame {j}/{len(selected)}")
    finally:
        cap.release()
    if seen != selected.tolist():
        raise ValueError("Incomplete frame extraction.")
    (out / "sparse/0/cameras.txt").write_text("\n".join(cam_lines) + "\n")
    (out / "sparse/0/images.txt").write_text("\n".join(image_lines) + "\n")
    env = []
    for view in views:
        with np.load(result_dir / view["file"], allow_pickle=False) as d:
            env.append(
                (view, {k: d[k] for k in ["points", "colors", "pixels", "depths"]})
            )
    points = np.concatenate([a["points"] for _, a in env])
    colors = np.concatenate([a["colors"] for _, a in env])
    points, colors, seed_report = filter_seeds(
        points, colors, env, min(150000, settings.get("splat_max_gaussians", 1000000))
    )
    with (out / "sparse/0/points3D.txt").open("w") as f:
        for i, (p, c) in enumerate(
            zip(points, np.rint(colors * 255).astype(np.uint8)), 1
        ):
            f.write(f"{i} {p[0]:.9g} {p[1]:.9g} {p[2]:.9g} {c[0]} {c[1]} {c[2]} 0\n")
    if preview:
        cv2.imwrite(str(out / "masks_preview.jpg"), np.concatenate(preview, axis=0))
    fps = m["source"]["fps_num"] / m["source"]["fps_den"]
    write_camera_path(out / "camera_path.json", poses, ks, *original, fps)
    write_camera_path(
        out / "nearby_path.json",
        poses,
        ks,
        *original,
        fps,
        offset=seed_report["depth_scale"] * 0.025,
    )
    report = dict(
        schema="vipe-splat-dataset",
        version=1,
        source_job=m["job_id"],
        source_sha256=m["source"]["sha256"],
        frame_indices=selected.tolist(),
        source_frame_indices=source_indices[selected].tolist(),
        width=width,
        height=height,
        count=len(selected),
        fps=fps,
        mask_excluded_mean=float(np.mean(mask_fractions)),
        mask_excluded_max=max(mask_fractions),
        masks="white=photometric training; black=ignore; dilated TrackAnything dynamics; sky retained",
        cameras="frozen ViPE; COLMAP-format interchange only; no COLMAP solve",
        seed_source="filtered dense environment; multi-view depth agreement and voxel deduplication",
        heldout="Every 8th training image held out of photometric fitting only; ViPE cameras/depth used the full shot.",
        world="ViPE OpenCV world; LichtFeld import/export applies Y/Z flip",
        **seed_report,
    )
    (out / "dataset.json").write_text(json.dumps(report, indent=2))
    progress("Splat dataset ready: " + json.dumps(report))
    return report
