"""Build artist-readable audit artifacts from sealed camera-free tracks."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from camera_evidence.cameras import CameraSequence
from camera_evidence.evaluate import (
    EvaluationConfig,
    _load_windows,
    _percentile,
    _project,
    _projection,
    _role_from_query_ids,
    _static_consensus,
    evaluate_camera,
    pair_support,
)
from camera_evidence.tapir_tracks import sha256_file


@dataclass(frozen=True, kw_only=True)
class AuditConfig:
    observability_baselines: tuple[int, ...] = (8, 16)
    minimum_observability_tracks: int = 25
    maximum_landmarks: int = 5000
    maximum_landmark_median_error_evaluation_px: float = 4.0
    maximum_landmark_observation_error_evaluation_px: float = 4.0
    maximum_landmark_rms_error_evaluation_px: float = 2.0
    minimum_positive_depth_fraction: float = 0.8
    minimum_landmark_parallax_degrees: float = 0.25
    trajectory_warning_robust_z: float = 8.0
    trajectory_hard_robust_z: float = 12.0
    image_corroboration_evaluation_px: float = 4.0

    def __post_init__(self) -> None:
        if not self.observability_baselines or any(value < 1 for value in self.observability_baselines):
            raise ValueError("observability baselines must be positive")
        if self.minimum_observability_tracks < 8 or self.maximum_landmarks < 1:
            raise ValueError("audit support thresholds must be positive")
        if (
            self.maximum_landmark_median_error_evaluation_px <= 0
            or self.maximum_landmark_observation_error_evaluation_px <= 0
            or self.maximum_landmark_rms_error_evaluation_px <= 0
        ):
            raise ValueError("landmark error threshold must be positive")
        if not 0 <= self.minimum_landmark_parallax_degrees < 180:
            raise ValueError("minimum landmark parallax must be in [0, 180)")


def _find_fundamental(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    threshold_px: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    if len(points_a) < 8:
        return None, np.zeros(len(points_a), dtype=bool)
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.FM_RANSAC
    try:
        fundamental, mask = cv2.findFundamentalMat(
            points_a.astype(np.float64),
            points_b.astype(np.float64),
            method,
            threshold_px,
            0.999,
            10000,
        )
    except cv2.error:
        return None, np.zeros(len(points_a), dtype=bool)
    if fundamental is None or mask is None or fundamental.shape != (3, 3):
        return None, np.zeros(len(points_a), dtype=bool)
    return fundamental, mask.ravel().astype(bool)


def _find_homography(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    threshold_px: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    if len(points_a) < 4:
        return None, np.zeros(len(points_a), dtype=bool)
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    homography, mask = cv2.findHomography(
        points_a.astype(np.float64),
        points_b.astype(np.float64),
        method,
        threshold_px,
        None,
        10000,
        0.999,
    )
    if homography is None or mask is None or homography.shape != (3, 3):
        return None, np.zeros(len(points_a), dtype=bool)
    return homography, mask.ravel().astype(bool)


def _nominal_pose_geometry(
    fundamental: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    inliers: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[float | None, float | None]:
    selected_a = points_a[inliers]
    selected_b = points_b[inliers]
    if len(selected_a) < 8:
        return None, None
    focal = float(max(width, height))
    intrinsic = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    essential = intrinsic.T @ fundamental @ intrinsic
    try:
        _, rotation, translation, pose_mask = cv2.recoverPose(
            essential,
            selected_a.astype(np.float64),
            selected_b.astype(np.float64),
            intrinsic,
        )
    except cv2.error:
        return None, None
    normalized_a = cv2.undistortPoints(selected_a[:, None], intrinsic, None).reshape(-1, 2)
    normalized_b = cv2.undistortPoints(selected_b[:, None], intrinsic, None).reshape(-1, 2)
    projection_a = np.column_stack([np.eye(3), np.zeros(3)])
    projection_b = np.column_stack([rotation, translation])
    homogeneous = cv2.triangulatePoints(
        projection_a,
        projection_b,
        normalized_a.T,
        normalized_b.T,
    )
    finite = np.abs(homogeneous[3]) > 1e-12
    points = np.full((len(selected_a), 3), np.nan, dtype=np.float64)
    points[finite] = (homogeneous[:3, finite] / homogeneous[3, finite]).T
    camera_b = -(rotation.T @ translation).ravel()
    ray_a = points
    ray_b = points - camera_b
    norms = np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
    valid = finite & np.isfinite(points).all(axis=1) & (norms > 1e-12)
    cosines = np.ones(len(points), dtype=np.float64)
    cosines[valid] = np.sum(ray_a[valid] * ray_b[valid], axis=1) / norms[valid]
    angles = np.degrees(np.arccos(np.clip(cosines[valid], -1.0, 1.0)))
    cheirality = float(np.count_nonzero(pose_mask) / max(len(pose_mask), 1))
    return _percentile(angles, 50), cheirality


def _observability(
    *,
    windows: list[dict[str, np.ndarray]],
    width: int,
    height: int,
    native_pixels_per_evaluation_pixel: float,
    evaluation_config: EvaluationConfig,
    audit_config: AuditConfig,
) -> dict[str, Any]:
    cv2.setRNGSeed(0)
    threshold = (
        evaluation_config.static_ransac_threshold_evaluation_px
        * native_pixels_per_evaluation_pixel
    )
    rows: list[dict[str, Any]] = []
    for window in windows:
        tracks = window["tracks_xy_native"].astype(np.float64).transpose(1, 0, 2)
        visible = window["visible"].astype(bool).T
        roles = _role_from_query_ids(window["query_ids"])
        for baseline in audit_config.observability_baselines:
            for local_a in range(max(0, len(tracks) - baseline)):
                local_b = local_a + baseline
                keep = visible[local_a] & visible[local_b] & (roles == 1)
                points_a = tracks[local_a, keep]
                points_b = tracks[local_b, keep]
                finite = np.isfinite(points_a).all(axis=1) & np.isfinite(points_b).all(axis=1)
                points_a, points_b = points_a[finite], points_b[finite]
                fundamental, fundamental_inliers = _find_fundamental(
                    points_a,
                    points_b,
                    threshold_px=threshold,
                )
                homography, homography_inliers = _find_homography(
                    points_a,
                    points_b,
                    threshold_px=threshold,
                )
                if fundamental is None or homography is None:
                    continue
                support = pair_support(
                    points_a[fundamental_inliers],
                    width=width,
                    height=height,
                    config=evaluation_config,
                )
                if not support["supported"]:
                    continue
                triangulation_angle, cheirality = _nominal_pose_geometry(
                    fundamental,
                    points_a,
                    points_b,
                    fundamental_inliers,
                    width=width,
                    height=height,
                )
                rows.append(
                    {
                        "frame_a": int(window["start_frame"]) + local_a,
                        "frame_b": int(window["start_frame"]) + local_b,
                        "baseline": baseline,
                        "track_count": int(len(points_a)),
                        "fundamental_inlier_fraction": float(fundamental_inliers.mean()),
                        "homography_inlier_fraction": float(homography_inliers.mean()),
                        "full_model_improvement": float(
                            fundamental_inliers.mean() - homography_inliers.mean()
                        ),
                        "median_displacement_evaluation_px": float(
                            np.median(np.linalg.norm(points_b - points_a, axis=1))
                            / native_pixels_per_evaluation_pixel
                        ),
                        "nominal_triangulation_angle_median_deg": triangulation_angle,
                        "nominal_cheirality_fraction": cheirality,
                    }
                )
    improvements = [row["full_model_improvement"] for row in rows]
    angles = [
        row["nominal_triangulation_angle_median_deg"]
        for row in rows
        if row["nominal_triangulation_angle_median_deg"] is not None
    ]
    cheirality = [
        row["nominal_cheirality_fraction"]
        for row in rows
        if row["nominal_cheirality_fraction"] is not None
    ]
    improvement_p50 = _percentile(improvements, 50)
    angle_p50 = _percentile(angles, 50)
    cheirality_p50 = _percentile(cheirality, 50)
    if not rows:
        state = "INSUFFICIENT_EVIDENCE"
    elif (
        improvement_p50 is not None
        and improvement_p50 >= 0.2
        and angle_p50 is not None
        and angle_p50 >= 0.5
        and cheirality_p50 is not None
        and cheirality_p50 >= 0.95
    ):
        state = "TRANSLATION_OBSERVABLE"
    else:
        state = "WEAK_PLANAR_OR_ROTATION_DOMINATED"
    return {
        "state": state,
        "camera_candidate_inputs_used": False,
        "population_role": "observability_only",
        "supported_pair_count": len(rows),
        "full_model_improvement_p50": improvement_p50,
        "nominal_triangulation_angle_median_deg_p50": angle_p50,
        "nominal_cheirality_fraction_p50": cheirality_p50,
        "median_displacement_evaluation_px_p50": _percentile(
            [row["median_displacement_evaluation_px"] for row in rows],
            50,
        ),
        "pairs": rows,
    }


def _static_query_mask(
    tracks: np.ndarray,
    visible: np.ndarray,
    roles: np.ndarray,
    *,
    evaluation_config: EvaluationConfig,
    native_pixels_per_evaluation_pixel: float,
) -> np.ndarray:
    query_count = tracks.shape[1]
    votes = np.zeros(query_count, dtype=np.int32)
    trials = np.zeros(query_count, dtype=np.int32)
    for baseline in evaluation_config.baselines:
        for local_a in range(max(0, len(tracks) - baseline)):
            local_b = local_a + baseline
            candidates = visible[local_a] & visible[local_b] & (roles == 0)
            indices = np.flatnonzero(candidates)
            points_a = tracks[local_a, indices]
            points_b = tracks[local_b, indices]
            finite = np.isfinite(points_a).all(axis=1) & np.isfinite(points_b).all(axis=1)
            indices = indices[finite]
            points_a, points_b = points_a[finite], points_b[finite]
            inliers = _static_consensus(
                points_a,
                points_b,
                evaluation_config,
                native_pixels_per_evaluation_pixel=native_pixels_per_evaluation_pixel,
            )
            trials[indices] += 1
            votes[indices[inliers]] += 1
    fractions = votes / np.maximum(trials, 1)
    return (
        (roles == 0)
        & (votes >= evaluation_config.minimum_static_votes)
        & (fractions >= evaluation_config.minimum_static_vote_fraction)
    )


def _checker_color(point_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    cell = max(width / 4.0, height / 4.0, 1.0)
    cell_x = int(point_xy[0] // cell)
    cell_y = int(point_xy[1] // cell)
    palette = np.array(
        [[40, 230, 70], [240, 55, 60], [55, 120, 245], [245, 210, 45]],
        dtype=np.uint8,
    )
    return palette[(cell_x + cell_y) % len(palette)]


def _triangulate_multiview(
    projections: list[np.ndarray],
    observations_xy: np.ndarray,
) -> np.ndarray | None:
    if len(projections) < 2 or len(projections) != len(observations_xy):
        return None
    rows: list[np.ndarray] = []
    for projection, (u, v) in zip(projections, observations_xy, strict=True):
        rows.append(u * projection[2] - projection[0])
        rows.append(v * projection[2] - projection[1])
    _, _, vh = np.linalg.svd(np.stack(rows), full_matrices=False)
    homogeneous = vh[-1]
    if not np.isfinite(homogeneous).all() or abs(homogeneous[3]) < 1e-12:
        return None
    point = homogeneous[:3] / homogeneous[3]
    return point if np.isfinite(point).all() else None


def _maximum_parallax_degrees(point: np.ndarray, poses_c2w: list[np.ndarray]) -> float:
    centers = np.stack([pose[:3, 3] for pose in poses_c2w])
    rays = point[None] - centers
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms <= 1e-12):
        return 0.0
    unit_rays = rays / norms[:, None]
    minimum_cosine = float(np.clip(np.min(unit_rays @ unit_rays.T), -1.0, 1.0))
    return float(np.degrees(np.arccos(minimum_cosine)))


def _collect_landmarks(
    *,
    camera: CameraSequence,
    windows: list[dict[str, np.ndarray]],
    native_pixels_per_evaluation_pixel: float,
    evaluation_config: EvaluationConfig,
    audit_config: AuditConfig,
) -> list[dict[str, Any]]:
    camera_by_frame = camera.by_frame()
    # Identical matrices, computed once per frame instead of once per query and frame.
    projection_by_frame = {frame: _projection(*value) for frame, value in camera_by_frame.items()}
    maximum_error_native_px = (
        audit_config.maximum_landmark_observation_error_evaluation_px
        * native_pixels_per_evaluation_pixel
    )
    landmarks: list[dict[str, Any]] = []
    for window in windows:
        start = int(window["start_frame"])
        tracks = window["tracks_xy_native"].astype(np.float64).transpose(1, 0, 2)
        visible = window["visible"].astype(bool).T
        roles = _role_from_query_ids(window["query_ids"])
        static_queries = _static_query_mask(
            tracks,
            visible,
            roles,
            evaluation_config=evaluation_config,
            native_pixels_per_evaluation_pixel=native_pixels_per_evaluation_pixel,
        )
        for query_index in np.flatnonzero(static_queries):
            local_frames = np.flatnonzero(visible[:, query_index])
            local_frames = np.asarray(
                [value for value in local_frames if start + int(value) in camera_by_frame],
                dtype=np.int64,
            )
            if (
                len(local_frames) < 3
                or local_frames[-1] - local_frames[0]
                < evaluation_config.minimum_triangulation_span_frames
            ):
                continue
            global_frames = np.asarray([start + int(value) for value in local_frames], dtype=np.int32)
            observations = tracks[local_frames, query_index]
            projections = [projection_by_frame[int(frame)] for frame in global_frames]
            poses = [camera_by_frame[int(frame)][0] for frame in global_frames]

            # Fit on a deterministic subset and keep disjoint validation
            # observations untouched by the hypothesis shown in the residual review.
            fit_mask = (np.arange(len(global_frames)) % 2) == 0
            fit_mask[-1] = True
            validation_mask = ~fit_mask
            if np.count_nonzero(fit_mask) < 2 or not np.any(validation_mask):
                continue
            fit_point = _triangulate_multiview(
                [projection for projection, keep in zip(projections, fit_mask, strict=True) if keep],
                observations[fit_mask],
            )
            if fit_point is None:
                continue

            initial_errors = np.full(len(global_frames), np.inf, dtype=np.float64)
            initial_positive = np.zeros(len(global_frames), dtype=bool)
            for index, (projection, observed) in enumerate(
                zip(projections, observations, strict=True)
            ):
                predicted, depth = _project(projection, fit_point)
                if np.isfinite(predicted).all():
                    initial_errors[index] = float(np.linalg.norm(predicted - observed))
                initial_positive[index] = depth > 0
            initial_validation = validation_mask & initial_positive & np.isfinite(initial_errors)
            if not np.any(initial_validation):
                continue
            validation_median = float(
                np.median(initial_errors[initial_validation])
                / native_pixels_per_evaluation_pixel
            )
            if validation_median > audit_config.maximum_landmark_median_error_evaluation_px:
                continue

            inliers = initial_positive & np.isfinite(initial_errors) & (initial_errors <= maximum_error_native_px)
            if np.count_nonzero(inliers) < 3 or not np.any(inliers & validation_mask):
                continue
            refit_point = _triangulate_multiview(
                [projection for projection, keep in zip(projections, inliers, strict=True) if keep],
                observations[inliers],
            )
            if refit_point is None:
                continue

            observation_frames: list[int] = []
            observed_xy: list[np.ndarray] = []
            projected_xy: list[np.ndarray] = []
            residuals: list[float] = []
            positive_depth: list[bool] = []
            refit_projected_xy: list[np.ndarray] = []
            refit_residuals: list[float] = []
            observation_roles: list[int] = []
            accepted_poses: list[np.ndarray] = []
            for index in np.flatnonzero(inliers):
                predicted, depth = _project(projections[index], fit_point)
                error = float(np.linalg.norm(predicted - observations[index]))
                predicted_refit, _ = _project(
                    projections[index],
                    refit_point,
                )
                refit_error = float(
                    np.linalg.norm(predicted_refit - observations[index])
                )
                if (
                    not np.isfinite(error)
                    or not np.isfinite(predicted).all()
                    or error > maximum_error_native_px
                ):
                    continue
                observation_frames.append(int(global_frames[index]))
                observed_xy.append(observations[index])
                projected_xy.append(predicted)
                residuals.append(error)
                positive_depth.append(depth > 0)
                refit_projected_xy.append(predicted_refit)
                refit_residuals.append(refit_error)
                observation_roles.append(0 if fit_mask[index] else 1)
                accepted_poses.append(poses[index])
            if len(observation_frames) < 3 or 1 not in observation_roles:
                continue
            positive_fraction = float(np.mean(positive_depth))
            validation_errors = [
                error
                for error, role in zip(residuals, observation_roles, strict=True)
                if role == 1
            ]
            median_evaluation_error = float(
                np.median(validation_errors) / native_pixels_per_evaluation_pixel
            )
            rms_evaluation_error = float(
                np.sqrt(np.mean(np.square(validation_errors)))
                / native_pixels_per_evaluation_pixel
            )
            parallax_degrees = _maximum_parallax_degrees(fit_point, accepted_poses)
            if (
                positive_fraction < audit_config.minimum_positive_depth_fraction
                or median_evaluation_error > audit_config.maximum_landmark_median_error_evaluation_px
                or rms_evaluation_error > audit_config.maximum_landmark_rms_error_evaluation_px
                or parallax_degrees < audit_config.minimum_landmark_parallax_degrees
            ):
                continue
            first_observed = np.asarray(observed_xy[0])
            landmarks.append(
                {
                    "id": int(window["query_ids"][query_index]),
                    "xyz": fit_point,
                    "refit_xyz": refit_point,
                    "color": _checker_color(
                        first_observed,
                        camera.plate_width,
                        camera.plate_height,
                    ),
                    "frames": np.asarray(observation_frames, dtype=np.int32),
                    "refit_projected_xy": np.asarray(refit_projected_xy, dtype=np.float32),
                    "refit_residual_px": np.asarray(refit_residuals, dtype=np.float32),
                    "observed_xy": np.asarray(observed_xy, dtype=np.float32),
                    "projected_xy": np.asarray(projected_xy, dtype=np.float32),
                    "residual_px": np.asarray(residuals, dtype=np.float32),
                    "positive_depth": np.asarray(positive_depth, dtype=np.uint8),
                    "observation_roles": np.asarray(observation_roles, dtype=np.uint8),
                    "span": observation_frames[-1] - observation_frames[0],
                    "median_evaluation_error": median_evaluation_error,
                    "rms_evaluation_error": rms_evaluation_error,
                    "positive_depth_fraction": positive_fraction,
                    "parallax_degrees": parallax_degrees,
                    "triangulation_method": "fit_subset_dlt_with_heldout_validation_and_separate_refit",
                }
            )
    landmarks.sort(
        key=lambda row: (-int(row["span"]), float(row["median_evaluation_error"]))
    )
    return landmarks[: audit_config.maximum_landmarks]


def _write_ply(path: Path, landmarks: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(landmarks)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for landmark in landmarks:
            x, y, z = landmark["xyz"]
            red, green, blue = landmark["color"]
            handle.write(
                f"{float(x):.9g} {float(y):.9g} {float(z):.9g} "
                f"{int(red)} {int(green)} {int(blue)}\n"
            )


def _write_landmark_npz(path: Path, landmarks: list[dict[str, Any]]) -> None:
    def concatenate(name: str, *, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
        if not landmarks:
            return np.empty(shape, dtype=dtype)
        return np.concatenate(
            [np.asarray(row[name]) for row in landmarks],
            dtype=dtype,
        )

    offsets = [0]
    for landmark in landmarks:
        offsets.append(offsets[-1] + len(landmark["frames"]))
    xyz = np.asarray([row["xyz"] for row in landmarks], dtype=np.float64).reshape(-1, 3)
    refit_xyz = np.asarray(
        [row["refit_xyz"] for row in landmarks], dtype=np.float64
    ).reshape(-1, 3)
    rgb = np.asarray([row["color"] for row in landmarks], dtype=np.uint8).reshape(-1, 3)
    np.savez_compressed(
        path,
        landmark_ids=np.asarray([row["id"] for row in landmarks], dtype=np.int64),
        landmark_xyz=xyz,
        landmark_refit_xyz=refit_xyz,
        landmark_rgb=rgb,
        landmark_span_frames=np.asarray([row["span"] for row in landmarks], dtype=np.int32),
        landmark_median_error_evaluation_px=np.asarray(
            [row["median_evaluation_error"] for row in landmarks],
            dtype=np.float32,
        ),
        landmark_rms_error_evaluation_px=np.asarray(
            [row["rms_evaluation_error"] for row in landmarks],
            dtype=np.float32,
        ),
        landmark_parallax_degrees=np.asarray(
            [row["parallax_degrees"] for row in landmarks],
            dtype=np.float32,
        ),
        observation_offsets=np.asarray(offsets, dtype=np.int64),
        observation_frames=concatenate("frames", shape=(0,), dtype=np.dtype(np.int32)),
        observed_xy=concatenate("observed_xy", shape=(0, 2), dtype=np.dtype(np.float32)),
        projected_xy=concatenate("projected_xy", shape=(0, 2), dtype=np.dtype(np.float32)),
        residual_px=concatenate("residual_px", shape=(0,), dtype=np.dtype(np.float32)),
        refit_projected_xy=concatenate(
            "refit_projected_xy", shape=(0, 2), dtype=np.dtype(np.float32)
        ),
        refit_residual_px=concatenate(
            "refit_residual_px", shape=(0,), dtype=np.dtype(np.float32)
        ),
        positive_depth=concatenate("positive_depth", shape=(0,), dtype=np.dtype(np.uint8)),
        observation_roles=concatenate(
            "observation_roles", shape=(0,), dtype=np.dtype(np.uint8)
        ),
    )


def _frame_metrics(landmarks: list[dict[str, Any]], frame_count: int) -> list[dict[str, Any]]:
    residuals_by_frame: dict[int, list[float]] = defaultdict(list)
    for landmark in landmarks:
        for frame, residual, role in zip(
            landmark["frames"],
            landmark["residual_px"],
            landmark["observation_roles"],
            strict=True,
        ):
            if role != 1:
                continue
            residuals_by_frame[int(frame)].append(float(residual))
    return [
        {
            "frame": frame,
            "observation_count": len(residuals_by_frame[frame]),
            "reprojection_p50_px": _percentile(residuals_by_frame[frame], 50),
            "reprojection_p95_px": _percentile(residuals_by_frame[frame], 95),
            "reprojection_max_px": _percentile(residuals_by_frame[frame], 100),
        }
        for frame in range(frame_count)
    ]


def _probe_anchors(
    camera: CameraSequence,
    landmarks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not landmarks:
        return []
    reference_index = 0
    reference_frame = int(camera.frame_indices[reference_index])
    world_to_camera = np.linalg.inv(camera.c2w[reference_index])
    candidates: list[dict[str, Any]] = []
    for row in landmarks:
        frame_matches = np.flatnonzero(row["frames"] == reference_frame)
        if not len(frame_matches):
            continue
        xyz = np.asarray(row["xyz"], dtype=np.float64)
        camera_xyz = world_to_camera @ np.r_[xyz, 1.0]
        observation_index = int(frame_matches[0])
        observed_xy = np.asarray(row["observed_xy"][observation_index], dtype=np.float64)
        if (
            not np.isfinite(xyz).all()
            or not np.isfinite(observed_xy).all()
            or not np.isfinite(camera_xyz[2])
            or camera_xyz[2] <= 0
        ):
            continue
        candidates.append(
            {
                "id": int(row["id"]),
                "xyz": xyz,
                "depth": float(camera_xyz[2]),
                "observed_xy": observed_xy,
            }
        )
    if not candidates:
        return []
    depths = np.asarray([row["depth"] for row in candidates], dtype=np.float64)
    anchors: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    layers = (
        ("foreground", 0.15, (0.2, 0.7)),
        ("midground", 0.5, (0.5, 0.5)),
        ("background", 0.85, (0.8, 0.3)),
    )
    for label, quantile, screen_target in layers:
        target_depth = float(np.quantile(depths, quantile))
        depth_order = np.argsort(np.abs(depths - target_depth))
        pool = [
            int(index)
            for index in depth_order[: min(25, len(depth_order))]
            if candidates[int(index)]["id"] not in used_ids
        ]
        if not pool:
            pool = [int(depth_order[0])]
        target_xy = np.asarray(
            [screen_target[0] * camera.plate_width, screen_target[1] * camera.plate_height],
            dtype=np.float64,
        )
        selected = min(
            pool,
            key=lambda index: float(
                np.linalg.norm(
                    (candidates[index]["observed_xy"] - target_xy)
                    / np.asarray([camera.plate_width, camera.plate_height])
                )
            ),
        )
        candidate = candidates[selected]
        used_ids.add(candidate["id"])
        anchors.append(
            {
                "label": label,
                "reference_frame": reference_frame,
                "depth_quantile": quantile,
                "target_camera_depth": target_depth,
                "reference_camera_depth": candidate["depth"],
                "xyz": candidate["xyz"].tolist(),
                "observed_xy": candidate["observed_xy"].tolist(),
                "landmark_id": candidate["id"],
                "supporting_landmark_count": int(len(pool)),
                "placement": "actual_supported_landmark",
                "camera_gauge_only": True,
            }
        )
    return anchors


def _trajectory_metrics(
    camera: CameraSequence,
    frame_metrics: list[dict[str, Any]],
    *,
    native_pixels_per_evaluation_pixel: float,
    audit_config: AuditConfig,
) -> dict[str, Any]:
    centers = camera.c2w[:, :3, 3]
    if len(centers) < 3:
        return {"acceleration_count": 0, "spikes": []}
    acceleration = np.diff(centers, n=2, axis=0)
    magnitudes = np.linalg.norm(acceleration, axis=1)
    median = float(np.median(magnitudes))
    mad = float(np.median(np.abs(magnitudes - median)))
    scale = max(1.4826 * mad, 1e-12)
    robust_z = (magnitudes - median) / scale
    p95_by_frame = {
        int(row["frame"]): row["reprojection_p95_px"]
        for row in frame_metrics
        if row["reprojection_p95_px"] is not None
    }
    spikes: list[dict[str, Any]] = []
    for local_index in np.flatnonzero(np.abs(robust_z) >= audit_config.trajectory_warning_robust_z):
        frame = int(camera.frame_indices[local_index + 1])
        image_p95 = p95_by_frame.get(frame)
        corroborated = (
            image_p95 is not None
            and image_p95 / native_pixels_per_evaluation_pixel
            >= audit_config.image_corroboration_evaluation_px
        )
        spikes.append(
            {
                "frame": frame,
                "acceleration_magnitude": float(magnitudes[local_index]),
                "robust_z": float(robust_z[local_index]),
                "image_reprojection_p95_px": image_p95,
                "corroborated_by_image_residual": corroborated,
                "hard_contradiction": bool(
                    corroborated
                    and abs(robust_z[local_index])
                    >= audit_config.trajectory_hard_robust_z
                ),
            }
        )
    return {
        "acceleration_count": int(len(magnitudes)),
        "acceleration_median": median,
        "acceleration_mad": mad,
        "robust_z_warning_threshold": audit_config.trajectory_warning_robust_z,
        "robust_z_hard_threshold": audit_config.trajectory_hard_robust_z,
        "spikes": spikes,
    }


def _provisional_gate(
    base_report: dict[str, Any],
    observability: dict[str, Any],
    trajectory: dict[str, Any],
    *,
    native_pixels_per_evaluation_pixel: float,
) -> dict[str, Any]:
    selection = base_report["camera_free_selection"]
    metrics = base_report["candidate_metrics"]
    pair_rows = base_report["worst_pair_medians"]
    adjacent = float(selection["adjacent_supported_fraction"])
    gap = int(selection["adjacent_longest_unsupported_gap_frames"])
    heldout_count = int(metrics["heldout_reprojection_observation_count"])
    epipolar_p95 = metrics["epipolar_all_p95_px"]
    reprojection_p95 = metrics["heldout_reprojection_p95_px"]
    epipolar_eval = (
        None if epipolar_p95 is None else epipolar_p95 / native_pixels_per_evaluation_pixel
    )
    reprojection_eval = (
        None
        if reprojection_p95 is None
        else reprojection_p95 / native_pixels_per_evaluation_pixel
    )
    worst_eval = [
        row["candidate_epipolar_p50_px"] / native_pixels_per_evaluation_pixel
        for row in pair_rows
        if row["candidate_epipolar_p50_px"] is not None
    ]
    hard_pair_contradiction = any(value >= 8.0 for value in worst_eval) or (
        sum(value >= 4.0 for value in worst_eval) >= 2
    )
    hard_trajectory = any(row["hard_contradiction"] for row in trajectory["spikes"])
    reasons: list[str] = []
    if adjacent < 0.85 or gap > 8 or heldout_count < 50:
        state = "INSUFFICIENT_EVIDENCE"
        reasons.append("insufficient adjacent coverage or held-out observations")
    elif observability["state"] != "TRANSLATION_OBSERVABLE":
        state = "INSUFFICIENT_EVIDENCE"
        reasons.append("translation/parallax is weak, planar, or rotation dominated")
    elif hard_pair_contradiction or hard_trajectory:
        state = "HARD_FAIL"
        reasons.append("independent image evidence contradicts the candidate camera")
    elif (
        adjacent >= 0.95
        and gap <= 2
        and epipolar_eval is not None
        and epipolar_eval <= 1.0
        and reprojection_eval is not None
        and reprojection_eval <= 2.0
    ):
        state = "PASS"
        reasons.append("archive-aligned provisional gates passed")
    else:
        state = "WARNING"
        reasons.append("usable evidence exists but one or more pass gates were missed")
    epipolar_score = (
        None if epipolar_eval is None else float(100.0 * np.exp(-epipolar_eval / 2.0))
    )
    reprojection_score = (
        None
        if reprojection_eval is None
        else float(100.0 * np.exp(-reprojection_eval / 3.0))
    )
    coverage_score = float(100.0 * adjacent * min(1.0, heldout_count / 500.0))
    trajectory_score = float(
        100.0
        * (
            1.0
            - min(
                1.0,
                sum(row["corroborated_by_image_residual"] for row in trajectory["spikes"])
                / 3.0,
            )
        )
    )
    weighted = [
        (epipolar_score, 0.35),
        (reprojection_score, 0.30),
        (coverage_score, 0.20),
        (trajectory_score, 0.10),
    ]
    available = [(value, weight) for value, weight in weighted if value is not None]
    score = (
        sum(value * weight for value, weight in available)
        / sum(weight for _, weight in available)
        if available
        else None
    )
    return {
        "state": state,
        "reasons": reasons,
        "camera_quality_score_without_sift": score,
        "profile": "archive_aligned_provisional_uncalibrated",
        "promotion_allowed": False,
        "epipolar_p95_evaluation_px": epipolar_eval,
        "heldout_reprojection_p95_evaluation_px": reprojection_eval,
    }


def build_camera_audit(
    *,
    camera: CameraSequence,
    evidence_dir: Path,
    partition: str,
    output_dir: Path,
    evaluation_config: EvaluationConfig | None = None,
    audit_config: AuditConfig | None = None,
    candidate_lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation_config = evaluation_config or EvaluationConfig()
    audit_config = audit_config or AuditConfig()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_report = evaluate_camera(
        camera=camera,
        evidence_dir=evidence_dir,
        partition=partition,
        config=evaluation_config,
    )
    evidence_manifest, windows = _load_windows(evidence_dir.resolve(), partition)
    native_pixels_per_evaluation_pixel = max(
        float(evidence_manifest["source"]["evaluation_to_native_scale_x"]),
        float(evidence_manifest["source"]["evaluation_to_native_scale_y"]),
    )
    observability = _observability(
        windows=windows,
        width=camera.plate_width,
        height=camera.plate_height,
        native_pixels_per_evaluation_pixel=native_pixels_per_evaluation_pixel,
        evaluation_config=evaluation_config,
        audit_config=audit_config,
    )
    landmarks = _collect_landmarks(
        camera=camera,
        windows=windows,
        native_pixels_per_evaluation_pixel=native_pixels_per_evaluation_pixel,
        evaluation_config=evaluation_config,
        audit_config=audit_config,
    )
    landmark_npz = output_dir / "audit_landmarks.npz"
    landmark_ply = output_dir / "audit_landmarks.ply"
    _write_landmark_npz(landmark_npz, landmarks)
    _write_ply(landmark_ply, landmarks)
    frames = _frame_metrics(landmarks, len(camera.frame_indices))
    frame_metrics_path = output_dir / "frame_metrics.json"
    frame_metrics_path.write_text(
        json.dumps(frames, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    probe_anchors = _probe_anchors(camera, landmarks)
    probes_path = output_dir / "probe_anchors.json"
    probes_path.write_text(
        json.dumps(probe_anchors, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    intrinsic_matrices = np.stack(
        [
            np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            for fx, fy, cx, cy in camera.intrinsics
        ]
    )
    camera_path = output_dir / "audit_camera.npz"
    np.savez_compressed(
        camera_path,
        frame_indices=camera.frame_indices,
        c2w=camera.c2w,
        intrinsic=intrinsic_matrices,
        plate_size=np.asarray([camera.plate_width, camera.plate_height], dtype=np.int32),
    )
    trajectory = _trajectory_metrics(
        camera,
        frames,
        native_pixels_per_evaluation_pixel=native_pixels_per_evaluation_pixel,
        audit_config=audit_config,
    )
    gate = _provisional_gate(
        base_report,
        observability,
        trajectory,
        native_pixels_per_evaluation_pixel=native_pixels_per_evaluation_pixel,
    )
    report = {
        "schema_version": 4,
        "metric_status": "UNTRUSTED_PENDING_PROGRESSIVE_HUMAN_VALIDATION",
        "candidate_id": camera.candidate_id,
        "partition": partition,
        "candidate_lock": candidate_lock,
        "evaluation_config": asdict(evaluation_config),
        "audit_config": asdict(audit_config),
        "base_evaluation": base_report,
        "observability": observability,
        "trajectory": trajectory,
        "provisional_gate": gate,
        "sparse_audit_landmarks": {
            "classification": "TRACK_DERIVED_CANDIDATE_CAMERA_HYPOTHESIS_NOT_DEPTH_SURFACE",
            "source": (
                f"camera-independent {evidence_manifest['tracker']['name']} tracks "
                "triangulated through frozen candidate"
            ),
            "construction": (
                "fit-only multiview DLT projections are evaluated on disjoint "
                "validation observations; an all-inlier refit is exported separately"
            ),
            "residual_projection_policy": "FIT_SUBSET_ONLY_VALIDATION_OBSERVATIONS_ONLY",
            "refit_geometry_policy": "EXPORTED_SEPARATELY_NEVER_USED_FOR_HELDOUT_RESIDUAL",
            "camera_score_dependency": "forbidden",
            "surface_attachment_guarantee": False,
            "landmark_count": len(landmarks),
            "observation_count": int(sum(len(row["frames"]) for row in landmarks)),
            "validation_observation_count": int(
                sum(np.count_nonzero(row["observation_roles"] == 1) for row in landmarks)
            ),
            "npz": str(landmark_npz),
            "npz_sha256": sha256_file(landmark_npz),
            "ply": str(landmark_ply),
            "ply_sha256": sha256_file(landmark_ply),
        },
        "frame_metrics": {
            "path": str(frame_metrics_path),
            "sha256": sha256_file(frame_metrics_path),
        },
        "probe_anchors": {
            "path": str(probes_path),
            "sha256": sha256_file(probes_path),
            "count": len(probe_anchors),
        },
        "camera_artifact": {
            "path": str(camera_path),
            "sha256": sha256_file(camera_path),
        },
        "review_status": "PROGRESSIVE_HUMAN_VALIDATION_REQUIRED_DO_NOT_SCORE_OR_PROMOTE",
    }
    report_path = output_dir / "camera_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    report["report_sha256"] = sha256_file(report_path)
    return report

