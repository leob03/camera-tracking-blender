"""Score frozen camera candidates with sealed, camera-free point tracks."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from camera_evidence.cameras import CameraSequence
from camera_evidence.tapir_tracks import sha256_file


@dataclass(frozen=True, kw_only=True)
class EvaluationConfig:
    baselines: tuple[int, ...] = (1, 2, 4, 8, 16)
    static_ransac_threshold_evaluation_px: float = 1.5
    static_ransac_confidence: float = 0.999
    minimum_pair_tracks: int = 16
    supported_pair_tracks: int = 25
    coverage_grid_columns: int = 6
    coverage_grid_rows: int = 4
    minimum_occupied_cells: int = 8
    minimum_occupied_quadrants: int = 3
    minimum_static_votes: int = 2
    minimum_static_vote_fraction: float = 0.6
    minimum_triangulation_span_frames: int = 8

    def __post_init__(self) -> None:
        if not self.baselines or any(value < 1 for value in self.baselines):
            raise ValueError("baselines must be positive")
        if self.static_ransac_threshold_evaluation_px <= 0:
            raise ValueError("RANSAC threshold must be positive")
        if not 0 < self.static_ransac_confidence < 1:
            raise ValueError("RANSAC confidence must be in (0,1)")
        if self.minimum_pair_tracks < 8 or self.supported_pair_tracks < self.minimum_pair_tracks:
            raise ValueError("pair support thresholds are inconsistent")


def _percentile(values: list[float] | np.ndarray, quantile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, quantile)) if array.size else None


def _role_from_query_ids(query_ids: np.ndarray) -> np.ndarray:
    """Return the deterministic sealed-evidence role for each query ID.

    Role 0 is the locked camera-test population. Role 1 is reserved for the
    camera-free observability/parallax check and must never enter a candidate
    camera score.
    """
    values = np.asarray(query_ids, dtype=np.uint64)
    mixed = values ^ (values >> np.uint64(23)) ^ (values * np.uint64(0x9E3779B185EBCA87))
    return ((mixed >> np.uint64(1)) & np.uint64(1)).astype(np.uint8)


def _camera_scoring_query_mask(partition: str, query_ids: np.ndarray) -> np.ndarray:
    """Select queries that may be compared to a candidate camera.

    The routing partition remains a development/tuning population and is not
    subdivided. The final-audit partition is split deterministically: only
    role 0 is camera-test evidence; role 1 remains observability-only.
    """
    query_ids = np.asarray(query_ids)
    if partition == "routing":
        return np.ones(len(query_ids), dtype=bool)
    if partition == "final_audit":
        return _role_from_query_ids(query_ids) == 0
    raise ValueError(f"Unknown evidence partition: {partition}")


def _native_to_evaluation_px(
    value: float | None,
    native_pixels_per_evaluation_pixel: float,
) -> float | None:
    return None if value is None else float(value / native_pixels_per_evaluation_pixel)


def _camera_matrix(intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental_from_camera(
    c2w_a: np.ndarray,
    c2w_b: np.ndarray,
    intrinsics_a: np.ndarray,
    intrinsics_b: np.ndarray,
) -> np.ndarray:
    transform_b_from_a = np.linalg.inv(c2w_b) @ c2w_a
    essential = _skew(transform_b_from_a[:3, 3]) @ transform_b_from_a[:3, :3]
    return np.linalg.inv(_camera_matrix(intrinsics_b)).T @ essential @ np.linalg.inv(_camera_matrix(intrinsics_a))


def symmetric_epipolar_error(
    fundamental: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> np.ndarray:
    homogeneous_a = np.column_stack([points_a, np.ones(len(points_a))])
    homogeneous_b = np.column_stack([points_b, np.ones(len(points_b))])
    lines_b = homogeneous_a @ fundamental.T
    lines_a = homogeneous_b @ fundamental
    numerator = np.sum(homogeneous_b * lines_b, axis=1) ** 2
    denominator_a = np.maximum(np.sum(lines_a[:, :2] ** 2, axis=1), 1e-12)
    denominator_b = np.maximum(np.sum(lines_b[:, :2] ** 2, axis=1), 1e-12)
    return np.sqrt(0.5 * numerator * (1.0 / denominator_a + 1.0 / denominator_b))


def pair_support(
    points: np.ndarray,
    *,
    width: int,
    height: int,
    config: EvaluationConfig,
) -> dict[str, int | bool]:
    if not len(points):
        return {"track_count": 0, "occupied_cells": 0, "occupied_quadrants": 0, "supported": False}
    columns = np.clip(
        (points[:, 0] * config.coverage_grid_columns / width).astype(int),
        0,
        config.coverage_grid_columns - 1,
    )
    rows = np.clip(
        (points[:, 1] * config.coverage_grid_rows / height).astype(int),
        0,
        config.coverage_grid_rows - 1,
    )
    occupied_cells = len(np.unique(rows * config.coverage_grid_columns + columns))
    quadrants = (points[:, 0] >= width / 2).astype(int) + 2 * (points[:, 1] >= height / 2).astype(int)
    occupied_quadrants = len(np.unique(quadrants))
    supported = (
        len(points) >= config.supported_pair_tracks
        and occupied_cells >= config.minimum_occupied_cells
        and occupied_quadrants >= config.minimum_occupied_quadrants
    )
    return {
        "track_count": int(len(points)),
        "occupied_cells": int(occupied_cells),
        "occupied_quadrants": int(occupied_quadrants),
        "supported": bool(supported),
    }


def _static_consensus(
    points_a: np.ndarray,
    points_b: np.ndarray,
    config: EvaluationConfig,
    *,
    native_pixels_per_evaluation_pixel: float,
) -> np.ndarray:
    if len(points_a) < config.minimum_pair_tracks:
        return np.zeros(len(points_a), dtype=bool)
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.FM_RANSAC
    try:
        fundamental, mask = cv2.findFundamentalMat(
            points_a.astype(np.float64),
            points_b.astype(np.float64),
            method,
            config.static_ransac_threshold_evaluation_px * native_pixels_per_evaluation_pixel,
            config.static_ransac_confidence,
            10000,
        )
    except cv2.error:
        return np.zeros(len(points_a), dtype=bool)
    if fundamental is None or mask is None or fundamental.shape != (3, 3):
        return np.zeros(len(points_a), dtype=bool)
    return mask.ravel().astype(bool)


def _projection(c2w: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    return _camera_matrix(intrinsics) @ np.linalg.inv(c2w)[:3]


def _triangulate(
    projection_a: np.ndarray,
    projection_b: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> np.ndarray | None:
    rows = np.stack(
        [
            point_a[0] * projection_a[2] - projection_a[0],
            point_a[1] * projection_a[2] - projection_a[1],
            point_b[0] * projection_b[2] - projection_b[0],
            point_b[1] * projection_b[2] - projection_b[1],
        ]
    )
    _, _, vh = np.linalg.svd(rows)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12 or not np.isfinite(homogeneous).all():
        return None
    point = homogeneous[:3] / homogeneous[3]
    return point if np.isfinite(point).all() else None


def _project(projection: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, float]:
    homogeneous = projection @ np.append(point, 1.0)
    if abs(homogeneous[2]) < 1e-12:
        return np.array([np.nan, np.nan]), float(homogeneous[2])
    return homogeneous[:2] / homogeneous[2], float(homogeneous[2])


def _load_windows(evidence_dir: Path, partition: str) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    manifest = json.loads((evidence_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "sealed_camera_free_track_evidence":
        raise ValueError("Track evidence is not sealed")
    partition_value = {"routing": 0, "final_audit": 1}.get(partition)
    if partition_value is None:
        raise ValueError(f"Unknown evidence partition: {partition}")
    windows: list[dict[str, np.ndarray]] = []
    for receipt in manifest["windows"]:
        path = evidence_dir / receipt["path"]
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"Evidence digest mismatch: {path}")
        with np.load(path, allow_pickle=False) as package:
            data = {key: np.asarray(package[key]) for key in package.files}
        keep = data["partitions"] == partition_value
        data["query_ids"] = data["query_ids"][keep]
        data["query_xy_native"] = data["query_xy_native"][keep]
        data["tracks_xy_native"] = data["tracks_xy_native"][keep]
        data["visible"] = data["visible"][keep]
        data["visibility_score"] = data["visibility_score"][keep]
        windows.append(data)
    return manifest, windows


def _longest_false_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def evaluate_camera(
    *,
    camera: CameraSequence,
    evidence_dir: Path,
    partition: str,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    config = config or EvaluationConfig()
    cv2.setRNGSeed(0)
    evidence_dir = evidence_dir.resolve()
    evidence_manifest, windows = _load_windows(evidence_dir, partition)
    if (
        camera.plate_width != evidence_manifest["source"]["native_width"]
        or camera.plate_height != evidence_manifest["source"]["native_height"]
    ):
        raise ValueError("Camera and track evidence use different plate dimensions")
    native_pixels_per_evaluation_pixel = max(
        float(evidence_manifest["source"]["evaluation_to_native_scale_x"]),
        float(evidence_manifest["source"]["evaluation_to_native_scale_y"]),
    )
    if not np.isfinite(native_pixels_per_evaluation_pixel) or native_pixels_per_evaluation_pixel <= 0:
        raise ValueError("Evidence contains an invalid evaluation-to-native pixel scale")
    static_ransac_threshold_native_px = (
        config.static_ransac_threshold_evaluation_px * native_pixels_per_evaluation_pixel
    )
    camera_by_frame = camera.by_frame()
    pair_rows: list[dict[str, Any]] = []
    all_epipolar_errors: list[float] = []
    reprojection_errors: list[float] = []
    adjacent_support: dict[int, bool] = defaultdict(bool)
    static_track_count = 0
    triangulated_track_count = 0
    partition_query_window_count = 0
    camera_scoring_query_window_count = 0

    for window in windows:
        start = int(window["start_frame"])
        query_ids = np.asarray(window["query_ids"])
        score_queries = _camera_scoring_query_mask(partition, query_ids)
        partition_query_window_count += len(query_ids)
        camera_scoring_query_window_count += int(np.count_nonzero(score_queries))
        tracks = window["tracks_xy_native"][score_queries].astype(np.float64).transpose(1, 0, 2)
        visible = window["visible"][score_queries].astype(bool).T
        query_count = tracks.shape[1]
        static_votes = np.zeros(query_count, dtype=np.int32)
        static_trials = np.zeros(query_count, dtype=np.int32)

        for baseline in config.baselines:
            for local_a in range(max(0, len(tracks) - baseline)):
                local_b = local_a + baseline
                frame_a, frame_b = start + local_a, start + local_b
                if frame_a not in camera_by_frame or frame_b not in camera_by_frame:
                    continue
                candidates = visible[local_a] & visible[local_b]
                candidate_indices = np.flatnonzero(candidates)
                points_a = tracks[local_a, candidate_indices]
                points_b = tracks[local_b, candidate_indices]
                finite = np.isfinite(points_a).all(axis=1) & np.isfinite(points_b).all(axis=1)
                candidate_indices = candidate_indices[finite]
                points_a, points_b = points_a[finite], points_b[finite]
                inliers = _static_consensus(
                    points_a,
                    points_b,
                    config,
                    native_pixels_per_evaluation_pixel=native_pixels_per_evaluation_pixel,
                )
                inlier_indices = candidate_indices[inliers]
                static_trials[candidate_indices] += 1
                static_votes[inlier_indices] += 1
                support = pair_support(
                    points_a[inliers],
                    width=camera.plate_width,
                    height=camera.plate_height,
                    config=config,
                )
                if baseline == 1:
                    adjacent_support[frame_a] |= bool(support["supported"])
                pose_a, intrinsic_a = camera_by_frame[frame_a]
                pose_b, intrinsic_b = camera_by_frame[frame_b]
                errors = symmetric_epipolar_error(
                    fundamental_from_camera(pose_a, pose_b, intrinsic_a, intrinsic_b),
                    points_a[inliers],
                    points_b[inliers],
                )
                all_epipolar_errors.extend(errors[np.isfinite(errors)].tolist())
                p50_native_px = _percentile(errors, 50)
                p95_native_px = _percentile(errors, 95)
                pair_rows.append(
                    {
                        "frame_a": frame_a,
                        "frame_b": frame_b,
                        "baseline": baseline,
                        "camera_free_inlier_count": int(np.count_nonzero(inliers)),
                        "support": support,
                        "candidate_epipolar_p50_native_px": p50_native_px,
                        "candidate_epipolar_p95_native_px": p95_native_px,
                        "candidate_epipolar_p50_evaluation_px": _native_to_evaluation_px(
                            p50_native_px, native_pixels_per_evaluation_pixel
                        ),
                        "candidate_epipolar_p95_evaluation_px": _native_to_evaluation_px(
                            p95_native_px, native_pixels_per_evaluation_pixel
                        ),
                        # Schema-v1 compatibility aliases; these values are native pixels.
                        "candidate_epipolar_p50_px": p50_native_px,
                        "candidate_epipolar_p95_px": p95_native_px,
                    }
                )

        static_fraction = static_votes / np.maximum(static_trials, 1)
        static_queries = (static_votes >= config.minimum_static_votes) & (
            static_fraction >= config.minimum_static_vote_fraction
        )
        static_track_count += int(np.count_nonzero(static_queries))
        for query_index in np.flatnonzero(static_queries):
            local_frames = np.flatnonzero(visible[:, query_index])
            local_frames = np.asarray(
                [local for local in local_frames if start + int(local) in camera_by_frame],
                dtype=np.int64,
            )
            if len(local_frames) < 3 or local_frames[-1] - local_frames[0] < config.minimum_triangulation_span_frames:
                continue
            first, last = int(local_frames[0]), int(local_frames[-1])
            global_first, global_last = start + first, start + last
            pose_first, intrinsic_first = camera_by_frame[global_first]
            pose_last, intrinsic_last = camera_by_frame[global_last]
            point = _triangulate(
                _projection(pose_first, intrinsic_first),
                _projection(pose_last, intrinsic_last),
                tracks[first, query_index],
                tracks[last, query_index],
            )
            if point is None:
                continue
            per_track_errors: list[float] = []
            positive_depths = 0
            for local_frame in local_frames[1:-1]:
                global_frame = start + int(local_frame)
                pose, intrinsic = camera_by_frame[global_frame]
                predicted, depth = _project(_projection(pose, intrinsic), point)
                positive_depths += depth > 0
                error = float(np.linalg.norm(predicted - tracks[local_frame, query_index]))
                if np.isfinite(error):
                    per_track_errors.append(error)
            if not per_track_errors or positive_depths / len(per_track_errors) < 0.8:
                continue
            triangulated_track_count += 1
            reprojection_errors.extend(per_track_errors)

    baseline_summary: dict[str, dict[str, Any]] = {}
    for baseline in config.baselines:
        rows = [row for row in pair_rows if row["baseline"] == baseline]
        medians = [
            row["candidate_epipolar_p50_native_px"]
            for row in rows
            if row["candidate_epipolar_p50_native_px"] is not None
        ]
        median_p50_native_px = _percentile(medians, 50)
        median_p95_native_px = _percentile(medians, 95)
        median_max_native_px = _percentile(medians, 100)
        baseline_summary[str(baseline)] = {
            "pair_count": len(rows),
            "pair_median_p50_native_px": median_p50_native_px,
            "pair_median_p95_native_px": median_p95_native_px,
            "pair_median_max_native_px": median_max_native_px,
            "pair_median_p50_evaluation_px": _native_to_evaluation_px(
                median_p50_native_px, native_pixels_per_evaluation_pixel
            ),
            "pair_median_p95_evaluation_px": _native_to_evaluation_px(
                median_p95_native_px, native_pixels_per_evaluation_pixel
            ),
            "pair_median_max_evaluation_px": _native_to_evaluation_px(
                median_max_native_px, native_pixels_per_evaluation_pixel
            ),
            # Schema-v1 compatibility aliases; these values are native pixels.
            "pair_median_p50_px": median_p50_native_px,
            "pair_median_p95_px": median_p95_native_px,
            "pair_median_max_px": median_max_native_px,
        }
    adjacent_values = np.array(
        [adjacent_support.get(frame_index, False) for frame_index in range(len(camera.frame_indices) - 1)],
        dtype=bool,
    )
    worst_pairs = sorted(
        [row for row in pair_rows if row["candidate_epipolar_p50_native_px"] is not None],
        key=lambda row: row["candidate_epipolar_p50_native_px"],
        reverse=True,
    )[:20]
    epipolar_p50_native_px = _percentile(all_epipolar_errors, 50)
    epipolar_p95_native_px = _percentile(all_epipolar_errors, 95)
    reprojection_p50_native_px = _percentile(reprojection_errors, 50)
    reprojection_p95_native_px = _percentile(reprojection_errors, 95)
    return {
        "schema_version": 2,
        "candidate_id": camera.candidate_id,
        "partition": partition,
        "population_selection": {
            "partition_query_window_count": partition_query_window_count,
            "camera_scoring_query_window_count": camera_scoring_query_window_count,
            "excluded_observability_query_window_count": (
                partition_query_window_count - camera_scoring_query_window_count
            ),
            "policy": (
                "all routing/development queries; metrics are tuning-only"
                if partition == "routing"
                else "final_audit deterministic role 0 (locked camera-test) only; role 1 is observability-only"
            ),
            "deterministic_role_function": "query_id_hash_v1",
        },
        "metric_units": {
            "coordinate_space": "native_plate_pixels",
            "native_pixels_per_evaluation_pixel": native_pixels_per_evaluation_pixel,
            "conversion": "evaluation_px = native_px / native_pixels_per_evaluation_pixel",
            "explicit_suffixes": {
                "*_native_px": "native plate pixels",
                "*_evaluation_px": "evaluation-raster pixels",
            },
            "legacy_unsuffixed_px_fields": (
                "Deprecated schema-v1 compatibility aliases; values are native plate pixels"
            ),
        },
        "camera": {
            "frame_count": len(camera.frame_indices),
            "plate_width": camera.plate_width,
            "plate_height": camera.plate_height,
            "adapter": camera.adapter_metadata,
        },
        "evidence": {
            "directory": str(evidence_dir),
            "manifest_sha256": sha256_file(evidence_dir / "MANIFEST.json"),
            "tracker": evidence_manifest["tracker"],
            "candidate_camera_inputs_used_to_generate_tracks": evidence_manifest["candidate_camera_inputs_used"],
        },
        "config": asdict(config),
        "camera_free_selection": {
            "static_track_window_count": static_track_count,
            "native_pixels_per_evaluation_pixel": native_pixels_per_evaluation_pixel,
            "static_ransac_threshold_native_px": static_ransac_threshold_native_px,
            "pair_count": len(pair_rows),
            "adjacent_supported_fraction": float(adjacent_values.mean()) if adjacent_values.size else 0.0,
            "adjacent_longest_unsupported_gap_frames": _longest_false_run(adjacent_values),
        },
        "candidate_metrics": {
            "epipolar_all_p50_native_px": epipolar_p50_native_px,
            "epipolar_all_p95_native_px": epipolar_p95_native_px,
            "epipolar_all_p50_evaluation_px": _native_to_evaluation_px(
                epipolar_p50_native_px, native_pixels_per_evaluation_pixel
            ),
            "epipolar_all_p95_evaluation_px": _native_to_evaluation_px(
                epipolar_p95_native_px, native_pixels_per_evaluation_pixel
            ),
            # Schema-v1 compatibility aliases; these values are native pixels.
            "epipolar_all_p50_px": epipolar_p50_native_px,
            "epipolar_all_p95_px": epipolar_p95_native_px,
            "baseline_summary": baseline_summary,
            "heldout_reprojection_p50_native_px": reprojection_p50_native_px,
            "heldout_reprojection_p95_native_px": reprojection_p95_native_px,
            "heldout_reprojection_p50_evaluation_px": _native_to_evaluation_px(
                reprojection_p50_native_px, native_pixels_per_evaluation_pixel
            ),
            "heldout_reprojection_p95_evaluation_px": _native_to_evaluation_px(
                reprojection_p95_native_px, native_pixels_per_evaluation_pixel
            ),
            # Schema-v1 compatibility aliases; these values are native pixels.
            "heldout_reprojection_p50_px": reprojection_p50_native_px,
            "heldout_reprojection_p95_px": reprojection_p95_native_px,
            "heldout_reprojection_observation_count": len(reprojection_errors),
            "triangulated_track_window_count": triangulated_track_count,
        },
        "worst_pair_medians": worst_pairs,
        "decision": "metrics_only_profile_not_yet_calibrated",
    }
