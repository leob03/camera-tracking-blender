"""Original artist cross and supported four-witness placement from study schema 8.

Ported from camera-tracking-blender/scripts/augment_audit_review_v2.py.
The selector requires actual track observations and camera-specific triangulated
landmarks. Native cloud projections must not be represented as observations.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import bpy
import numpy as np
from mathutils import Matrix


def slug(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


MINIMUM_TRIANGULATION_ANGLE_DEG = 0.2


PROBE_REVIEW_EXTENT_PX = 80.0


DEFAULT_VERY_FOREGROUND_MAX_DEPTH_QUANTILE = 0.15


DEPTH_LAYER_COLORS = np.asarray(
    [
        [0.95, 0.08, 0.04],
        [0.08, 0.82, 0.12],
        [0.04, 0.28, 0.95],
    ],
    dtype=np.float32,
)


DEPTH_LAYER_NAMES = ("foreground", "midground", "background")


def landmark_geometry(
    *,
    xyz: np.ndarray,
    observation_offsets: np.ndarray,
    observation_frames: np.ndarray,
    frame_indices: np.ndarray,
    c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pose_by_frame = {
        int(frame): pose
        for frame, pose in zip(frame_indices, c2w, strict=True)
    }
    w2c_by_frame = {
        frame: np.linalg.inv(pose)
        for frame, pose in pose_by_frame.items()
    }
    depths = np.full(len(xyz), np.nan, dtype=np.float64)
    angles = np.full(len(xyz), np.nan, dtype=np.float64)
    for index, point in enumerate(xyz):
        start = int(observation_offsets[index])
        stop = int(observation_offsets[index + 1])
        frames = [
            int(frame)
            for frame in observation_frames[start:stop]
            if int(frame) in pose_by_frame
        ]
        if not frames:
            continue
        point_h = np.r_[point, 1.0]
        positive_depths = [
            float((w2c_by_frame[frame] @ point_h)[2])
            for frame in frames
            if float((w2c_by_frame[frame] @ point_h)[2]) > 0
        ]
        if positive_depths:
            depths[index] = float(np.median(positive_depths))
        first_ray = point - pose_by_frame[frames[0]][:3, 3]
        last_ray = point - pose_by_frame[frames[-1]][:3, 3]
        denominator = float(np.linalg.norm(first_ray) * np.linalg.norm(last_ray))
        if denominator > 1e-12:
            cosine = float(np.dot(first_ray, last_ray) / denominator)
            angles[index] = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return depths, angles


def conditioned_landmarks(
    depths: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(depths) & (depths > 0) & np.isfinite(angles)
    keep = finite & (angles >= MINIMUM_TRIANGULATION_ANGLE_DEG)
    minimum_count = min(64, int(np.count_nonzero(finite)))
    if np.count_nonzero(keep) < minimum_count:
        ranked = np.flatnonzero(finite)[np.argsort(angles[finite])[::-1]]
        keep[ranked[:minimum_count]] = True
    return keep


def depth_layers(depths: np.ndarray) -> np.ndarray:
    low, high = np.quantile(depths, [1.0 / 3.0, 2.0 / 3.0])
    return np.where(depths <= low, 0, np.where(depths <= high, 1, 2)).astype(np.int32)


PIN_LAYOUT = (
    # The caller can request a closer real track without moving its 3D anchor.
    (0, "very_foreground_left", 0.24, 0.72, 0.20, 0.07, 0.15),
    # The second foreground witness targets the rest of the foreground third.
    (0, "foreground_right", 0.72, 0.68, 0.76, 0.25, None),
    (1, "midground_right", 0.68, 0.52, 0.45, 0.50, None),
    (2, "background_left", 0.31, 0.32, 0.65, 0.84, None),
)


def select_screen_safe_track_pins(
    *,
    selected: np.ndarray,
    depths: np.ndarray,
    layers: np.ndarray,
    spans: np.ndarray,
    median_errors: np.ndarray,
    landmark_ids: np.ndarray,
    observation_offsets: np.ndarray,
    observation_frames: np.ndarray,
    observed_xy: np.ndarray,
    observation_roles: np.ndarray,
    positive_depth: np.ndarray,
    frame_indices: np.ndarray,
    plate_width: int,
    plate_height: int,
    very_foreground_max_depth_quantile: float,
) -> list[dict[str, object]]:
    """Pick four real landmarks, including one deliberately very-near witness."""
    registered = {int(frame) for frame in frame_indices}
    first_frame = float(frame_indices[0])
    frame_span = max(float(frame_indices[-1] - frame_indices[0]), 1.0)
    finite_errors = median_errors[np.isfinite(median_errors)]
    error_scale = max(
        float(np.percentile(finite_errors, 75)) if len(finite_errors) else 1.0,
        0.25,
    )
    maximum_span = max(float(np.max(spans[selected])), 1.0)
    depth_order = np.argsort(depths[selected], kind="stable")
    depth_percentiles = np.empty(len(selected), dtype=np.float64)
    depth_percentiles[depth_order] = (
        np.arange(len(selected), dtype=np.float64) + 0.5
    ) / float(len(selected))
    very_foreground_depth_threshold = float(
        np.quantile(depths[selected], very_foreground_max_depth_quantile)
    )
    used: set[int] = set()
    result: list[dict[str, object]] = []

    for (
        layer_index,
        slot,
        target_x,
        target_y,
        target_time,
        target_depth_percentile,
        maximum_depth_percentile,
    ) in PIN_LAYOUT:
        if slot.startswith("very_foreground"):
            target_depth_percentile = 0.5 * very_foreground_max_depth_quantile
            maximum_depth_percentile = very_foreground_max_depth_quantile
        best: tuple[float, int, int, int] | None = None
        for minimum_margin in (0.12, 0.08, 0.04):
            for chosen_local in np.flatnonzero(layers == layer_index):
                source_index = int(selected[chosen_local])
                if source_index in used:
                    continue
                depth_percentile = float(depth_percentiles[chosen_local])
                if (
                    maximum_depth_percentile is not None
                    and (
                        depth_percentile > maximum_depth_percentile
                        or float(depths[source_index]) > very_foreground_depth_threshold
                    )
                ):
                    continue
                start = int(observation_offsets[source_index])
                stop = int(observation_offsets[source_index + 1])
                for row in range(start, stop):
                    frame = int(observation_frames[row])
                    xy = observed_xy[row]
                    if (
                        frame not in registered
                        or int(observation_roles[row]) != 1
                        or not bool(positive_depth[row])
                        or not np.isfinite(xy).all()
                    ):
                        continue
                    x = float(xy[0]) / float(plate_width)
                    y = float(xy[1]) / float(plate_height)
                    margin = min(x, 1.0 - x, y, 1.0 - y)
                    if margin < minimum_margin:
                        continue
                    screen_cost = float(np.hypot(x - target_x, y - target_y))
                    time = (float(frame) - first_frame) / frame_span
                    time_cost = abs(time - target_time)
                    error_cost = min(
                        float(median_errors[source_index]) / error_scale,
                        4.0,
                    )
                    span_reward = min(float(spans[source_index]) / maximum_span, 1.0)
                    depth_cost = abs(depth_percentile - target_depth_percentile)
                    score = (
                        screen_cost
                        + 0.18 * time_cost
                        + 0.06 * error_cost
                        + 0.24 * depth_cost
                        - 0.18 * span_reward
                    )
                    candidate = (score, int(landmark_ids[source_index]), source_index, row)
                    if best is None or candidate < best:
                        best = candidate
            if best is not None:
                break
        if best is None:
            continue
        _, _, source_index, row = best
        used.add(source_index)
        result.append(
            {
                "slot": slot,
                "layer_index": layer_index,
                "layer_name": DEPTH_LAYER_NAMES[layer_index],
                "source_index": source_index,
                "chosen_local": int(np.flatnonzero(selected == source_index)[0]),
                "reference_frame": int(observation_frames[row]),
                "reference_xy": [float(value) for value in observed_xy[row]],
                "depth_percentile": float(
                    depth_percentiles[int(np.flatnonzero(selected == source_index)[0])]
                ),
                "very_foreground_depth_threshold_candidate_units": very_foreground_depth_threshold,
                "proximity_class": (
                    "very_foreground"
                    if slot.startswith("very_foreground")
                    else DEPTH_LAYER_NAMES[layer_index]
                ),
            }
        )
    return result


def create_depth_pin(
    *,
    collection: bpy.types.Collection,
    parent: bpy.types.Object,
    label: str,
    slot: str,
    layer_name: str,
    point: np.ndarray,
    landmark_id: int,
    proximity_class: str,
    depth_quantile_0_1: float,
    very_foreground_depth_threshold: float,
    depth: float,
    angle: float,
    display_radius: float,
    reference_frame: int,
    reference_xy: list[float],
) -> bpy.types.Object:
    name = f"AUDIT_{slug(label)}_{slug(slot)}_TRACK_PIN"
    pin = bpy.data.objects.new(name, None)
    pin.empty_display_type = "PLAIN_AXES"
    pin.empty_display_size = max(float(display_radius) * 6.0, 0.0001)
    pin.location = tuple(float(value) for value in point)
    pin.parent = parent
    pin.show_name = False
    pin.show_in_front = True
    pin.color = (*DEPTH_LAYER_COLORS[DEPTH_LAYER_NAMES.index(layer_name)], 1.0)
    pin["classification"] = "SCREEN_SAFE_SUPPORTED_LANDMARK_PIN_NOT_A_SURFACE"
    pin["depth_layer"] = layer_name
    pin["proximity_class"] = proximity_class
    pin["selected_median_depth_quantile_0_1"] = float(depth_quantile_0_1)
    pin["very_foreground_depth_threshold_candidate_units"] = float(very_foreground_depth_threshold)
    pin["landmark_id"] = str(int(landmark_id))
    pin["median_camera_depth"] = float(depth)
    pin["triangulation_angle_deg"] = float(angle)
    pin["reference_frame_zero_based"] = int(reference_frame)
    pin["reference_observation_xy_px"] = reference_xy
    pin["minimum_reference_frame_margin_fraction"] = 0.04
    pin["artist_editable"] = True
    collection.objects.link(pin)
    return pin


def load_directional_proxy_asset(asset_path: Path) -> dict[str, object]:
    """Import mesh-only GLB content and retain self-contained mesh templates."""
    asset_path = asset_path.resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)

    objects_before = set(bpy.data.objects)
    result = bpy.ops.import_scene.gltf(filepath=str(asset_path))
    if "FINISHED" not in result:
        raise RuntimeError(f"failed to import directional proxy asset: {asset_path}")
    imported_objects = [
        obj for obj in bpy.data.objects if obj not in objects_before
    ]

    parts: list[tuple[str, bpy.types.Mesh]] = []
    all_vertices: list[np.ndarray] = []
    for imported in imported_objects:
        if imported.type != "MESH":
            continue
        mesh = imported.data.copy()
        mesh.name = f"AUDIT_AXIS_ASSET_{slug(imported.name)}_MeshTemplate"
        mesh.transform(imported.matrix_world)
        parts.append((imported.name, mesh))
        all_vertices.extend(
            np.asarray(vertex.co[:], dtype=np.float64) for vertex in mesh.vertices
        )

    for imported in imported_objects:
        bpy.data.objects.remove(imported, do_unlink=True)

    if not parts or not all_vertices:
        raise ValueError(f"directional proxy asset contains no mesh geometry: {asset_path}")

    coordinates = np.stack(all_vertices)
    bbox_min = coordinates.min(axis=0)
    bbox_max = coordinates.max(axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_dimensions = bbox_max - bbox_min
    center_transform = Matrix.Translation(
        tuple(float(value) for value in -bbox_center)
    )
    for _, mesh in parts:
        mesh.transform(center_transform)

    return {
        "path": asset_path,
        "sha256": sha256_file(asset_path),
        "parts": parts,
        "part_names": [name for name, _ in parts],
        "bbox_dimensions": bbox_dimensions,
    }


def create_supported_probe(
    *,
    collection: bpy.types.Collection,
    parent: bpy.types.Object,
    label: str,
    slot: str,
    layer_name: str,
    proximity_class: str,
    point: np.ndarray,
    landmark_id: int,
    reference_frame: int,
    reference_c2w: np.ndarray,
    camera_depth: float,
    depth_percentile: float,
    focal_x: float,
    focal_y: float,
    plate_width: int,
    plate_height: int,
    proxy_asset: dict[str, object],
    very_foreground_max_depth_quantile: float,
) -> bpy.types.Object:
    """Place the artist-authored axis proxy on a real track hypothesis."""
    right = np.asarray(reference_c2w[:3, 0], dtype=np.float64)
    up = -np.asarray(reference_c2w[:3, 1], dtype=np.float64)
    forward = np.asarray(reference_c2w[:3, 2], dtype=np.float64)
    asset_dimensions = np.asarray(proxy_asset["bbox_dimensions"], dtype=np.float64)
    image_plane_extent = max(float(asset_dimensions[0]), float(asset_dimensions[2]))
    if not np.isfinite(image_plane_extent) or image_plane_extent <= 0:
        raise ValueError("directional proxy asset has invalid X/Z dimensions")
    scene = bpy.context.scene
    render_scale = float(scene.render.resolution_percentage) / 100.0
    render_width = float(scene.render.resolution_x) * render_scale
    render_height = float(scene.render.resolution_y) * render_scale
    display_focal_x = float(focal_x) * render_width / float(plate_width)
    display_focal_y = float(focal_y) * render_height / float(plate_height)
    display_focal = float(np.sqrt(display_focal_x * display_focal_y))
    scale = float(
        camera_depth * PROBE_REVIEW_EXTENT_PX / display_focal / image_plane_extent
    )

    object_name = f"AUDIT_{slug(label)}_{slug(slot)}_AXIS_PROXY"
    probe = bpy.data.objects.new(object_name, None)
    probe.empty_display_type = "PLAIN_AXES"
    probe.empty_display_size = max(scale * 0.05, 0.0001)
    probe.show_name = False
    probe.show_in_front = True
    probe.parent = parent
    probe.matrix_parent_inverse = Matrix.Identity(4)
    basis = Matrix(np.column_stack((right, forward, up)).tolist()).to_4x4()
    probe.matrix_basis = (
        Matrix.Translation(tuple(float(value) for value in point))
        @ basis
        @ Matrix.Scale(scale, 4)
    )
    collection.objects.link(probe)

    part_objects: list[str] = []
    for part_name, template_mesh in proxy_asset["parts"]:
        child_name = f"{object_name}_{slug(part_name)}"
        child = bpy.data.objects.new(child_name, template_mesh.copy())
        child.parent = probe
        child.matrix_parent_inverse = Matrix.Identity(4)
        child.matrix_basis = Matrix.Identity(4)
        child.show_name = False
        child.show_in_front = True
        child["classification"] = "EMBEDDED_ARTIST_AXIS_PROXY_PART"
        child["source_asset_object"] = str(part_name)
        collection.objects.link(child)
        part_objects.append(child.name)

    probe["classification"] = (
        "AUTO_TRACK_ANCHORED_ARTIST_AXIS_ORIENTATION_PROXY_NOT_GROUND_TRUTH"
    )
    probe["proxy_shape"] = "USER_SUPPLIED_AXIS_SHAPED_CROSS_GLB"
    probe["rotational_symmetry"] = False
    probe["orientation_ground_truth"] = False
    probe["reference_alignment"] = (
        "asset_local_x_to_camera_right_y_to_camera_forward_z_to_camera_up"
    )
    probe["depth_layer"] = layer_name
    probe["proximity_class"] = proximity_class
    probe["landmark_id"] = str(int(landmark_id))
    probe["anchor_world_xyz"] = [float(value) for value in point]
    probe["reference_frame_zero_based"] = int(reference_frame)
    probe["reference_camera_depth"] = float(camera_depth)
    probe["selected_median_depth_quantile_0_1"] = float(depth_percentile)
    probe["very_foreground_max_depth_quantile_0_1"] = float(
        very_foreground_max_depth_quantile
    )
    probe["absolute_near_depth_claim"] = False
    probe["near_classification"] = "relative_to_conditioned_candidate_camera_gauge"
    probe["target_review_render_extent_px_at_reference"] = PROBE_REVIEW_EXTENT_PX
    probe["review_render_resolution_px"] = [int(render_width), int(render_height)]
    probe["uniform_asset_scale_world_per_asset_unit"] = scale
    probe["depth_extent_world"] = float(asset_dimensions[1] * scale)
    probe["asset_source"] = str(proxy_asset["path"])
    probe["asset_source_sha256"] = str(proxy_asset["sha256"])
    probe["asset_part_objects_json"] = json.dumps(part_objects)
    probe["asset_embedded_in_blend"] = True
    probe["placement"] = "exact_object_origin_at_conditioned_track_hypothesis"
    probe["surface_or_ground_guarantee"] = False
    probe["artist_editable"] = True
    return probe
