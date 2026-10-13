"""Camera-sequence adapters for held-out evaluation."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True, kw_only=True)
class CameraSequence:
    candidate_id: str
    frame_indices: np.ndarray
    c2w: np.ndarray
    intrinsics: np.ndarray
    plate_width: int
    plate_height: int
    adapter_metadata: dict[str, object]

    def __post_init__(self) -> None:
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        c2w = np.asarray(self.c2w, dtype=np.float64)
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        object.__setattr__(self, "frame_indices", frame_indices)
        object.__setattr__(self, "c2w", c2w)
        object.__setattr__(self, "intrinsics", intrinsics)
        if frame_indices.ndim != 1 or c2w.shape != (len(frame_indices), 4, 4):
            raise ValueError("camera frames and C2W poses must have shapes (N,) and (N,4,4)")
        if intrinsics.shape != (len(frame_indices), 4):
            raise ValueError("intrinsics must have shape (N,4) as fx,fy,cx,cy")
        if len(np.unique(frame_indices)) != len(frame_indices):
            raise ValueError("camera frame indices must be unique")
        if not np.isfinite(c2w).all() or not np.isfinite(intrinsics).all():
            raise ValueError("camera arrays must be finite")
        if np.any(intrinsics[:, :2] <= 0) or self.plate_width <= 0 or self.plate_height <= 0:
            raise ValueError("camera focal lengths and plate dimensions must be positive")

    def by_frame(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        return {
            int(frame_index): (pose, calibration)
            for frame_index, pose, calibration in zip(
                self.frame_indices,
                self.c2w,
                self.intrinsics,
                strict=True,
            )
        }


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as package:
        if key not in package:
            raise ValueError(f"{path} does not contain {key}")
        return np.asarray(package[key])


def load_vipe_camera(
    *,
    candidate_id: str,
    pose_path: Path,
    intrinsics_path: Path,
    plate_width: int,
    plate_height: int,
) -> CameraSequence:
    with np.load(pose_path, allow_pickle=False) as package:
        if not {"data", "inds"}.issubset(package.files):
            raise ValueError(f"Invalid ViPE pose artifact: {pose_path}")
        c2w = np.asarray(package["data"], dtype=np.float64)
        frame_indices = np.asarray(package["inds"], dtype=np.int64)
    with np.load(intrinsics_path, allow_pickle=False) as package:
        if "data" not in package:
            raise ValueError(f"Invalid ViPE intrinsics artifact: {intrinsics_path}")
        intrinsics = np.asarray(package["data"], dtype=np.float64)
        intrinsic_indices = (
            np.asarray(package["inds"], dtype=np.int64) if "inds" in package else frame_indices[: len(intrinsics)]
        )
    calibration_by_frame = {
        int(frame_index): calibration for frame_index, calibration in zip(intrinsic_indices, intrinsics, strict=True)
    }
    keep = np.array([int(frame_index) in calibration_by_frame for frame_index in frame_indices])
    frame_indices = frame_indices[keep]
    c2w = c2w[keep]
    aligned_intrinsics = np.stack([calibration_by_frame[int(frame_index)] for frame_index in frame_indices])
    return CameraSequence(
        candidate_id=candidate_id,
        frame_indices=frame_indices,
        c2w=c2w,
        intrinsics=aligned_intrinsics,
        plate_width=plate_width,
        plate_height=plate_height,
        adapter_metadata={
            "type": "vipe_npz",
            "pose_path": str(pose_path.resolve()),
            "intrinsics_path": str(intrinsics_path.resolve()),
            "coordinate_convention": "opencv_camera_to_world",
            "intrinsics_space": "native_plate",
        },
    )


def load_megasam_camera(
    *,
    candidate_id: str,
    package_path: Path,
    plate_width: int,
    plate_height: int,
) -> CameraSequence:
    with np.load(package_path, allow_pickle=False) as package:
        required = {"cam_c2w", "intrinsic", "images"}
        missing = required.difference(package.files)
        if missing:
            raise ValueError(f"MegaSAM package is missing {sorted(missing)}")
        c2w = np.asarray(package["cam_c2w"], dtype=np.float64)
        intrinsic = np.asarray(package["intrinsic"], dtype=np.float64)
        images = np.asarray(package["images"])
    if images.ndim != 4:
        raise ValueError(f"Unexpected MegaSAM images shape: {images.shape}")
    if images.shape[-1] == 3:
        processed_height, processed_width = images.shape[1:3]
    elif images.shape[1] == 3:
        processed_height, processed_width = images.shape[2:4]
    else:
        raise ValueError(f"Unexpected MegaSAM channel layout: {images.shape}")
    if intrinsic.shape == (3, 3):
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    elif intrinsic.shape == (4,):
        fx, fy, cx, cy = map(float, intrinsic)
    else:
        raise ValueError(f"Unexpected MegaSAM intrinsic shape: {intrinsic.shape}")

    # MegaSAM's demo resizes to a target area, then crops only the right/bottom
    # edges to multiples of eight. Its saved K retains the pre-crop center.
    pre_crop_width = int(round(2.0 * cx))
    pre_crop_height = int(round(2.0 * cy))
    if pre_crop_width < processed_width or pre_crop_height < processed_height:
        raise ValueError("MegaSAM principal point is inconsistent with its saved raster")
    scale_x = plate_width / pre_crop_width
    scale_y = plate_height / pre_crop_height
    plate_intrinsic = np.array([fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y])
    intrinsics = np.repeat(plate_intrinsic[None], len(c2w), axis=0)
    return CameraSequence(
        candidate_id=candidate_id,
        frame_indices=np.arange(len(c2w), dtype=np.int64),
        c2w=c2w,
        intrinsics=intrinsics,
        plate_width=plate_width,
        plate_height=plate_height,
        adapter_metadata={
            "type": "megasam_npz",
            "package_path": str(package_path.resolve()),
            "coordinate_convention": "opencv_camera_to_world",
            "processed_size": [processed_width, processed_height],
            "inferred_pre_crop_size": [pre_crop_width, pre_crop_height],
            "plate_scale": [scale_x, scale_y],
            "intrinsics_mapping": "centered resize followed by right/bottom multiple-of-eight crop",
        },
    )


_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
_SIMPLE_FOCAL_MODELS = {
    "SIMPLE_PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
}
_NATURAL_PARTS = re.compile(r"(\d+)")
_TRAILING_FRAME_NUMBER = re.compile(r"(\d+)$")


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    payload = handle.read(size)
    if len(payload) != size:
        raise EOFError("Unexpected end of COLMAP binary model")
    return payload


def _unpack(handle: BinaryIO, fmt: str) -> tuple:
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))


def _read_c_string(handle: BinaryIO) -> str:
    payload = bytearray()
    while True:
        value = _read_exact(handle, 1)
        if value == b"\x00":
            return payload.decode("utf-8")
        payload.extend(value)


def _qvec_rotation(qvec: tuple[float, ...]) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in _NATURAL_PARTS.split(value))


def _colmap_frame_indices(
    names: list[str],
    *,
    frame_number_origin: str,
) -> tuple[np.ndarray, int]:
    """Recover original source-frame indices from COLMAP image names.

    Enumerating registered images hides dropped frames. The trailing numeric
    token is preserved instead. Auto uses zero-origin when zero is present and
    otherwise ffmpeg's conventional one-origin sequence naming.
    """
    if frame_number_origin not in {"auto", "zero", "one"}:
        raise ValueError("frame_number_origin must be auto, zero, or one")
    frame_numbers = []
    for name in names:
        match = _TRAILING_FRAME_NUMBER.search(Path(name).stem)
        if match is None:
            raise ValueError(f"COLMAP image name has no trailing frame number: {name!r}")
        frame_numbers.append(int(match.group(1)))
    if not frame_numbers:
        raise ValueError("COLMAP reconstruction has no registered images")
    if frame_number_origin == "auto":
        resolved_origin = 0 if 0 in frame_numbers else 1
    else:
        resolved_origin = 0 if frame_number_origin == "zero" else 1
    frame_indices = np.asarray(frame_numbers, dtype=np.int64) - resolved_origin
    if np.any(frame_indices < 0):
        raise ValueError("COLMAP frame numbers are incompatible with the requested origin")
    if len(np.unique(frame_indices)) != len(frame_indices):
        raise ValueError("COLMAP image names map to duplicate source-frame indices")
    return frame_indices, resolved_origin


def load_colmap_camera(
    *,
    candidate_id: str,
    model_dir: Path,
    plate_width: int,
    plate_height: int,
    frame_number_origin: str = "auto",
) -> CameraSequence:
    model_dir = model_dir.resolve()
    cameras: dict[int, tuple[str, int, int, tuple[float, ...]]] = {}
    with (model_dir / "cameras.bin").open("rb") as handle:
        count = int(_unpack(handle, "<Q")[0])
        for _ in range(count):
            camera_id, model_id = _unpack(handle, "<Ii")
            width, height = _unpack(handle, "<QQ")
            if model_id not in _CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model ID {model_id}")
            model, parameter_count = _CAMERA_MODELS[model_id]
            params = tuple(float(value) for value in _unpack(handle, f"<{parameter_count}d"))
            cameras[int(camera_id)] = (model, int(width), int(height), params)

    images: list[tuple[str, int, tuple[float, ...], tuple[float, ...]]] = []
    with (model_dir / "images.bin").open("rb") as handle:
        count = int(_unpack(handle, "<Q")[0])
        for _ in range(count):
            values = _unpack(handle, "<I7dI")
            qvec = tuple(float(value) for value in values[1:5])
            tvec = tuple(float(value) for value in values[5:8])
            camera_id = int(values[8])
            name = _read_c_string(handle)
            point_count = int(_unpack(handle, "<Q")[0])
            handle.seek(point_count * 24, 1)
            images.append((name, camera_id, qvec, tvec))
    images.sort(key=lambda row: _natural_key(row[0]))
    frame_indices, resolved_origin = _colmap_frame_indices(
        [row[0] for row in images],
        frame_number_origin=frame_number_origin,
    )

    poses: list[np.ndarray] = []
    calibrations: list[np.ndarray] = []
    source_sizes: set[tuple[int, int]] = set()
    camera_models: set[str] = set()
    for _, camera_id, qvec, tvec in images:
        model, width, height, params = cameras[camera_id]
        camera_models.add(model)
        source_sizes.add((width, height))
        if model in _SIMPLE_FOCAL_MODELS:
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            fx, fy, cx, cy = params[:4]
        calibrations.append(
            np.array(
                [
                    fx * plate_width / width,
                    fy * plate_height / height,
                    cx * plate_width / width,
                    cy * plate_height / height,
                ],
                dtype=np.float64,
            )
        )
        rotation_w2c = _qvec_rotation(qvec)
        rotation_c2w = rotation_w2c.T
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation_c2w
        pose[:3, 3] = -(rotation_c2w @ np.asarray(tvec))
        poses.append(pose)
    return CameraSequence(
        candidate_id=candidate_id,
        frame_indices=frame_indices,
        c2w=np.stack(poses),
        intrinsics=np.stack(calibrations),
        plate_width=plate_width,
        plate_height=plate_height,
        adapter_metadata={
            "type": "colmap_binary",
            "model_dir": str(model_dir),
            "coordinate_convention": "colmap_opencv_camera_to_world",
            "camera_models": sorted(camera_models),
            "source_sizes": sorted([list(value) for value in source_sizes]),
            "frame_index_policy": "trailing numeric token from registered image name",
            "requested_frame_number_origin": frame_number_origin,
            "resolved_frame_number_origin": resolved_origin,
            "registered_frame_indices": frame_indices.tolist(),
            "distortion_policy": "pinhole terms only; input is expected to be undistorted",
        },
    )
