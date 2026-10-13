"""Generate camera-free, partitioned TAPIR point-track evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PINNED_TAPNET_COMMIT = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
BOOTSTAPIR_CHECKPOINT_SHA256 = "8493c7a69e02c85b9382fbb3c7b8b539b36bc08ede744b9e99feb739a0129f4b"
DEEPLABV3_MOBILENET_V3_LARGE_SHA256 = "fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70"
TRACK_EVIDENCE_SCHEMA_VERSION = 2
DYNAMIC_CATEGORIES = frozenset(
    {
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bus",
        "car",
        "cat",
        "cow",
        "dog",
        "horse",
        "motorbike",
        "person",
        "sheep",
        "train",
    }
)


@dataclass(frozen=True, kw_only=True)
class TapirTrackConfig:
    max_dimension: int = 512
    window_frames: int = 32
    stride_frames: int = 16
    minimum_tail_frames: int = 17
    query_step_pixels: int = 16
    query_chunk_size: int = 64
    visibility_threshold: float = 0.5
    semantic_masking: bool = True
    semantic_mask_batch_size: int = 8
    semantic_mask_dilation_pixels: int = 5

    def __post_init__(self) -> None:
        if self.max_dimension < 64:
            raise ValueError("max_dimension must be at least 64")
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least 2")
        if not 1 <= self.stride_frames <= self.window_frames:
            raise ValueError("stride_frames must be in [1, window_frames]")
        if not 2 <= self.minimum_tail_frames <= self.window_frames:
            raise ValueError("minimum_tail_frames must be in [2, window_frames]")
        if self.query_step_pixels < 2 or self.query_chunk_size < 1:
            raise ValueError("query step and query chunk size must be positive")
        if not 0.0 < self.visibility_threshold < 1.0:
            raise ValueError("visibility_threshold must be in (0, 1)")
        if self.semantic_mask_batch_size < 1:
            raise ValueError("semantic mask batch size must be positive")
        if self.semantic_mask_dilation_pixels < 0:
            raise ValueError("semantic mask dilation must not be negative")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def window_starts(frame_count: int, config: TapirTrackConfig) -> list[int]:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if frame_count <= config.window_frames:
        return [0]
    starts = list(
        range(
            0,
            max(frame_count - config.minimum_tail_frames + 1, 1),
            config.stride_frames,
        )
    )
    final_start = frame_count - config.window_frames
    starts.append(final_start)
    return sorted(set(start for start in starts if start + config.minimum_tail_frames <= frame_count))


def evaluation_size(width: int, height: int, max_dimension: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    scale = min(1.0, max_dimension / max(width, height))
    eval_width = max(8, int(round(width * scale / 8.0)) * 8)
    eval_height = max(8, int(round(height * scale / 8.0)) * 8)
    return eval_width, eval_height


def query_grid(
    width: int,
    height: int,
    *,
    step: int,
    window_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(step // 2, width, step, dtype=np.float32)
    ys = np.arange(step // 2, height, step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    ix, iy = np.meshgrid(np.arange(len(xs), dtype=np.int64), np.arange(len(ys), dtype=np.int64))
    query_points = np.column_stack(
        [
            np.zeros(grid_x.size, dtype=np.float32),
            grid_y.ravel(),
            grid_x.ravel(),
        ]
    )
    query_ids = (np.int64(window_start) << np.int64(40)) | (iy.ravel() << np.int64(20)) | ix.ravel()
    mixed = (
        ix.ravel() * np.int64(73856093) ^ iy.ravel() * np.int64(19349663) ^ np.int64(window_start) * np.int64(83492791)
    )
    partitions = (mixed & np.int64(1)).astype(np.uint8)
    return query_points, query_ids, partitions


def _decode_video(path: Path, max_dimension: int) -> tuple[np.ndarray, dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Could not open source video: {path}")
    native_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    eval_width, eval_height = evaluation_size(native_width, native_height, max_dimension)
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if (eval_width, eval_height) != (native_width, native_height):
            frame_rgb = cv2.resize(
                frame_rgb,
                (eval_width, eval_height),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(frame_rgb)
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    video = np.stack(frames)
    metadata = {
        "native_width": native_width,
        "native_height": native_height,
        "evaluation_width": eval_width,
        "evaluation_height": eval_height,
        "frame_count": len(video),
        "fps": fps,
        "evaluation_to_native_scale_x": native_width / eval_width,
        "evaluation_to_native_scale_y": native_height / eval_height,
    }
    return video, metadata
def _sample_mask(mask: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    x = np.rint(points_xy[:, 0]).astype(np.int64)
    y = np.rint(points_xy[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    result = np.zeros(len(points_xy), dtype=bool)
    result[inside] = mask[y[inside], x[inside]] > 0
    return result


def _build_semantic_static_masks(
    images: np.ndarray,
    *,
    device: str,
    batch_size: int,
    dilation_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torchvision.models.segmentation import (
        DeepLabV3_MobileNet_V3_Large_Weights,
        deeplabv3_mobilenet_v3_large,
    )

    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    checkpoint = Path(torch.hub.get_dir()) / 'checkpoints' / Path(weights.url).name
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Preinstall the DeepLab checkpoint: {checkpoint}')
    model = deeplabv3_mobilenet_v3_large(weights=weights).eval().to(device)
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != DEEPLABV3_MOBILENET_V3_LARGE_SHA256:
        raise ValueError(
            "DeepLabV3 MobileNet V3 Large checkpoint mismatch: "
            f"{checkpoint_sha256} != {DEEPLABV3_MOBILENET_V3_LARGE_SHA256}"
        )
    categories = list(weights.meta["categories"])
    dynamic_names = sorted(DYNAMIC_CATEGORIES)
    dynamic_ids = torch.tensor(
        [categories.index(name) for name in dynamic_names],
        dtype=torch.long,
        device=device,
    )
    transform = weights.transforms()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_pixels * 2 + 1, dilation_pixels * 2 + 1),
    )
    masks: list[np.ndarray] = []
    class_counts = {name: 0 for name in dynamic_names}
    total_pixels = 0
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch_np = images[start : start + batch_size]
            batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float() / 255.0
            transformed = transform(batch).to(device)
            logits = model(transformed)["out"]
            logits = torch.nn.functional.interpolate(
                logits,
                size=images.shape[1:3],
                mode="bilinear",
                align_corners=False,
            )
            labels = logits.argmax(dim=1)
            dynamic = torch.isin(labels, dynamic_ids).cpu().numpy()
            label_np = labels.cpu().numpy()
            for local_index, frame_mask in enumerate(dynamic):
                expanded = cv2.dilate(frame_mask.astype(np.uint8), kernel) > 0
                masks.append(~expanded)
                total_pixels += int(frame_mask.size)
                frame_labels = label_np[local_index]
                for name in class_counts:
                    class_counts[name] += int(
                        np.count_nonzero(frame_labels == categories.index(name))
                    )
            del batch, transformed, logits, labels
    static_masks = np.stack(masks).astype(np.uint8)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return static_masks, {
        "enabled": True,
        "model": "deeplabv3_mobilenet_v3_large",
        "weights": str(weights),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "dynamic_categories": dynamic_names,
        "dynamic_fraction_before_dilation": float(
            sum(class_counts.values()) / max(total_pixels, 1)
        ),
        "static_fraction_after_dilation": float(static_masks.mean()),
        "dynamic_class_pixel_counts": class_counts,
        "dilation_pixels_at_evaluation_resolution": dilation_pixels,
        "candidate_camera_inputs_used": False,
    }


def _git_commit(repository: Path) -> str:
    # Resolve only explicit Git metadata paths; no Git process or repository scan.
    git_dir = repository / '.git'
    if git_dir.is_file():
        git_dir = (repository / git_dir.read_text().strip().split(': ', 1)[1]).resolve()
    head = (git_dir / 'HEAD').read_text().strip()
    if not head.startswith('ref: '):
        return head
    ref = head[5:]
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text().strip()
    with (git_dir / 'packed-refs').open() as stream:
        for line in stream:
            fields = line.strip().split()
            if len(fields) == 2 and fields[1] == ref:
                return fields[0]
    raise ValueError('Cannot resolve the pinned CoTracker revision')


def _load_model(tapnet_repository: Path, checkpoint: Path, device: str):
    import torch

    source_root = str(tapnet_repository.resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from tapnet.torch import tapir_model

    model = tapir_model.TAPIR(pyramid_level=1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval().to(device)


def _track_window(
    model,
    frames: np.ndarray,
    query_points: np.ndarray,
    *,
    query_chunk_size: int,
    visibility_threshold: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    video_tensor = torch.from_numpy(frames).to(device=device, dtype=torch.float32)
    video_tensor = video_tensor / 255.0 * 2.0 - 1.0
    query_tensor = torch.from_numpy(query_points).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = model(
            video_tensor[None],
            query_tensor[None],
            query_chunk_size=query_chunk_size,
        )
        tracks = output["tracks"][0]
        visibility_score = (1.0 - torch.sigmoid(output["occlusion"][0])) * (
            1.0 - torch.sigmoid(output["expected_dist"][0])
        )
        visible = visibility_score > visibility_threshold
    result = (
        tracks.detach().cpu().numpy().astype(np.float32),
        visible.detach().cpu().numpy().astype(np.uint8),
        visibility_score.detach().cpu().numpy().astype(np.float16),
    )
    del output, tracks, visible, visibility_score, video_tensor, query_tensor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def generate_track_evidence(
    *,
    source_video: Path,
    output_dir: Path,
    tapnet_repository: Path,
    checkpoint: Path,
    config: TapirTrackConfig,
    expected_tapnet_commit: str = PINNED_TAPNET_COMMIT,
    expected_checkpoint_sha256: str = BOOTSTAPIR_CHECKPOINT_SHA256,
    device: str = "cuda",
) -> dict[str, Any]:
    source_video = source_video.resolve()
    output_dir = output_dir.resolve()
    tapnet_repository = tapnet_repository.resolve()
    checkpoint = checkpoint.resolve()
    for path in (source_video, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (tapnet_repository / "tapnet" / "torch" / "tapir_model.py").is_file():
        raise FileNotFoundError(f"Not a TAPNet source checkout: {tapnet_repository}")
    actual_commit = _git_commit(tapnet_repository)
    if actual_commit != expected_tapnet_commit:
        raise ValueError(f"TAPNet commit mismatch: {actual_commit} != {expected_tapnet_commit}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(f"BootsTAPIR checkpoint mismatch: {checkpoint_sha256} != {expected_checkpoint_sha256}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evidence directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, video_metadata = _decode_video(source_video, config.max_dimension)
    if config.semantic_masking:
        static_masks, semantic_mask_receipt = _build_semantic_static_masks(
            frames,
            device=device,
            batch_size=config.semantic_mask_batch_size,
            dilation_pixels=config.semantic_mask_dilation_pixels,
        )
    else:
        static_masks = np.ones(frames.shape[:3], dtype=np.uint8)
        semantic_mask_receipt = {
            "enabled": False,
            "reason": "explicitly_disabled",
            "static_fraction_after_dilation": 1.0,
            "candidate_camera_inputs_used": False,
        }
    mask_path = output_dir / "STATIC_MASKS.npz"
    np.savez_compressed(
        mask_path,
        schema_version=np.array(TRACK_EVIDENCE_SCHEMA_VERSION, dtype=np.int32),
        static_masks=static_masks,
    )
    semantic_mask_receipt.update(
        {
            "path": mask_path.name,
            "sha256": sha256_file(mask_path),
            "format": [int(static_masks.shape[2]), int(static_masks.shape[1])],
            "frame_count": int(len(static_masks)),
        }
    )
    model = _load_model(tapnet_repository, checkpoint, device)
    native_scale = np.array(
        [
            video_metadata["evaluation_to_native_scale_x"],
            video_metadata["evaluation_to_native_scale_y"],
        ],
        dtype=np.float32,
    )
    window_receipts: list[dict[str, Any]] = []
    for start in window_starts(len(frames), config):
        end = min(start + config.window_frames, len(frames))
        query_points, query_ids, partitions = query_grid(
            video_metadata["evaluation_width"],
            video_metadata["evaluation_height"],
            step=config.query_step_pixels,
            window_start=start,
        )
        query_count_before_semantic_mask = len(query_points)
        query_static = _sample_mask(static_masks[start], query_points[:, [2, 1]])
        query_points = query_points[query_static]
        query_ids = query_ids[query_static]
        partitions = partitions[query_static]
        tracks_eval, visible, visibility_score = _track_window(
            model,
            frames[start:end],
            query_points,
            query_chunk_size=config.query_chunk_size,
            visibility_threshold=config.visibility_threshold,
            device=device,
        )
        tracker_visible = visible.astype(bool)
        if tracks_eval.ndim != 3 or tracks_eval.shape[0] != len(query_points):
            raise ValueError(f"Unexpected TAPIR track layout: {tracks_eval.shape}")
        static_valid = np.stack(
            [
                _sample_mask(static_masks[start + offset], tracks_eval[:, offset])
                for offset in range(end - start)
            ],
            axis=1,
        )
        visible = (tracker_visible & static_valid).astype(np.uint8)
        tracks_native = tracks_eval * native_scale[None, None, :]
        query_xy_native = query_points[:, [2, 1]] * native_scale[None, :]
        output_path = output_dir / f"window_{start:06d}_{end:06d}.npz"
        np.savez_compressed(
            output_path,
            schema_version=np.array(TRACK_EVIDENCE_SCHEMA_VERSION, dtype=np.int32),
            start_frame=np.array(start, dtype=np.int32),
            end_frame_exclusive=np.array(end, dtype=np.int32),
            source_frame_indices=np.arange(start, end, dtype=np.int32),
            query_ids=query_ids,
            query_xy_native=query_xy_native.astype(np.float32),
            partitions=partitions,
            tracks_xy_native=tracks_native.astype(np.float32),
            visible=visible,
            tracker_visible=tracker_visible.astype(np.uint8),
            static_valid=static_valid.astype(np.uint8),
            visibility_score=visibility_score,
        )
        window_receipts.append(
            {
                "path": output_path.name,
                "sha256": sha256_file(output_path),
                "start_frame": start,
                "end_frame_exclusive": end,
                "query_count_before_semantic_mask": query_count_before_semantic_mask,
                "query_count": len(query_points),
                "semantic_static_query_fraction": float(query_static.mean()),
                "routing_query_count": int(np.count_nonzero(partitions == 0)),
                "final_audit_query_count": int(np.count_nonzero(partitions == 1)),
                "tracker_visible_fraction": float(tracker_visible.mean()),
                "rigid_visible_fraction": float(visible.mean()),
                "visible_fraction": float(visible.mean()),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": TRACK_EVIDENCE_SCHEMA_VERSION,
        "status": "sealed_camera_free_track_evidence",
        "source": {
            "path": str(source_video),
            "sha256": sha256_file(source_video),
            **video_metadata,
        },
        "tracker": {
            "name": "bootstapir_pytorch",
            "license": "Apache-2.0",
            "tapnet_repository": str(tapnet_repository),
            "tapnet_commit": actual_commit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "device": device,
        },
        "semantic_static_mask": semantic_mask_receipt,
        "config": asdict(config),
        "partition_policy": {
            "0": "routing_validation; may guide retry selection",
            "1": "final_audit; forbidden until a candidate is locked",
            "assignment": "stable hash of query grid coordinates and window start",
        },
        "candidate_camera_inputs_used": False,
        "windows": window_receipts,
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
