"""Generate camera-free, partitioned CoTracker evidence.

This adapter intentionally writes the same sealed schema as the BootsTAPIR
builder. Candidate cameras are never read here; the two trackers remain
separate evidence recipes with explicit code, checkpoint, and license
provenance.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from camera_evidence.tapir_tracks import (
    TRACK_EVIDENCE_SCHEMA_VERSION,
    _build_semantic_static_masks,
    _decode_video,
    _git_commit,
    _sample_mask,
    query_grid,
    sha256_file,
)

PINNED_COTRACKER_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
COTRACKER_SCALED_OFFLINE_SHA256 = "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"


@dataclass(frozen=True, kw_only=True)
class CoTrackerTrackConfig:
    max_dimension: int = 608
    window_frames: int = 32
    stride_frames: int = 16
    minimum_tail_frames: int = 17
    query_step_pixels: int = 8
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
        if self.query_step_pixels < 2:
            raise ValueError("query_step_pixels must be at least 2")
        if self.semantic_mask_batch_size < 1:
            raise ValueError("semantic mask batch size must be positive")
        if self.semantic_mask_dilation_pixels < 0:
            raise ValueError("semantic mask dilation must not be negative")


def cotracker_window_starts(frame_count: int, config: CoTrackerTrackConfig) -> list[int]:
    """Match Robin's overlapping-window recipe, including the final tail."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    starts = list(range(0, max(frame_count - config.minimum_tail_frames + 1, 1), config.stride_frames))
    final_start = max(0, frame_count - config.window_frames)
    starts.append(final_start)
    return sorted(
        set(
            start
            for start in starts
            if start + config.minimum_tail_frames <= frame_count or start == final_start
        )
    )


def _load_model(repository: Path, checkpoint: Path, device: str):
    source_root = str(repository.resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from cotracker.predictor import CoTrackerPredictor

    return CoTrackerPredictor(
        checkpoint=str(checkpoint),
        offline=True,
        v2=False,
    ).eval().to(device)


def _track_window(model, frames: np.ndarray, queries_txy: np.ndarray, *, device: str):
    import torch

    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].to(device=device, dtype=torch.float32)
    queries = torch.from_numpy(queries_txy[None]).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        tracks, visibility = model(video, queries=queries, backward_tracking=False)
    tracks_np = tracks[0].detach().cpu().numpy().astype(np.float32).transpose(1, 0, 2)
    visible_np = visibility[0].detach().cpu().numpy().astype(np.uint8).transpose(1, 0)
    result = tracks_np, visible_np, visible_np.astype(np.float16)
    del video, queries, tracks, visibility
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def generate_cotracker_evidence(
    *,
    source_video: Path,
    output_dir: Path,
    cotracker_repository: Path,
    checkpoint: Path,
    config: CoTrackerTrackConfig,
    expected_cotracker_commit: str = PINNED_COTRACKER_COMMIT,
    expected_checkpoint_sha256: str = COTRACKER_SCALED_OFFLINE_SHA256,
    device: str = "cuda",
) -> dict[str, Any]:
    source_video = source_video.resolve()
    output_dir = output_dir.resolve()
    cotracker_repository = cotracker_repository.resolve()
    checkpoint = checkpoint.resolve()
    for path in (source_video, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (cotracker_repository / "cotracker" / "predictor.py").is_file():
        raise FileNotFoundError(f"Not a CoTracker source checkout: {cotracker_repository}")
    actual_commit = _git_commit(cotracker_repository)
    if actual_commit != expected_cotracker_commit:
        raise ValueError(f"CoTracker commit mismatch: {actual_commit} != {expected_cotracker_commit}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(f"CoTracker checkpoint mismatch: {checkpoint_sha256} != {expected_checkpoint_sha256}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evidence directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[ViPE witnesses] Decoding source video", flush=True)
    frames, video_metadata = _decode_video(source_video, config.max_dimension)
    if config.semantic_masking:
        print(f"[ViPE witnesses] DeepLab masks for {len(frames)} frames", flush=True)
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

    print("[ViPE witnesses] Loading CoTracker", flush=True)
    model = _load_model(cotracker_repository, checkpoint, device)
    native_scale = np.array(
        [
            video_metadata["evaluation_to_native_scale_x"],
            video_metadata["evaluation_to_native_scale_y"],
        ],
        dtype=np.float32,
    )
    window_receipts: list[dict[str, Any]] = []
    for start in cotracker_window_starts(len(frames), config):
        end = min(start + config.window_frames, len(frames))
        query_points_tyx, query_ids, partitions = query_grid(
            video_metadata["evaluation_width"],
            video_metadata["evaluation_height"],
            step=config.query_step_pixels,
            window_start=start,
        )
        query_count_before_semantic_mask = len(query_points_tyx)
        query_static = _sample_mask(static_masks[start], query_points_tyx[:, [2, 1]])
        query_points_tyx = query_points_tyx[query_static]
        query_ids = query_ids[query_static]
        partitions = partitions[query_static]
        queries_txy = query_points_tyx[:, [0, 2, 1]].copy()
        print(f"[ViPE witnesses] CoTracker window {len(window_receipts)+1}: source frames {start+1}-{end}, {len(queries_txy)} queries", flush=True)
        tracks_eval, tracker_visible, visibility_score = _track_window(
            model,
            frames[start:end],
            queries_txy,
            device=device,
        )
        if tracks_eval.shape != (len(queries_txy), end - start, 2):
            raise ValueError(f"Unexpected CoTracker track layout: {tracks_eval.shape}")
        tracker_visible_bool = tracker_visible.astype(bool)
        static_valid = np.stack(
            [
                _sample_mask(static_masks[start + offset], tracks_eval[:, offset])
                for offset in range(end - start)
            ],
            axis=1,
        )
        visible = (tracker_visible_bool & static_valid).astype(np.uint8)
        tracks_native = tracks_eval * native_scale[None, None, :]
        query_xy_native = queries_txy[:, 1:] * native_scale[None, :]
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
            tracker_visible=tracker_visible_bool.astype(np.uint8),
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
                "query_count": len(queries_txy),
                "semantic_static_query_fraction": float(query_static.mean()),
                "routing_query_count": int(np.count_nonzero(partitions == 0)),
                "final_audit_query_count": int(np.count_nonzero(partitions == 1)),
                "tracker_visible_fraction": float(tracker_visible_bool.mean()),
                "rigid_visible_fraction": float(visible.mean()),
                "visible_fraction": float(visible.mean()),
            }
        )

    print(f"[ViPE witnesses] Saved {len(window_receipts)} tracking windows", flush=True)
    del model
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
    manifest: dict[str, Any] = {
        "schema_version": TRACK_EVIDENCE_SCHEMA_VERSION,
        "status": "sealed_camera_free_track_evidence",
        "source": {"path": str(source_video), "sha256": sha256_file(source_video), **video_metadata},
        "tracker": {
            "name": "cotracker3_scaled_offline",
            "license": "CC-BY-NC-4.0",
            "use_restriction": "internal_research_noncommercial_dependency",
            "cotracker_repository": str(cotracker_repository),
            "cotracker_commit": actual_commit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "device": device,
            "predictor_visibility_threshold": 0.9,
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
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
