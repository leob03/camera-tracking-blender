"""Independent camera evidence generation and scoring."""

from camera_evidence.tapir_tracks import (
    BOOTSTAPIR_CHECKPOINT_SHA256,
    PINNED_TAPNET_COMMIT,
    TapirTrackConfig,
    generate_track_evidence,
)

__all__ = [
    "BOOTSTAPIR_CHECKPOINT_SHA256",
    "PINNED_TAPNET_COMMIT",
    "TapirTrackConfig",
    "generate_track_evidence",
]
