"""Image-sequence submission: build the proxy archive before upload; no Blender imports."""
from pathlib import Path

from .plate_proxy import build_proxy, is_sequence


def prepare_upload(record, progress=lambda *args: None, stop=lambda: False):
    """Return the path to upload. For image sequences, build the proxy and adjust the record.

    The record keeps 'source' (the artist's plate), and gains 'upload' (archive),
    'background' (local proxy frames for playback) and 'plate' (archive plate.json).
    The proxy already holds only the selected frames at the processing size, so the
    worker receives the full proxy range with no further downscaling.
    """
    if not is_sequence(record['source']):
        return record['source']
    settings = record['settings']
    archive, plate, first = build_proxy(record['source'], Path(record['folder']), first_index=settings['frame_start'], last_index=settings['frame_end'],
                                        long_edge=settings['max_dimension'], colorspace=record.get('plate_colorspace', 'ACEScg'),
                                        fps=tuple(record.get('plate_fps') or (24, 1)), ocio_config=record.get('ocio_config', ''), progress=progress, stop=stop)
    record.update(upload=str(archive), background=str(first), plate=plate)
    record['settings'] = {**settings, 'frame_start': 1, 'frame_end': 0}
    if record.get('use_plate_frames', True):
        record['start_frame'] = plate['first_frame']
    return str(archive)


def plate_pattern(record):
    """Nuke-style path of the original plate for exports (#### padding for sequences)."""
    if not record.get('plate'):
        return record.get('source', '')
    return str(Path(record['source']).parent / record['plate']['pattern'])
