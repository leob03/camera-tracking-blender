"""Read a bounded incremental chunk, opening only the requested job's log."""
import base64
import os

CHUNK_BYTES = 32768
MAX_BACKLOG_BYTES = 256 * 1024


def read_log_chunk(path, offset=-1, generation=None):
    if offset < -1:
        raise ValueError('Log offset must be -1 (recent output) or a nonnegative byte offset.')
    try:
        with path.open('rb') as stream:
            stat = os.fstat(stream.fileno())
            current_generation = f'{stat.st_dev}:{stat.st_ino}'
            reset = offset > stat.st_size or (generation is not None and generation != current_generation)
            if offset == -1 or reset:
                start = max(0, stat.st_size - CHUNK_BYTES)
            else:
                start = max(offset, stat.st_size - MAX_BACKLOG_BYTES)
            stream.seek(start)
            raw = stream.read(CHUNK_BYTES)
    except FileNotFoundError:
        return dict(data='', offset=0, next_offset=0, size=0, generation=None, reset=False, skipped_bytes=0, exists=False)
    return dict(data=base64.b64encode(raw).decode('ascii'), offset=start, next_offset=start + len(raw),
                size=stat.st_size, generation=current_generation, reset=reset,
                skipped_bytes=max(0, start - max(0, offset)) if not reset else start, exists=True)
