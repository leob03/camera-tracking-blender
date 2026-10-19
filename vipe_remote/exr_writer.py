"""Minimal single-part scanline OpenEXR writer: 32-bit FLOAT channels, ZIP or no compression."""
import struct
import zlib

import numpy as np


COMPRESSION = {'NONE': (0, 1), 'ZIP': (3, 16)}


def _attribute(name, kind, data):
    return name.encode() + b'\0' + kind.encode() + b'\0' + struct.pack('<i', len(data)) + data


def _zip(raw):
    # OpenEXR ZIP: interleave bytes into two halves, delta-predict, then deflate.
    data = np.frombuffer(raw, np.uint8)
    data = np.concatenate([data[0::2], data[1::2]])
    data[1:] = data[1:] - data[:-1] + np.uint8(128)
    packed = zlib.compress(data.tobytes(), 6)
    return packed if len(packed) < len(raw) else raw


def write_exr(path, channels, compression='ZIP'):
    """Write `channels` ({name: HxW array}, row 0 = top) as FLOAT data; returns the path."""
    if compression not in COMPRESSION:
        raise ValueError(f"Unsupported EXR compression: {compression}")
    names = sorted(channels)
    if not names:
        raise ValueError("An EXR image needs at least one channel.")
    planes = [np.asarray(channels[name], dtype='<f4') for name in names]
    height, width = planes[0].shape
    if width < 1 or height < 1 or any(p.shape != (height, width) for p in planes):
        raise ValueError("EXR channels must be non-empty 2D arrays of one size.")
    code, lines = COMPRESSION[compression]
    box = struct.pack('<4i', 0, 0, width - 1, height - 1)
    chlist = b''.join(n.encode() + b'\0' + struct.pack('<iB3xii', 2, 0, 1, 1) for n in names) + b'\0'
    header = (struct.pack('<ii', 20000630, 2)
        + _attribute('channels', 'chlist', chlist)
        + _attribute('compression', 'compression', bytes([code]))
        + _attribute('dataWindow', 'box2i', box)
        + _attribute('displayWindow', 'box2i', box)
        + _attribute('lineOrder', 'lineOrder', b'\0')
        + _attribute('pixelAspectRatio', 'float', struct.pack('<f', 1.0))
        + _attribute('screenWindowCenter', 'v2f', struct.pack('<ff', 0.0, 0.0))
        + _attribute('screenWindowWidth', 'float', struct.pack('<f', 1.0)) + b'\0')
    # Scanline layout: for each line, every channel's row in alphabetical channel order.
    pixels = np.ascontiguousarray(np.stack(planes, axis=1))
    chunks = []
    for y in range(0, height, lines):
        raw = pixels[y:y + lines].tobytes()
        data = _zip(raw) if code else raw
        chunks.append(struct.pack('<ii', y, len(data)) + data)
    offset = len(header) + 8 * len(chunks)
    table = []
    for chunk in chunks:
        table.append(offset)
        offset += len(chunk)
    with open(path, 'wb') as handle:
        handle.write(header)
        handle.write(struct.pack(f'<{len(table)}Q', *table))
        for chunk in chunks:
            handle.write(chunk)
    return path
