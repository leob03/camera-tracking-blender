"""Gaussian splat PLY to GPU-ready arrays with numpy only (no Blender), for the viewport splat viewer.

Per Gaussian, `base` holds three RGBA texels: position and opacity, then the six unique
entries of its 3D covariance in the PLY's own (ViPE world) frame. `sh` holds the colour
as 16 spherical-harmonic RGB coefficients (degree 3) packed into 12 RGBA texels.
"""
import numpy as np

BASE_TEXELS = 3
SH_TEXELS = 12
SH_DEGREE_FOR_REST = {0: 0, 9: 1, 24: 2, 45: 3}  # f_rest count per colour channel set -> SH degree
CHUNK = 1 << 20


def read_header(stream):
    lines = []
    while not lines or lines[-1] != 'end_header':
        line = stream.readline()
        if not line or len(lines) > 400:
            raise ValueError('Unreadable PLY header.')
        lines.append(line.decode('ascii', 'replace').strip())
    if 'format binary_little_endian 1.0' not in lines:
        raise ValueError('Expected a binary little-endian Gaussian PLY.')
    count = next(int(line.split()[2]) for line in lines if line.startswith('element vertex'))
    properties = [line.split() for line in lines if line.startswith('property')]
    if any(p[1] != 'float' for p in properties):
        raise ValueError('Unexpected Gaussian PLY layout (non-float properties).')
    return count, [p[2] for p in properties], stream.tell()


def quaternion_matrices(q):
    """Rotation matrices for (w, x, y, z) quaternions, normalised first."""
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)


def load_gaussians(path, pad=1):
    """Read a 3DGS PLY (INRIA layout, as LichtFeld exports it) into arrays for the GPU.

    Returns count, sh_degree, positions (N, 3), importance (N,), base (P, 3, 4) and sh (P, 12, 4), all float32,
    where P is N rounded up to a multiple of pad (zero Gaussians), so texture rows need no extra copy.
    """
    with open(path, 'rb') as stream:
        count, names, offset = read_header(stream)
    column = {name: i for i, name in enumerate(names)}
    required = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    missing = [name for name in required if name not in column]
    if missing:
        raise ValueError('Not a Gaussian splat PLY; missing ' + ', '.join(missing))
    rest = sorted((name for name in names if name.startswith('f_rest_')), key=lambda name: int(name[7:]))
    if len(rest) not in SH_DEGREE_FOR_REST:
        raise ValueError(f'Unsupported spherical-harmonic layout ({len(rest)} f_rest values).')
    degree, per_channel = SH_DEGREE_FOR_REST[len(rest)], len(rest) // 3
    raw = np.memmap(path, dtype='<f4', mode='r', offset=offset, shape=(count, len(names)))
    positions = np.empty((count, 3), np.float32)
    importance = np.empty(count, np.float32)
    padded = -(-count // pad) * pad
    base = np.zeros((padded, BASE_TEXELS, 4), np.float32)
    sh = np.zeros((padded, SH_TEXELS * 4), np.float32)
    pick = lambda chunk, keys: chunk[:, [column[k] for k in keys]].astype(np.float32)
    for start in range(0, count, CHUNK):
        chunk = np.asarray(raw[start:start + CHUNK])
        end = start + len(chunk)
        xyz = pick(chunk, ['x', 'y', 'z'])
        alpha = 1 / (1 + np.exp(-pick(chunk, ['opacity'])[:, 0]))
        scale = np.exp(pick(chunk, ['scale_0', 'scale_1', 'scale_2']))
        m = quaternion_matrices(pick(chunk, ['rot_0', 'rot_1', 'rot_2', 'rot_3'])) * scale[:, None, :]
        cov = m @ m.transpose(0, 2, 1)
        positions[start:end] = xyz
        importance[start:end] = alpha * np.cbrt(scale.prod(axis=1))
        base[start:end, 0, :3], base[start:end, 0, 3] = xyz, alpha
        base[start:end, 1] = cov[:, [0, 0, 0, 1], [0, 1, 2, 1]]
        base[start:end, 2, :2] = cov[:, [1, 2], [2, 2]]
        # Coefficient-major RGB: DC, then f_rest, which the PLY stores channel-major (all red, all green, all blue).
        sh[start:end, 0:3] = pick(chunk, ['f_dc_0', 'f_dc_1', 'f_dc_2'])
        if per_channel:
            channels = pick(chunk, rest).reshape(-1, 3, per_channel)
            sh[start:end, 3:3 + 3 * per_channel] = channels.transpose(0, 2, 1).reshape(-1, 3 * per_channel)
    del raw
    return dict(count=count, sh_degree=degree, positions=positions, importance=importance, base=base, sh=sh.reshape(padded, SH_TEXELS, 4))


def display_subset(importance, percent):
    """Indices of the most visible Gaussians (opacity times size) for a lighter viewport, in file order."""
    count = len(importance)
    keep = max(1, int(round(count * min(max(percent, 1), 100) / 100)))
    if keep >= count:
        return np.arange(count, dtype=np.int64)
    return np.sort(np.argpartition(-importance, keep - 1)[:keep])


def back_to_front(points, ids, view_row):
    """Draw order (as float32 Gaussian indices) for alpha blending: farthest first along the view axis.

    points are the positions of the Gaussians `ids`; view_row is the third row of the model-view matrix.
    Blender's camera looks down -Z, so farther means more negative. Depths are quantised to 16 bits so
    numpy uses its linear-time radix sort.
    """
    depth = points @ np.asarray(view_row[:3], np.float32) + np.float32(view_row[3])
    low, high = float(depth.min()), float(depth.max())
    key = ((depth - low) * (65535.0 / max(high - low, 1e-9))).astype(np.uint16)
    return ids[np.argsort(key, kind='stable')].astype(np.float32)
