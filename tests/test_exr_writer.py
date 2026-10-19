"""EXR writer checks with an independent decoder (NumPy); also reads back with OpenImageIO when importable,
e.g. Blender's Python: <blender>/4.2/python/bin/python3.11 tests/test_exr_writer.py"""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vipe_remote.exr_writer import write_exr
try:
    import OpenImageIO as oiio
except ImportError:
    oiio = None


def _unzip(data):
    t = np.frombuffer(zlib.decompress(data), np.uint8).copy()
    t[1:] -= np.uint8(128)
    t = np.cumsum(t, dtype=np.uint8)
    out = np.empty_like(t)
    half = (len(t) + 1) // 2
    out[0::2], out[1::2] = t[:half], t[half:]
    return out.tobytes()


def read_exr(path):
    b = Path(path).read_bytes()
    assert struct.unpack_from('<ii', b) == (20000630, 2)
    pos, attrs = 8, {}
    while b[pos]:
        name_end = b.index(b'\0', pos); kind_end = b.index(b'\0', name_end + 1)
        size, = struct.unpack_from('<i', b, kind_end + 1)
        attrs[b[pos:name_end].decode()] = (b[name_end + 1:kind_end].decode(), b[kind_end + 5:kind_end + 5 + size])
        pos = kind_end + 5 + size
    pos += 1
    chlist, names, i = attrs['channels'][1], [], 0
    while chlist[i]:
        end = chlist.index(b'\0', i)
        names.append(chlist[i:end].decode())
        assert struct.unpack_from('<iB3xii', chlist, end + 1) == (2, 0, 1, 1)
        i = end + 17
    x0, y0, x1, y1 = struct.unpack('<4i', attrs['dataWindow'][1])
    width, height = x1 - x0 + 1, y1 - y0 + 1
    lines = {0: 1, 3: 16}[attrs['compression'][1][0]]
    offsets = struct.unpack_from(f'<{-(-height // lines)}Q', b, pos)
    out, raw = np.zeros((height, len(names), width), '<f4'), []
    for offset in offsets:
        y, size = struct.unpack_from('<ii', b, offset)
        count = min(lines, height - y)
        data = b[offset + 8:offset + 8 + size]
        raw.append(size == count * len(names) * width * 4)
        if not raw[-1]:
            data = _unzip(data)
        out[y:y + count] = np.frombuffer(data, '<f4').reshape(count, len(names), width)
    return attrs, {n: out[:, k] for k, n in enumerate(names)}, raw


class ExrWriter(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def check(self, width, height, compression, fill):
        rng = np.random.default_rng(width * 1000 + height)
        channels = {'R': fill(rng, (height, width)), 'G': np.linspace(-2, 3, width * height).reshape(height, width), 'B': np.zeros((height, width))}
        path = Path(self.temp.name) / f'{width}x{height}_{compression}.exr'
        write_exr(path, channels, compression)
        attrs, back, raw = read_exr(path)
        self.assertEqual(set(attrs), {'channels', 'compression', 'dataWindow', 'displayWindow', 'lineOrder', 'pixelAspectRatio', 'screenWindowCenter', 'screenWindowWidth'})
        for name in 'RGB':
            np.testing.assert_array_equal(back[name], np.float32(channels[name]))
        if oiio is not None:
            image = oiio.ImageInput.open(str(path))
            self.assertIsNotNone(image, oiio.geterror())
            spec = image.spec()
            self.assertEqual((spec.width, spec.height, list(spec.channelnames)), (width, height, ['R', 'G', 'B']))
            pixels = image.read_image(format='float')
            image.close()
            for k, name in enumerate('RGB'):
                np.testing.assert_allclose(pixels[..., k], np.float32(channels[name]), rtol=0, atol=1e-7)
        return raw

    def test_sizes_and_compressions(self):
        for compression in ('ZIP', 'NONE'):
            for width, height in ((1, 1), (37, 23), (64, 48), (5, 33)):
                self.check(width, height, compression, lambda rng, shape: rng.normal(size=shape) * 100)

    def test_incompressible_blocks_are_stored_raw(self):
        self.assertEqual(self.check(1, 40, 'ZIP', lambda rng, shape: rng.random(shape)), [True] * 3)
        self.assertEqual(self.check(256, 64, 'ZIP', lambda rng, shape: np.zeros(shape)), [False] * 4)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            write_exr(Path(self.temp.name) / 'x.exr', {'R': np.zeros((2, 2)), 'G': np.zeros((3, 2))})
        with self.assertRaises(ValueError):
            write_exr(Path(self.temp.name) / 'x.exr', {'R': np.zeros((2, 2))}, 'PIZ')


if __name__ == '__main__':
    print('OpenImageIO:', oiio.VERSION_STRING if oiio else 'not available')
    unittest.main()
