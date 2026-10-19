"""Build a small display-referred proxy of an image-sequence plate for upload.

Production plates are scene-linear EXR sequences that are far too large to send to the worker.
The networks need display-referred 8-bit frames at the processing size, so the artist machine
tone-maps and resizes each frame and uploads a ZIP of JPEGs plus plate.json. Frames are opened one
at a time with ImageInput and closed immediately (no image cache holding files open on shares).
"""
import json
from pathlib import Path
import re
import zipfile

SEQUENCE_SUFFIXES = {'.exr', '.dpx', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}
FRAME = re.compile(r'^(.*?)(\d+)(\.[A-Za-z0-9]+)$')
# Input colour spaces offered to artists, mapped to names in Blender's bundled OCIO config.
COLORSPACES = [('ACEScg', 'ACEScg', 'Scene-linear ACEScg (typical VFX plates)'),
               ('ACES2065-1', 'ACES2065-1', 'Scene-linear ACES AP0'),
               ('Linear Rec.709', 'Linear Rec.709', 'Scene-linear Rec.709 / sRGB primaries'),
               ('Linear Rec.2020', 'Linear Rec.2020', 'Scene-linear Rec.2020'),
               ('sRGB', 'sRGB', 'Already display-referred (sRGB)'),
               ('RAW', 'No transform', 'Use pixel values as they are (log or display-referred plates)')]
DISPLAY = 'AgX Base sRGB'


def is_sequence(path):
    return Path(path).suffix.lower() in SEQUENCE_SUFFIXES


def find_sequence(path):
    """Return (directory, prefix, digits, suffix, frame numbers) for one frame of a numbered sequence."""
    path = Path(path)
    match = FRAME.match(path.name)
    if not match:
        raise ValueError('Choose one frame of a numbered image sequence, e.g. plate.1001.exr.')
    prefix, number, suffix = match.groups()
    frames = []
    # One bounded directory listing; never recursive.
    with __import__('os').scandir(path.parent) as entries:
        for entry in entries:
            other = FRAME.match(entry.name)
            if other and other.group(1) == prefix and other.group(3) == suffix and len(other.group(2)) == len(number):
                frames.append(int(other.group(2)))
            if len(frames) > 20000:
                raise ValueError('The sequence has more than 20000 frames.')
    frames.sort()
    return path.parent, prefix, len(number), suffix, frames


def frame_path(directory, prefix, digits, suffix, frame):
    return Path(directory) / f'{prefix}{frame:0{digits}d}{suffix}'


def describe(path):
    """Header-only plate description: size, pixel aspect, fps and frame range."""
    import OpenImageIO as oiio
    directory, prefix, digits, suffix, frames = find_sequence(path)
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise ValueError(f'Cannot read {Path(path).name}: {oiio.geterror()}')
    try:
        spec = image.spec()
        fps = spec.getattribute('framesPerSecond')
        width, height = spec.full_width or spec.width, spec.full_height or spec.height
        aspect = float(spec.getattribute('PixelAspectRatio') or spec.getattribute('pixelAspectRatio') or 1.0)
    finally:
        image.close()
    gaps = (frames[-1] - frames[0] + 1) - len(frames) if frames else 0
    return dict(width=width, height=height, pixel_aspect=aspect, fps=fps if isinstance(fps, (tuple, list)) else None,
                first=frames[0], last=frames[-1], count=len(frames), gaps=gaps,
                pattern=f'{prefix}{"#" * digits}{suffix}', directory=str(directory), prefix=prefix, digits=digits, suffix=suffix)


def proxy_size(width, height, long_edge):
    scale = min(1.0, long_edge / max(width, height)) if long_edge else 1.0
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def build_proxy(path, destination, *, first_index=1, last_index=0, long_edge=1280, colorspace='ACEScg', fps=(24, 1), ocio_config='', progress=lambda *a: None, stop=lambda: False):
    """Write proxy JPEG frames and an upload ZIP under destination. Returns (zip path, plate dict, first frame file)."""
    import numpy as np
    import OpenImageIO as oiio
    info = describe(path)
    if abs(info['pixel_aspect'] - 1.0) > 1e-3:
        raise ValueError('Anamorphic/non-square-pixel plates are not supported yet; desqueeze the plate first.')
    if info['gaps']:
        raise ValueError(f"The sequence has {info['gaps']} missing frames between {info['first']} and {info['last']}.")
    frames = list(range(info['first'], info['last'] + 1))
    selected = frames[first_index - 1:(last_index or len(frames))]
    if len(selected) < 2:
        raise ValueError('Select at least two frames of the sequence.')
    width, height = proxy_size(info['width'], info['height'], long_edge)
    destination = Path(destination)
    frames_dir = destination / 'proxy'
    frames_dir.mkdir(parents=True, exist_ok=True)
    archive_path = destination / 'proxy_frames.zip'
    source_space = dict((key, name) for key, name, _ in COLORSPACES)[colorspace]
    processor = None
    if source_space not in ('No transform', 'sRGB'):
        import PyOpenColorIO as ocio
        config = ocio.Config.CreateFromFile(ocio_config) if ocio_config else ocio.GetCurrentConfig()
        processor = config.getProcessor(source_space, DISPLAY).getDefaultCPUProcessor()
    with zipfile.ZipFile(archive_path.with_suffix('.zip.part'), 'w', zipfile.ZIP_STORED) as archive:
        for index, frame in enumerate(selected):
            if stop():
                raise InterruptedError('Proxy creation stopped.')
            source = frame_path(info['directory'], info['prefix'], info['digits'], info['suffix'], frame)
            image = oiio.ImageInput.open(str(source))
            if image is None:
                raise ValueError(f'Cannot read {source.name}: {oiio.geterror()}')
            try:
                spec = image.spec()
                if (spec.full_width or spec.width, spec.full_height or spec.height) != (info['width'], info['height']):
                    raise ValueError(f'{source.name} has a different size from the first frame.')
                channels = min(3, spec.nchannels)
                pixels = image.read_image(0, 0, 0, channels, 'float')
            finally:
                image.close()
            if pixels is None:
                raise ValueError(f'Cannot decode {source.name}.')
            pixels = np.nan_to_num(np.asarray(pixels, dtype=np.float32).reshape(spec.height, spec.width, channels), nan=0, posinf=0, neginf=0)
            if channels == 1:
                pixels = np.repeat(pixels, 3, axis=2)
            buf = oiio.ImageBuf(oiio.ImageSpec(spec.width, spec.height, 3, 'float'))
            buf.set_pixels(oiio.ROI(0, spec.width, 0, spec.height, 0, 1, 0, 3), pixels)
            if (spec.x, spec.y, spec.width, spec.height) != (spec.full_x, spec.full_y, info['width'], info['height']):
                # Place the data window inside the display window before resizing.
                full = oiio.ImageBuf(oiio.ImageSpec(info['width'], info['height'], 3, 'float'))
                oiio.ImageBufAlgo.paste(full, spec.x - spec.full_x, spec.y - spec.full_y, 0, 0, buf)
                buf = full
            # Resize in linear light, then tone-map to display (Blender's OIIO has no OCIO support).
            small = oiio.ImageBufAlgo.resize(buf, roi=oiio.ROI(0, width, 0, height, 0, 1, 0, 3), nthreads=2)
            rgb = np.ascontiguousarray(small.get_pixels(oiio.FLOAT).reshape(height, width, 3), dtype=np.float32)
            if processor is not None:
                processor.applyRGB(rgb)
            out = oiio.ImageBuf(oiio.ImageSpec(width, height, 3, 'uint8'))
            out.set_pixels(oiio.ROI(0, width, 0, height, 0, 1, 0, 3), np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8))
            output = frames_dir / f'proxy.{frame:0{max(4, info["digits"])}d}.jpg'
            out.specmod().attribute('jpeg:subsampling', '4:4:4')
            out.specmod().attribute('Compression', 'jpeg:95')
            if not out.write(str(output)):
                raise ValueError(f'Cannot write the proxy frame: {out.geterror()}')
            archive.write(output, f'frames/{index:06d}.jpg')
            progress('Preparing plate proxy', (index + 1) / len(selected))
        plate = dict(width=info['width'], height=info['height'], first_frame=selected[0], fps_num=int(fps[0]), fps_den=int(fps[1]),
                     colorspace=colorspace, pattern=info['pattern'], proxy_width=width, proxy_height=height, frame_count=len(selected))
        archive.writestr('plate.json', json.dumps(plate, indent=2))
    archive_path.with_suffix('.zip.part').replace(archive_path)
    first = frames_dir / f'proxy.{selected[0]:0{max(4, info["digits"])}d}.jpg'
    return archive_path, plate, first
