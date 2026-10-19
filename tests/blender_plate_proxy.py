"""Build an upload proxy from an image sequence in Blender: blender -b --factory-startup --python tests/blender_plate_proxy.py -- <frame.exr> <out_dir>"""
import json
import os
from pathlib import Path
import sys
import zipfile
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vipe_remote.plate_proxy import build_proxy, describe

frame, out = sys.argv[sys.argv.index('--') + 1:][:2]
info = describe(frame)
print('PLATE', json.dumps(info))
config = os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles', 'colormanagement', 'config.ocio')
archive, plate, first = build_proxy(frame, out, long_edge=1280, colorspace='ACEScg', fps=info['fps'] or (24, 1), ocio_config=config)
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
assert names[-1] == 'plate.json' and names[0] == 'frames/000000.jpg' and len(names) == info['count'] + 1, names[:3]
assert plate['first_frame'] == info['first'] and (plate['proxy_width'], plate['proxy_height']) == (1280, 676 if info['height'] * 1280 / info['width'] < 677 else plate['proxy_height'])
clip = bpy.data.movieclips.load(str(first))
assert tuple(clip.size) == (plate['proxy_width'], plate['proxy_height']) and clip.frame_duration == info['count'], (tuple(clip.size), clip.frame_duration)
print('PROXY_OK', archive, os.path.getsize(archive), json.dumps(plate))
