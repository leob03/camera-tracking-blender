"""Artist flow against a live worker: solve with dense, train a splat, attach it and show it.

blender -b --factory-startup --python-exit-code 1 --python tests/blender_splat_e2e.py -- <plate> <cache_dir> <worker_url> <quick|quality> [steps]
For an image-sequence plate, Quality uploads a larger training plate of the solved frames.
"""
import json
import sys
import time
from pathlib import Path
import addon_utils
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
args = sys.argv[sys.argv.index('--') + 1:]
plate, cache, url, preset = args[:4]
steps = int(args[4]) if len(args) > 4 else 0
for module in addon_utils.modules():
    if 'dgs_render_by_kiri_engine' in module.__name__:
        addon_utils.enable(module.__name__, default_set=True)
addon_utils.enable('vipe_remote', default_set=True)
from vipe_remote import addon

bpy.data.objects.remove(bpy.data.objects['Cube'])
prefs = addon.preferences(bpy.context)
prefs.cache_directory, prefs.worker_url = cache, url
props = bpy.context.scene.vipe_remote
timings = {}


def run(operator, limit):
    start = time.monotonic()
    assert operator() == {'FINISHED'}
    while addon._active is not None and time.monotonic() - start < limit:
        addon._poll()
        time.sleep(0.2)
    assert addon._active is None and props.state == 'COMPLETE', (props.state, props.status)
    return round(time.monotonic() - start, 1)


props.source = plate
props.dense_environment = True
timings['solve_with_dense'] = run(bpy.ops.vipe_remote.submit, 3600)
collection = addon.job_collection(bpy.context)
props.splat_preset = preset
if steps:
    props.splat_custom, props.splat_iterations = True, steps
    props.splat_max_dimension = 2048 if preset == 'quality' else 1536
    props.splat_max_gaussians = 500000
timings['splat'] = run(bpy.ops.vipe_remote.train_splat, 7200)
record = json.loads(props.job_record)
assert collection.get('vipe_splat_ply') and Path(collection['vipe_splat_ply']).is_file()
splat_record = json.loads((Path(cache) / collection['vipe_splat_job_id'] / 'job.json').read_text())
report = json.loads((Path(collection['vipe_splat_directory']) / 'dataset.json').read_text())
assert bpy.ops.vipe_remote.view_splat(show=True) == {'FINISHED'}
names = json.loads(collection['vipe_splat_objects'])
print('E2E', json.dumps(dict(preset=preset, timings=timings, training_long_edge=splat_record.get('training_long_edge'), training_plate=report['training_plate'],
                             decoded=report['decoded_size'], size=[report['width'], report['height']], views=report['count'], seeds=report['seed_points'],
                             objects=names, ply_mb=round(Path(collection['vipe_splat_ply']).stat().st_size / 1e6, 1))))
bpy.ops.wm.save_as_mainfile(filepath=str(Path(cache) / f'e2e_{preset}.blend'))
print('PASS splat e2e')
