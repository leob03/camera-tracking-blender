"""Attach a real splat bundle to its job and load it with Show Gaussian Splat (KIRI 3DGS Render or Blender 5.3+).

blender -b --factory-startup --python-exit-code 1 --python tests/blender_splat_view.py -- parent.zip splat_result.zip [plate.mp4]
The splat bundle is materialized beside itself, so pass a local copy. Needs KIRI enabled (Blender < 5.3).
"""
from pathlib import Path
import json
import sys
import addon_utils
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
args = sys.argv[sys.argv.index('--') + 1:]
parent_bundle, splat_bundle = args[:2]
plate = args[2] if len(args) > 2 else ''
for module in addon_utils.modules():
    if 'dgs_render_by_kiri_engine' in module.__name__:
        addon_utils.enable(module.__name__, default_set=True)
addon_utils.enable('vipe_remote', default_set=True)
from vipe_remote.importer import import_bundle
from vipe_remote.splat import kiri_available, native_splats_available

bpy.data.objects.remove(bpy.data.objects['Cube'])
collection, created = import_bundle(parent_bundle, source_path=plate, start_frame=1001)
merged, _ = import_bundle(splat_bundle)
assert created and merged == collection and Path(collection['vipe_splat_ply']).is_file()
assert kiri_available() or native_splats_available(), 'install KIRI 3DGS Render or use Blender 5.3+'
bpy.context.scene.vipe_remote.job_id = collection['vipe_job_id']
assert bpy.ops.vipe_remote.view_splat(show=True) == {'FINISHED'}
names = json.loads(collection['vipe_splat_objects'])
splat = bpy.data.objects[names[0]]
root = next(o for o in collection.all_objects if o.get('vipe_role') == 'alignment')
assert splat.parent == root and splat.get('vipe_role') == 'gaussian_splat'
assert splat.users_collection[0].get('vipe_role') == 'splat_group'
if splat.type == 'MESH':  # KIRI: camera-facing splat display, not its black point-editing display
    assert splat.modifiers['KIRI_3DGS_Render_GN'].show_viewport and splat.get('update_rot_to_cam')
bpy.context.view_layer.update()


def world(obj, step):
    data = obj.data
    points = data.vertices if obj.type == 'MESH' else data.points
    co = np.empty(len(points) * 3, np.float32)
    points.foreach_get('co', co)
    m = np.array(obj.matrix_world)
    return co.reshape(-1, 3)[::step] @ m[:3, :3].T + m[:3, 3]


# Placement: the splat sits on the dense samples (quartiles agree; a flipped axis would mirror one of them).
dense = np.concatenate([world(o, 20) for o in collection.all_objects if o.get('vipe_role') == 'dense_samples'])
points = world(splat, 20)
centre, spread = np.median(dense, 0), np.percentile(dense, 75, 0) - np.percentile(dense, 25, 0)
near = points[(np.abs(points - centre) < 3 * spread).all(1)]
offset = np.abs(np.median(near, 0) - centre) / spread
print('PLACEMENT dense median', centre.round(2), 'splat median', np.median(near, 0).round(2), 'offset/IQR', offset.round(2), 'kept', len(near), len(points))
assert (offset < 0.5).all(), 'splat does not overlap the dense environment'
assert bpy.ops.vipe_remote.view_splat(show=False) == {'FINISHED'} and splat.hide_viewport and splat.hide_render
assert bpy.ops.vipe_remote.view_splat(show=True) == {'FINISHED'} and not splat.hide_viewport
assert len(json.loads(collection['vipe_splat_objects'])) == len(names), 'toggling must not import again'
print('PASS splat view;', splat.type, len(splat.data.vertices if splat.type == 'MESH' else splat.data.points), 'Gaussians; objects', names)
