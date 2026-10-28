"""Attach a real splat bundle to its job, then Show / Hide Gaussian Splat (viewport viewer) and, if installed, Open in KIRI.

blender -b --factory-startup --python-exit-code 1 --python tests/blender_splat_view.py -- parent.zip splat_result.zip [plate.mp4]
The splat bundle is materialized beside itself, so pass a local copy. Drawing needs a GPU: see blender_splat_viewer_gui.py.
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
from vipe_remote.splat import kiri_available
from vipe_remote.splat_data import load_gaussians

bpy.data.objects.remove(bpy.data.objects['Cube'])
collection, created = import_bundle(parent_bundle, source_path=plate, start_frame=1001)
merged, _ = import_bundle(splat_bundle)
assert created and merged == collection and Path(collection['vipe_splat_ply']).is_file()
bpy.context.scene.vipe_remote.job_id = collection['vipe_job_id']
layers = bpy.context.view_layer.layer_collection.children[collection.name]
dense = next(c for c in collection.children if c.get('vipe_role') == 'dense_environment_group')
before = len(bpy.data.objects)
assert bpy.ops.vipe_remote.view_environment(show=True) == {'FINISHED'}  # an artist looking at the dense samples

assert bpy.ops.vipe_remote.view_splat(show=True) == {'FINISHED'}
splat = bpy.data.objects[collection['vipe_splat_viewer']]
root = next(o for o in collection.all_objects if o.get('vipe_role') == 'alignment')
assert splat.type == 'EMPTY' and splat.parent == root and splat.get('vipe_role') == 'gaussian_splat'
assert splat['vipe_splat_ply'] == collection['vipe_splat_ply'] and splat.matrix_basis == splat.matrix_basis.Identity(4)
assert splat.users_collection[0].get('vipe_role') == 'splat_group' and len(bpy.data.objects) == before + 1
assert layers.children[dense.name].hide_viewport, 'the dense environment makes way for the splat'
assert bpy.ops.vipe_remote.view_splat(show=False) == {'FINISHED'} and splat.hide_viewport
assert not layers.children[dense.name].hide_viewport, 'hiding the splat brings the dense environment back'
assert bpy.ops.vipe_remote.view_splat(show=True) == {'FINISHED'} and not splat.hide_viewport and len(bpy.data.objects) == before + 1


def world(obj, step):
    data = obj.data
    points = data.vertices if obj.type == 'MESH' else data.points
    co = np.empty(len(points) * 3, np.float32)
    points.foreach_get('co', co)
    m = np.array(obj.matrix_world)
    return co.reshape(-1, 3)[::step] @ m[:3, :3].T + m[:3, 3]


def placement(points, label):
    # The splat sits on the dense samples (quartiles agree; a flipped axis would mirror one of them).
    samples = np.concatenate([world(o, 20) for o in collection.all_objects if o.get('vipe_role') == 'dense_samples'])
    centre, spread = np.median(samples, 0), np.percentile(samples, 75, 0) - np.percentile(samples, 25, 0)
    near = points[(np.abs(points - centre) < 3 * spread).all(1)]
    offset = np.abs(np.median(near, 0) - centre) / spread
    print('PLACEMENT', label, 'dense median', centre.round(2), 'splat median', np.median(near, 0).round(2), 'offset/IQR', offset.round(2))
    assert (offset < 0.5).all(), f'{label} splat does not overlap the dense environment'


data = load_gaussians(splat['vipe_splat_ply'])
m = np.array(splat.matrix_world)
placement(data['positions'][::20] @ m[:3, :3].T + m[:3, 3], 'viewer')
print('VIEWER', data['count'], 'Gaussians, SH degree', data['sh_degree'])

if kiri_available():
    assert bpy.ops.vipe_remote.kiri_splat() == {'FINISHED'}
    kiri = bpy.data.objects[json.loads(collection['vipe_splat_objects'])[0]]
    assert kiri.parent == root and kiri.get('vipe_role') == 'kiri_splat' and splat.hide_viewport
    assert kiri.modifiers['KIRI_3DGS_Render_GN'].show_viewport and kiri.get('update_rot_to_cam'), "KIRI's splat display, not its black editing points"
    placement(world(kiri, 20), 'KIRI')
    assert bpy.ops.vipe_remote.view_splat(show=True) == {'FINISHED'} and kiri.hide_viewport and not splat.hide_viewport
print('PASS splat view; KIRI hand-off', 'tested' if kiri_available() else 'not installed')
