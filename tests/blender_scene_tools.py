"""Run: blender -b --factory-startup --python-exit-code 1 --python tests/blender_scene_tools.py

Scene alignment tools on a synthetic job (tilted floor in ViPE/OpenCV space); no bundle or network."""
from pathlib import Path
import json, math, sys
import bpy, addon_utils
from mathutils import Vector
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
addon_utils.enable('vipe_remote', default_set=True, persistent=True)
scene, layer = bpy.context.scene, bpy.context.view_layer
JOB = '00000000-0000-4000-8000-0000000000aa'
SCALE = 2.0
scene.vipe_remote.job_id = JOB
scene.vipe_remote.job_record = json.dumps(dict(job_id=JOB, scene_scale=SCALE))

collection = bpy.data.collections.new(f'ViPE | test | {JOB[:8]}'); scene.collection.children.link(collection)
collection['vipe_job_id'] = JOB
root = bpy.data.objects.new('ViPE alignment', None); collection.objects.link(root)
root.rotation_euler.x = math.radians(-90); root.scale = (SCALE,) * 3; root['vipe_role'] = 'alignment'
normal = Vector((0.2, -1.0, 0.3)).normalized()
u = normal.orthogonal().normalized(); w = normal.cross(u)
origin = Vector((0.5, 1.2, 4.0))
plane = [origin + a * u + b * w for a, b in [(0, 0), (2, 0), (0, 3), (2.5, 2), (-1, 1.5)]]

def empty(name, local, role='scene_marker'):
    obj = bpy.data.objects.new(name, None); collection.objects.link(obj)
    obj.parent = root; obj.location = local; obj['vipe_role'] = role
    return obj

markers = [empty(f'Static track {i}', p) for i, p in enumerate(plane)]
middle = empty('Collinear', (plane[0] + plane[1]) / 2)
cam = bpy.data.objects.new('Solved camera', bpy.data.cameras.new('Solved camera')); collection.objects.link(cam)
cam.parent = root; cam['vipe_role'] = 'camera'; collection['vipe_camera'] = cam.name; scene.camera = cam
layer.update()
original = root.matrix_world.copy()
initial = [m.matrix_world.translation.copy() for m in markers]

def select(objs, active=None):
    for obj in layer.objects: obj.select_set(False)
    for obj in objs: obj.select_set(True)
    layer.objects.active = active or objs[-1]

def world(obj): return obj.matrix_world.translation

def close(a, b, eps=1e-5): return (Vector(a) - Vector(b)).length < eps

def expect_error(op, text, **kw):
    try:
        op(**kw)
    except RuntimeError as exc:
        assert text in str(exc), exc
    else:
        raise AssertionError(f'{op} should fail with: {text}')

results = {}
for side in (1, -1):  # Camera on either side of the fitted plane: SVD normal sign must be oriented toward it.
    bpy.ops.vipe_remote.reset_alignment(); layer.update()
    cam.location = origin + side * 3 * normal + 0.5 * u; layer.update()
    select(markers + [cam, root], active=markers[0])  # Camera and alignment root are ignored.
    assert bpy.ops.vipe_remote.set_floor() == {'FINISHED'}
    points = [world(m) for m in markers]
    assert max(abs(p.z) for p in points) < 1e-5, points
    assert (sum(points, Vector()) / len(points)).length < 1e-5
    assert abs(world(cam).z - 3 * SCALE) < 1e-4, world(cam)
    for i in range(len(markers)):
        assert abs((points[i] - points[0]).length - SCALE * (plane[i] - plane[0]).length) < 1e-5
    before = root.matrix_world.copy()
    assert bpy.ops.vipe_remote.set_floor() == {'FINISHED'}  # Already level: minimal rotation means no change.
    assert all(close(before[i], root.matrix_world[i], 1e-5) for i in range(4))
    results[f'floor_camera_side_{side}'] = round(world(cam).z, 6)
assert 'vipe_original_matrix' in root and abs(root['vipe_scene_scale'] - SCALE) < 1e-9

select([markers[1]])
assert bpy.ops.vipe_remote.set_origin() == {'FINISHED'}
assert world(markers[1]).length < 1e-5 and max(abs(world(m).z) for m in markers) < 1e-5

select([markers[0], markers[2]])
assert bpy.ops.vipe_remote.set_scale(distance=5.0) == {'FINISHED'}
assert abs((world(markers[2]) - world(markers[0])).length - 5.0) < 1e-5
assert world(markers[1]).length < 1e-5 and max(abs(world(m).z) for m in markers) < 1e-5

select([markers[0], markers[3]], active=markers[3])
assert bpy.ops.vipe_remote.set_axis() == {'FINISHED'}
d = world(markers[3]) - world(markers[0])
assert abs(d.y) < 1e-5 and d.x > 0 and abs(d.z) < 1e-5, d
assert world(markers[1]).length < 1e-5 and abs((world(markers[2]) - world(markers[0])).length - 5.0) < 1e-5
results['axis_vector'] = [round(v, 6) for v in d]

select(markers[:2]); expect_error(bpy.ops.vipe_remote.set_floor, 'at least three')
select([markers[0], markers[1], middle]); expect_error(bpy.ops.vipe_remote.set_floor, 'collinear')
select(markers[:3]); expect_error(bpy.ops.vipe_remote.set_scale, 'exactly two', distance=1.0)
select([markers[0]]); expect_error(bpy.ops.vipe_remote.set_axis, 'exactly two')
for obj in layer.objects: obj.select_set(False)
expect_error(bpy.ops.vipe_remote.set_origin, 'one or more')

assert bpy.ops.vipe_remote.reset_alignment() == {'FINISHED'}
assert all(close(original[i], root.matrix_world[i], 1e-5) for i in range(4))
assert all(close(world(m), p, 1e-5) for m, p in zip(markers, initial))

# Without a stored matrix, reset rebuilds the import layout from the stored scene scale.
del root['vipe_original_matrix']; root.location = (1, 2, 3); root.rotation_euler = (0.1, 0.2, 0.3); root.scale = (5, 5, 5)
assert bpy.ops.vipe_remote.reset_alignment() == {'FINISHED'}
assert all(close(original[i], root.matrix_world[i], 1e-5) for i in range(4))

# Edit Mode: selected vertices of a native-cloud style mesh, outlier unselected.
mesh = bpy.data.meshes.new('cloud'); mesh.from_pydata([tuple(p) for p in plane] + [tuple(origin + 7 * normal)], [], [])
cloud = bpy.data.objects.new('Native cloud', mesh); collection.objects.link(cloud); cloud.parent = root; cloud['vipe_role'] = 'native_cloud'
for v in mesh.vertices: v.select = v.index < len(plane)
select([cloud]); layer.update()
bpy.ops.object.mode_set(mode='EDIT')
assert bpy.context.mode == 'EDIT_MESH'
assert bpy.ops.vipe_remote.set_floor() == {'FINISHED'}
bpy.ops.object.mode_set(mode='OBJECT'); layer.update()
verts = [cloud.matrix_world @ v.co for v in mesh.vertices]
assert max(abs(p.z) for p in verts[:len(plane)]) < 1e-5, verts
assert abs(verts[-1].z) > 1.0 and all(abs(world(m).z) < 1e-5 for m in markers)
# In Object Mode the cloud object itself (origin at the root) is not a marker.
select([cloud, markers[0]]); assert bpy.ops.vipe_remote.set_origin() == {'FINISHED'}
assert world(markers[0]).length < 1e-5
results['edit_mode_floor'] = True
print(json.dumps(dict(state='PASS', blender=bpy.app.version_string, **results)))
