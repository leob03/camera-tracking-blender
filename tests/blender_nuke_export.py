"""Run: blender -b --factory-startup --python-exit-code 1 --python tests/blender_nuke_export.py -- OUTPUT_DIR
Synthetic ViPE job -> export_nuke; parses the .nk/.chan back and checks EXRs with Blender and OpenImageIO."""
from pathlib import Path
import json
import math
import re
import sys
import tempfile
import bpy
import numpy as np
from mathutils import Euler, Matrix
from bpy_extras.object_utils import world_to_camera_view
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vipe_remote.nuke_export import export_nuke, mei_project, mei_unproject, stmaps
from vipe_remote.exr_writer import write_exr
try:
    import OpenImageIO as oiio
except ImportError:
    oiio = None

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
out = Path(args[0]) if args else Path(tempfile.mkdtemp())
W, H, START, COUNT, XI, F_U = 960, 540, 1001, 20, 0.08, 700.0
CX, CY = W / 2 + 12.5, H / 2 - 7.25
C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)


def child(parent, name):
    item = bpy.data.collections.new(name)
    parent.children.link(item)
    return item


def rot(axis, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return {'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), 'y': np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), 'z': np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def nuke_expected(obj):
    m = np.array(obj.matrix_world)
    rotation = C @ m[:3, :3]
    return rotation / np.linalg.norm(rotation, axis=0), C @ m[:3, 3]


def encode(u, v):
    return np.stack([(u + 0.5) / W, (H - v - 0.5) / H], -1)


def decode(rg):
    return rg[..., 0] * W - 0.5, H - 0.5 - rg[..., 1] * H


def bilinear(image, x, y):
    x0 = np.clip(np.floor(x).astype(int), 0, image.shape[1] - 2); y0 = np.clip(np.floor(y).astype(int), 0, image.shape[0] - 2)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x0 + 1] * fx * (1 - fy)
        + image[y0 + 1, x0] * (1 - fx) * fy + image[y0 + 1, x0 + 1] * fx * fy)


def round_trip(undistort, redistort, margin=3):
    ud, vd = decode(undistort)
    rows, cols = np.mgrid[0:undistort.shape[0], 0:undistort.shape[1]]
    inside = (ud > margin) & (ud < W - 1 - margin) & (vd > margin) & (vd < H - 1 - margin)
    inside[:margin] = inside[-margin:] = False; inside[:, :margin] = inside[:, -margin:] = False
    u, v = decode(bilinear(redistort, ud[inside], vd[inside]))
    return float(np.max(np.hypot(u - cols[inside], v - rows[inside]))), int(inside.sum())


# Job collection laid out like importer.import_bundle.
job = bpy.data.collections.new('ViPE | shot_010 | 0123abcd')
scene.collection.children.link(job)
camera_collection, cloud_collection = child(job, '01 Camera and source plate'), child(job, '04 Native point cloud')
root = bpy.data.objects.new('ViPE alignment', None)
job.objects.link(root)
root.location, root.rotation_euler, root.scale = (0.4, -1.3, 0.7), (math.radians(-90), math.radians(10), math.radians(25)), (1.7,) * 3
data = bpy.data.cameras.new('shot_010 solved camera')
cam = bpy.data.objects.new(data.name, data)
camera_collection.objects.link(cam)
cam.parent, cam.rotation_mode = root, 'QUATERNION'
data.sensor_fit, data.sensor_width, data.clip_start, data.clip_end = 'HORIZONTAL', 36.0, 0.001, 1000000
previous = None
for i in range(COUNT):
    cam.location = (0.15 * i, 0.05 * math.sin(i), 0.03 * i)
    q = Euler((math.radians(180 + 25 * math.sin(i / 3)), math.radians(-35 + 6 * i), math.radians(-150 + 17 * i)), 'XYZ').to_quaternion()
    if previous is not None and previous.dot(q) < 0:
        q.negate()
    cam.rotation_quaternion, previous = q, q.copy()
    cam.keyframe_insert('location', frame=START + i); cam.keyframe_insert('rotation_quaternion', frame=START + i)
    data.lens, data.shift_x, data.shift_y = F_U / W * 36, -(CX - W / 2) / W, (CY - H / 2) / W
    for key in ('lens', 'shift_x', 'shift_y'):
        data.keyframe_insert(key, frame=START + i)
rng = np.random.default_rng(7)
points = rng.normal(size=(500, 3)) * [2, 1, 3] + [1, 0, 9]
mesh = bpy.data.meshes.new('ViPE native points')
mesh.from_pydata(points.tolist(), [], [])
rgba = rng.random((500, 4)).astype(np.float32); rgba[:, 3] = 1
mesh.color_attributes.new(name='Color', type='FLOAT_COLOR', domain='POINT').data.foreach_set('color', rgba.reshape(-1))
cloud = bpy.data.objects.new('Native cloud', mesh)
cloud_collection.objects.link(cloud)
cloud.parent, cloud['vipe_role'] = root, 'native_cloud'
group = bpy.data.node_groups.new('ViPE point display', 'GeometryNodeTree')
group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
group_in, group_out, sphere, instance = (group.nodes.new(k) for k in ('NodeGroupInput', 'NodeGroupOutput', 'GeometryNodeMeshIcoSphere', 'GeometryNodeInstanceOnPoints'))
group.links.new(group_in.outputs[0], instance.inputs['Points']); group.links.new(sphere.outputs['Mesh'], instance.inputs['Instance'])
group.links.new(instance.outputs['Instances'], group_out.inputs[0])
cloud.modifiers.new('Point display', 'NODES').node_group = group
cloud_collection.hide_render = True
manifest = {'job_id': '0123abcd-0000-4000-8000-000000000001', 'source': {'name': 'shot_010.mov', 'width': W, 'height': H, 'fps_num': 24000, 'fps_den': 1001},
    'selection': {'count': COUNT, 'first_source_frame': 0}}
text = bpy.data.texts.new('ViPE shot_010 result'); text.write(json.dumps(manifest))
lens = {'model': 'mei', 'fx': F_U * (1 + XI), 'fy': F_U * (1 + XI), 'cx': CX, 'cy': CY, 'xi': XI, 'undistorted_focal_px': F_U, 'width': W, 'height': H}
out.mkdir(parents=True, exist_ok=True)
for key, value in {'vipe_job_id': manifest['job_id'], 'vipe_camera': cam.name, 'vipe_start_frame': START, 'vipe_manifest_text': text.name,
        'vipe_plate_path': str(out / 'plate.####.exr'), 'vipe_plate_first_frame': 1, 'vipe_distortion': json.dumps(lens)}.items():
    job[key] = value
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.frame_set(1010)
bpy.context.view_layer.update()
bpy.context.view_layer.layer_collection.children[job.name].children[cloud_collection.name].hide_viewport = True
for obj in bpy.context.view_layer.objects:
    obj.select_set(False)
root.select_set(True); bpy.context.view_layer.objects.active = root
scene.render.resolution_percentage = 50

result = export_nuke(bpy.context, job, str(out / 'export'))
print('WARNINGS', *result['warnings'], sep='\n  ')
assert all(Path(f).is_file() for f in result['files']) and result['nk'] in result['files']
assert scene.frame_current == 1010 and scene.render.resolution_percentage == 50
assert root.select_get() and not cam.select_get() and not cloud.select_get() and bpy.context.view_layer.objects.active == root
assert bpy.context.view_layer.layer_collection.children[job.name].children[cloud_collection.name].hide_viewport and cloud.modifiers[0].show_viewport
assert not [w for w in result['warnings'] if 'focal' in w or 'principal' in w], result['warnings']
nk = Path(result['nk']).read_text()
assert 'Root {' not in nk and nk.count('STMap {') == 2 and 'frame_mode offset' in nk and 'frame 1000' in nk and 'first 1\n last 20' in nk
block = re.search(r'\nCamera2 \{\n(.*?)\n\}', nk, re.S).group(1)
knobs = dict(line.strip().split(' ', 1) for line in block.splitlines())
assert knobs['rot_order'] == 'ZXY' and knobs['name'] == 'ViPE_Camera'
curves = {k: [np.array(c.split(), float) for s, c in re.findall(r'\{curve x(\d+) ([^}]*)\}', knobs[k]) if int(s) == START] for k in ('translate', 'rotate')}
assert all(len(c) == 3 and all(len(x) == COUNT for x in c) for c in curves.values())
focal, hap, vap = float(knobs['focal']), float(knobs['haperture']), float(knobs['vaperture'])
assert abs(focal - F_U / W * 36) < 1e-7 and hap == 36 and abs(vap - 36 * H / W) < 1e-7
win = np.array(knobs['win_translate'].strip('{}').split(), float)
assert np.allclose(win, [2 * data.shift_x, 2 * data.shift_y], atol=1e-9)
errors, projection = [], []
for i in range(COUNT):
    scene.frame_set(START + i)
    expected_r, expected_t = nuke_expected(cam)
    rx, ry, rz = (curves['rotate'][k][i] for k in range(3))
    got_r = rot('y', ry) @ rot('x', rx) @ rot('z', rz)
    got_t = np.array([curves['translate'][k][i] for k in range(3)])
    errors.append(max(np.abs(got_r - expected_r).max(), np.abs(got_t - expected_t).max()))
    # Documented Nuke projection model: ndc = focal/(haperture/2) * (x, y)/-z - win_translate.
    world = np.array(cloud.matrix_world) @ np.c_[points, np.ones(len(points))].T
    local = got_r.T @ (C @ world[:3] - got_t[:, None])
    ndc = focal / (hap / 2) * local[:2] / -local[2] - win[:, None]
    blender = np.array([world_to_camera_view(scene, cam, cloud.matrix_world @ vertex.co)[:2] for vertex in mesh.vertices]).T
    front = (local[2] < -1) & (np.abs(ndc).max(0) < 1.5)
    projection.append(np.abs(np.array([(ndc[0] + 1) / 2, 0.5 + ndc[1] / 2 * W / H]) - blender)[:, front].max())
print('camera max |R,T| error', max(errors), 'projection vs Blender (frame units)', max(projection))
assert max(errors) < 1e-5 and max(projection) < 1e-5
rotations = np.array(curves['rotate']).T
assert np.abs(np.diff(rotations, axis=0)).max() < 90, 'Euler flip between frames'
chan = np.loadtxt(next(f for f in result['files'] if f.endswith('.chan')))
assert chan.shape == (COUNT, 8) and np.array_equal(chan[:, 0], np.arange(START, START + COUNT))
assert np.allclose(chan[:, 1:4], np.array(curves['translate']).T, atol=1e-9) and np.allclose(chan[:, 4:7], rotations, atol=1e-9)
assert np.allclose(chan[:, 7], np.degrees(2 * np.arctan(vap / 2 / focal)), atol=1e-7)
cloud_block = re.search(r'\nBakedPointCloud \{\n(.*?)\n\}', nk, re.S).group(1)
serial = {k: np.array(v.split(), float) for k, v in re.findall(r' (serialize\w+) "([^"]*)"', cloud_block)}
assert all(v[0] == 500 and len(v) == 1501 for v in serial.values())
scene.frame_set(START)
world = (np.array(cloud.matrix_world) @ np.c_[points, np.ones(len(points))].T)[:3]
assert np.allclose(serial['serializePoints'][1:].reshape(-1, 3), (C @ world).T, rtol=1e-6, atol=1e-5)
assert np.allclose(serial['serializeColors'][1:].reshape(-1, 3), rgba[:, :3], atol=1e-6)

# STMaps: Blender and OpenImageIO read the same float pixels; values match an independent MEI evaluation.
u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
x, y = (u - CX) / F_U, (v - CY) / F_U
r = np.sqrt(x * x + y * y + 1)
expected = {'stmap_undistort.exr': encode(lens['fx'] * x / (1 + XI * r) + CX, lens['fy'] * y / (1 + XI * r) + CY)}
mx, my = (u - CX) / lens['fx'], (v - CY) / lens['fy']
k = (XI + np.sqrt(1 + (1 - XI * XI) * (mx * mx + my * my))) / (1 + mx * mx + my * my)
expected['stmap_redistort.exr'] = encode(F_U * mx * k / (k - XI) + CX, F_U * my * k / (k - XI) + CY)
loaded = {}
for name, values in expected.items():
    path = out / 'export' / name
    image = bpy.data.images.load(str(path))
    image.colorspace_settings.name = 'Non-Color'
    assert tuple(image.size) == (W, H) and image.is_float, (image.size, image.is_float)
    pixels = np.empty(W * H * 4, np.float32); image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(H, W, 4)[::-1]
    assert np.abs(pixels[..., :2] - values).max() < 1e-6 and not pixels[..., 2].any()
    if oiio is not None:
        file = oiio.ImageInput.open(str(path)); data_oiio = file.read_image(format='float'); file.close()
        assert data_oiio.shape == (H, W, 3) and np.abs(data_oiio - pixels[..., :3]).max() <= 1e-7
    loaded[name] = pixels[..., :2].astype(np.float64)
trip, used = round_trip(loaded['stmap_undistort.exr'], loaded['stmap_redistort.exr'])
print(f'exported maps: redistort(undistort) max error {trip:.2e} px over {used} interior pixels; OpenImageIO', oiio.VERSION_STRING if oiio else 'unavailable')
assert trip < 1e-3
for xi in (0.0, 0.05, 0.2):
    test_lens = dict(lens, xi=xi, fx=F_U * (1 + xi), fy=F_U * (1 + xi))
    ud, vd, ok = mei_project(test_lens, x, y, np.ones_like(x))
    bx, by, bz, ok2 = mei_unproject(test_lens, ud, vd)
    analytic = np.hypot(F_U * bx / bz + CX - u, F_U * by / bz + CY - v).max()
    maps = stmaps(test_lens, W, H)
    trip, used = round_trip(*(np.stack([m['R'], m['G']], -1).astype(np.float64) for m in maps[:2]))
    print(f'xi={xi}: analytic round trip {analytic:.1e} px, sampled maps {trip:.2e} px over {used} pixels, invalid {maps[2]}')
    assert ok.all() and ok2.all() and analytic < 1e-6 and trip < 1e-3 and used > W * H * 0.5

# EXR writer through Blender's and OpenImageIO's readers (odd size, partial ZIP block, both compressions).
for compression in ('ZIP', 'NONE'):
    planes = {c: rng.normal(size=(23, 37)).astype(np.float32) * 10 for c in 'RGB'}
    path = write_exr(out / f'writer_{compression}.exr', planes, compression)
    image = bpy.data.images.load(str(path)); image.colorspace_settings.name = 'Non-Color'
    pixels = np.empty(37 * 23 * 4, np.float32); image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(23, 37, 4)[::-1]
    assert max(np.abs(pixels[..., i] - planes[c]).max() for i, c in enumerate('RGB')) <= 1e-7
    if oiio is not None:
        file = oiio.ImageInput.open(str(path)); data_oiio = file.read_image(format='float'); file.close()
        assert max(np.abs(data_oiio[..., i] - planes[c]).max() for i, c in enumerate('RGB')) <= 1e-7

# Alembic: re-import and compare the camera and raw cloud with the Blender scene.
abc = out / 'export' / 'shot_010_camera.abc'
if abc.is_file():
    before = set(bpy.data.objects)
    bpy.ops.wm.alembic_import(filepath=str(abc), as_background_job=False)
    new = [o for o in bpy.data.objects if o not in before]
    abc_cam = next(o for o in new if o.type == 'CAMERA'); abc_cloud = next(o for o in new if o.type == 'MESH')
    worst = 0
    for frame in (START, START + 7, START + COUNT - 1):
        scene.frame_set(frame); bpy.context.view_layer.update()
        a, b = np.array(abc_cam.matrix_world), np.array(cam.matrix_world)
        worst = max(worst, np.abs(a[:3, :3] / np.linalg.norm(a[:3, :3], axis=0) - b[:3, :3] / np.linalg.norm(b[:3, :3], axis=0)).max(), np.abs(a[:3, 3] - b[:3, 3]).max())
    evaluated = abc_cloud.evaluated_get(bpy.context.evaluated_depsgraph_get())
    abc_points = np.array([evaluated.matrix_world @ p.co for p in evaluated.data.vertices])
    print('alembic camera max error', worst, 'cloud points', len(abc_points), 'lens', abc_cam.data.lens)
    assert worst < 1e-4 and len(abc_points) == 500 and np.allclose(abc_points, world.T, atol=1e-4) and abs(abc_cam.data.lens - data.lens) < 1e-3
else:
    assert any('Alembic' in w for w in result['warnings'])
    print('alembic not written:', [w for w in result['warnings'] if 'Alembic' in w])

# Movie plate, no lens model and a centred principal point.
data.animation_data_clear(); data.shift_x = data.shift_y = 0
job['vipe_plate_path'] = str(out / 'shot_010.mov'); del job['vipe_distortion']
plain = export_nuke(bpy.context, job, str(out / 'plain'))
text_plain = Path(plain['nk']).read_text()
assert 'STMap' not in text_plain and 'win_translate' not in text_plain and 'frame_mode "start at"\n frame 1001' in text_plain
assert not (out / 'plain' / 'stmap_undistort.exr').exists()
for bad in ((None, str(out)), (job, '')):
    try:
        export_nuke(bpy.context, *bad)
        raise AssertionError('expected ValueError')
    except ValueError as error:
        print('ValueError:', error)
print('PASS: nuke export', len(result['files']), 'files:', *[Path(f).name for f in result['files']])
