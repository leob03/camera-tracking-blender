"""Import checks for 0.5.0 bundles.
Run: blender -b --factory-startup --python-exit-code 1 --python tests/blender_import_features.py -- camera.zip review.zip distortion.zip sequence.zip proxy_first_frame.jpg plate.mp4
"""
import json
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vipe_remote.bundle import load_bundle
from vipe_remote.importer import import_bundle

camera_zip, review_zip, distortion_zip, sequence_zip, proxy_frame, plate = sys.argv[sys.argv.index('--') + 1:][:6]
scene = bpy.context.scene

# Camera-only job: real filmback, constant lens without keys, square pixels.
manifest, camera, _ = load_bundle(camera_zip)
collection, created = import_bundle(camera_zip, source_path=plate, start_frame=1001, sensor_width=24.89)
cam = bpy.data.objects[collection['vipe_camera']]
assert created and abs(cam.data.sensor_width - 24.89) < 1e-6
expected_mm = camera['intrinsics'][0, 0] / manifest['source']['width'] * 24.89
assert abs(cam.data.lens - expected_mm) < 1e-4, (cam.data.lens, expected_mm)
assert not (cam.data.animation_data and cam.data.animation_data.action), 'constant lens must not be keyed'
assert scene.render.pixel_aspect_x == 1 and scene.render.pixel_aspect_y == 1
summary = json.loads(collection['vipe_summary'])
assert summary['has_witnesses'] is False and summary['reprojection_px_median'] is None
# Projection of native points is unchanged by the lens/filmback change.
_, _, cloud = load_bundle(camera_zip)
root = next(o for o in collection.objects if o.get('vipe_role') == 'alignment')
scene.frame_set(1001); bpy.context.view_layer.update()
k = camera['intrinsics'][0]; w2c = np.linalg.inv(camera['c2w'][0])
errors = []
for point in cloud['xyz'][:: max(1, len(cloud['xyz']) // 50)]:
    local = w2c[:3, :3] @ point + w2c[:3, 3]
    if local[2] <= 0.01:
        continue
    expected = np.array([k[0] * local[0] / local[2] + k[2], k[1] * local[1] / local[2] + k[3]])
    uv = world_to_camera_view(scene, cam, root.matrix_world @ Vector(point.tolist()))
    errors.append(np.linalg.norm(np.array([uv.x * manifest['source']['width'], (1 - uv.y) * manifest['source']['height']]) - expected))
assert errors and max(errors) < 0.01, max(errors)
print('PASS camera import; max native projection error', max(errors))

# Review job merges into the parent collection.
before = len(bpy.data.collections)
merged, created = import_bundle(review_zip, start_frame=1001)
assert merged == collection and not created and merged['vipe_review_merged'] and len(bpy.data.collections) == before
summary = json.loads(merged['vipe_summary'])
assert summary['has_witnesses'] and summary['reprojection_px_median'] > 0, summary
keys = cam.animation_data.action
assert cam.get('vipe_reprojection_px') is not None
crosses = next(c for c in merged.children if c.get('vipe_role') == 'review_crosses_group')
print('PASS review merge; median held-out px', round(summary['reprojection_px_median'], 3), 'witnesses', crosses.get('witness_count'))

# Distortion job: undistorted pinhole camera, fitted polynomial on the clip, undistorted background.
manifest, camera, _ = load_bundle(distortion_zip)
lens = manifest['camera']['lens_distortion']
dist, created = import_bundle(distortion_zip, source_path=plate, start_frame=2001, sensor_width=36.0)
dcam = bpy.data.objects[dist['vipe_camera']]
assert abs(dcam.data.lens - lens['undistorted_focal_px'] / manifest['source']['width'] * 36.0) < 1e-4
info = json.loads(dist['vipe_distortion'])
clip = dcam.data.background_images[0].clip
assert clip.tracking.camera.distortion_model == 'POLYNOMIAL' and abs(clip.tracking.camera.k1 - info['blender_polynomial_k'][0]) < 1e-6
assert dcam.data.background_images[0].clip_user.use_render_undistorted
print('PASS distortion import; xi', round(lens['xi'], 4), 'k', np.round(info['blender_polynomial_k'], 5), 'fit px', round(info['polynomial_fit_max_px'], 4))

# Image-sequence proxy job: plate pixels, original pattern kept for export, proxy background.
manifest, camera, _ = load_bundle(sequence_zip)
seq, created = import_bundle(sequence_zip, source_path=proxy_frame, start_frame=manifest['source']['first_frame'], plate_path='/plates/synthetic_plate.####.exr')
assert scene.render.resolution_x == manifest['source']['width'] and scene.frame_start == 1001
assert seq['vipe_plate_path'] == '/plates/synthetic_plate.####.exr' and seq['vipe_plate_first_frame'] == 1001
clip = bpy.data.objects[seq['vipe_camera']].data.background_images[0].clip
assert clip.frame_start == 1001 and clip.frame_duration == manifest['selection']['count'], (clip.frame_start, clip.frame_duration)
print('PASS sequence import; plate', manifest['source']['width'], 'x', manifest['source']['height'], 'clip frames', clip.frame_duration)
