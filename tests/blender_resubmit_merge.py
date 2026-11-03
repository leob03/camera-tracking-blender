"""A resubmission that reuses the scene's camera solve adds its new outputs to the existing collection.

blender -b --factory-startup --python-exit-code 1 --python tests/blender_resubmit_merge.py -- camera_only.zip with_outputs.zip
Both bundles come from the same plate and solve settings: the first without optional outputs, the second
resubmitted with review witnesses and dense geometry (the worker reuses the solve and records camera_reused_from).
"""
from pathlib import Path
import json
import sys
import bpy
import addon_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
camera_only, with_outputs = sys.argv[sys.argv.index('--') + 1:][:2]
addon_utils.enable('vipe_remote', default_set=True)
from vipe_remote.importer import import_bundle, job_ids

scene = bpy.context.scene
bpy.data.objects.remove(bpy.data.objects['Cube'])
first, created = import_bundle(camera_only, start_frame=1001)
assert created and first.name.startswith('Camera Track | ')
roles = {c.get('vipe_role') for c in first.children}
assert 'dense_environment_group' not in roles and not json.loads(first['vipe_summary']).get('has_witnesses')
objects, collections = len(bpy.data.objects), len(scene.collection.children)

merged, created = import_bundle(with_outputs, start_frame=1001)
assert merged == first and not created and len(scene.collection.children) == collections, 'no second collection for the same solve'
assert json.loads(first['vipe_last_merge']) == ['review witnesses', 'dense environment']
assert 'dense_environment_group' in {c.get('vipe_role') for c in first.children} and json.loads(first['vipe_summary'])['has_witnesses']
assert len(job_ids(first)) == 2 and len(bpy.data.objects) > objects

again, created = import_bundle(with_outputs, start_frame=1001)
assert again == first and not created and len(scene.collection.children) == collections, 're-importing changes nothing'

with bpy.context.temp_override(scene=bpy.data.scenes.new('fresh')):
    base, _ = import_bundle(camera_only, start_frame=1001)
    other, created = import_bundle(with_outputs, start_frame=1001, marker_count=50)
    assert created and other != base, 'a resubmission with a different scene setting gets its own collection'
print('PASS resubmit merge:', first.name, sorted(i[:8] for i in job_ids(first)))
