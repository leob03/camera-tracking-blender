"""Merge review and dense extension bundles into one imported job.
Run: blender -b --factory-startup --python-exit-code 1 --python tests/blender_extension_merge.py -- base.zip review_extension.zip dense_extension.zip
"""
import json
from pathlib import Path
import sys
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vipe_remote.bundle import load_bundle
from vipe_remote.importer import import_bundle

base, review, dense = sys.argv[sys.argv.index('--') + 1:][:3]
collection, created = import_bundle(base, start_frame=1001)
assert created and not json.loads(collection['vipe_summary'])['has_witnesses']
count = len(bpy.data.collections)

merged, created = import_bundle(review, start_frame=1001)
summary = json.loads(merged['vipe_summary'])
assert merged == collection and not created and summary['has_witnesses'] and summary['reprojection_px_median'] > 0, summary
crosses = next(c for c in merged.children if c.get('vipe_role') == 'review_crosses_group')
witness_objects = len(crosses.all_objects)

merged, created = import_bundle(dense, start_frame=1001)
summary = json.loads(merged['vipe_summary'])
groups = {c.get('vipe_role'): c for c in merged.children}
assert merged == collection and not created and summary['dense'] and summary['has_witnesses'], summary
assert len(crosses.all_objects) == witness_objects, 'dense merge must keep the review witnesses'
layer = bpy.context.view_layer.layer_collection.children[merged.name].children[groups['dense_environment_group'].name]
assert layer.hide_viewport, 'merged dense geometry starts hidden'
manifest = json.loads(bpy.data.texts[merged['vipe_manifest_text']].as_string())
assert manifest.get('witnesses') and manifest.get('environment') and manifest['version'] == 3
assert len(bpy.data.collections) == count + 3  # dense group with its two sub-collections, nothing else
_, _, cloud = load_bundle(dense)
dense_points = sum(item['view']['point_count'] for item in cloud['environment'])

# Show / Hide toggle from the add-on operator (background mode: no 3D view to shade).
import vipe_remote
vipe_remote.register()
try:
    props = bpy.context.scene.vipe_remote
    props.job_id = collection['vipe_job_id']
    assert bpy.ops.vipe_remote.view_environment(show=True) == {'FINISHED'} and not layer.hide_viewport
    assert bpy.ops.vipe_remote.view_environment(show=False) == {'FINISHED'} and layer.hide_viewport
finally:
    vipe_remote.unregister()
print('PASS extension merge; median held-out px', round(summary['reprojection_px_median'], 3), 'dense points', dense_points, 'timings', sorted(summary['timings']))
