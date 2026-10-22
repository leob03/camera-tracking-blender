"""Plate and review markers in Rendered view and renders.
Run: blender -b --factory-startup --python-exit-code 1 --python tests/blender_review_render.py -- bundle_with_witnesses.zip plate_first_frame_or_movie
"""
from pathlib import Path
import sys
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vipe_remote.importer import import_bundle

bundle, plate = sys.argv[sys.argv.index('--') + 1:][:2]
scene = bpy.context.scene

# An artist's own compositor setup is never replaced.
scene.use_nodes = True
scene.node_tree.nodes.new('CompositorNodeBlur').name = 'Artist blur'
collection, _ = import_bundle(bundle, source_path=plate, start_frame=1001)
assert not collection['vipe_plate_composited'] and 'Artist blur' in scene.node_tree.nodes
for obj in list(collection.all_objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for child in list(collection.children_recursive) + [collection]:
    bpy.data.collections.remove(child)
scene.node_tree.nodes.remove(scene.node_tree.nodes['Artist blur'])

collection, _ = import_bundle(bundle, source_path=plate, start_frame=1001)
assert scene.render.film_transparent and scene.view_settings.view_transform == 'Standard'
assert collection['vipe_plate_composited'], 'default compositor should get the plate'
nodes = scene.node_tree.nodes
assert nodes['ViPE Plate'].clip == scene.camera.data.background_images[0].clip
assert nodes['ViPE Composite'].inputs['Image'].links[0].from_node == nodes['ViPE Render over plate']
groups = {c.get('vipe_role'): c for c in collection.children}
markers = groups['static_tracks_group']
geometry = next(o for o in markers.objects if o.get('vipe_role') == 'static_marker_geometry')
empties = [o for o in markers.objects if o.get('vipe_role') == 'scene_marker']
assert len(geometry.data.vertices) == len(empties) > 0 and not geometry.hide_render and geometry.hide_select
assert geometry.modifiers['Track crosses'].node_group.name == 'ViPE static track crosses'
crosses = [o for o in groups['review_crosses_group'].all_objects]
assert crosses and all(not o.hide_render for o in crosses), 'review crosses render by default'

# Toggle through the add-on operator.
import vipe_remote
vipe_remote.register()
try:
    scene.vipe_remote.job_id = collection['vipe_job_id']
    assert bpy.ops.vipe_remote.markers_in_render(enable=False) == {'FINISHED'}
    assert geometry.hide_render and all(o.hide_render for o in crosses) and not collection['vipe_markers_in_render']
    assert bpy.ops.vipe_remote.markers_in_render(enable=True) == {'FINISHED'} and not geometry.hide_render
finally:
    vipe_remote.unregister()
print('PASS review render; markers', len(empties), 'crosses', len(crosses))
