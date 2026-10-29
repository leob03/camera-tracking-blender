"""Make the plate and review markers visible in Rendered view and in renders.

Camera background images are viewport-only references: Rendered shading draws the world over
them and renders never include them. The job scene therefore renders with a transparent film,
composites the plate underneath, and gives the static tracks render-visible marker geometry.
"""
import bpy
import numpy as np

MARKER_COLOR = (1.0, 0.78, 0.0, 1.0)
MARKER_ROLE = 'static_marker_geometry'
NODE_PREFIX = 'ViPE '


def _marker_material():
    material = bpy.data.materials.get('ViPE static track markers')
    if material is None:
        material = bpy.data.materials.new('ViPE static track markers')
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = MARKER_COLOR
        emission.inputs['Strength'].default_value = 1.0
        output = nodes.new('ShaderNodeOutputMaterial')
        material.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
        material.diffuse_color = MARKER_COLOR  # Solid shading uses the viewport colour
    return material


def _marker_nodes():
    group = bpy.data.node_groups.get('ViPE static track crosses')
    if group is not None:
        return group
    group = bpy.data.node_groups.new('ViPE static track crosses', 'GeometryNodeTree')
    group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = group.nodes, group.links
    inputs, output = nodes.new('NodeGroupInput'), nodes.new('NodeGroupOutput')
    join = nodes.new('GeometryNodeJoinGeometry')
    for size in ((2, .12, .12), (.12, 2, .12), (.12, .12, 2)):
        bar = nodes.new('GeometryNodeMeshCube')
        bar.inputs['Size'].default_value = size
        links.new(bar.outputs['Mesh'], join.inputs['Geometry'])
    material = nodes.new('GeometryNodeSetMaterial')
    material.inputs['Material'].default_value = _marker_material()
    links.new(join.outputs['Geometry'], material.inputs['Geometry'])
    size = nodes.new('GeometryNodeInputNamedAttribute')
    size.data_type = 'FLOAT'
    size.inputs['Name'].default_value = 'marker_size'
    instance = nodes.new('GeometryNodeInstanceOnPoints')
    links.new(inputs.outputs['Geometry'], instance.inputs['Points'])
    links.new(material.outputs['Geometry'], instance.inputs['Instance'])
    links.new(size.outputs['Attribute'], instance.inputs['Scale'])
    links.new(instance.outputs['Instances'], output.inputs['Geometry'])
    return group


def build_marker_geometry(marker_collection, root):
    """One render-visible mesh of crosses at the static-track empties (which stay for selection)."""
    for obj in list(marker_collection.objects):
        if obj.get('vipe_role') == MARKER_ROLE:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    markers = [obj for obj in marker_collection.objects if obj.get('vipe_role') == 'scene_marker']
    if not markers:
        return None
    mesh = bpy.data.meshes.new('ViPE static track markers')
    mesh.from_pydata([tuple(marker.location) for marker in markers], [], [])
    sizes = mesh.attributes.new('marker_size', 'FLOAT', 'POINT')
    sizes.data.foreach_set('value', np.array([marker.empty_display_size for marker in markers], dtype=np.float32))
    obj = bpy.data.objects.new('Static track markers', mesh)
    marker_collection.objects.link(obj)
    obj.parent = root
    obj.show_in_front = True
    obj.hide_select = True  # select the empties; this mesh only draws them
    obj['vipe_role'] = MARKER_ROLE
    obj.modifiers.new('Track crosses', 'NODES').node_group = _marker_nodes()
    return obj


def set_markers_in_render(collection, enabled):
    """Show or hide static-track markers and review crosses in renders."""
    for obj in list(collection.all_objects):  # changing hide_render rebuilds all_objects on Blender 5.2, mid-loop
        if obj.get('vipe_role') in (MARKER_ROLE, 'review_cross', 'review_cross_part') or (obj.parent and obj.parent.get('vipe_role') == 'review_cross'):
            obj.hide_render = not enabled
    collection['vipe_markers_in_render'] = enabled


BLENDER_5_COMPOSITOR = 'compositing_node_group' in bpy.types.Scene.bl_rna.properties


def _owned(tree):
    return all(node.type in ('R_LAYERS', 'COMPOSITE', 'GROUP_OUTPUT') or node.name.startswith(NODE_PREFIX) for node in tree.nodes)


def compositor_tree(scene, create=False):
    """The scene's compositor nodes: scene.node_tree up to Blender 4, a compositor node group from Blender 5."""
    if BLENDER_5_COMPOSITOR:
        tree = scene.compositing_node_group
        if tree is None and create:
            tree = scene.compositing_node_group = bpy.data.node_groups.new(NODE_PREFIX + 'plate compositor', 'CompositorNodeTree')
        if create and not any(item.item_type == 'SOCKET' and item.in_out == 'OUTPUT' and item.name == 'Image' for item in tree.interface.items_tree):
            tree.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
        return tree
    if create:
        scene.use_nodes = True
    return scene.node_tree if scene.use_nodes else None


def setup_plate_compositor(scene, clip, undistort=False):
    """Composite the plate under the render; never replaces an artist's own compositor setup."""
    existing = compositor_tree(scene)
    if existing is not None and not _owned(existing):
        return False
    tree = compositor_tree(scene, create=True)
    for node in [n for n in tree.nodes if n.name.startswith(NODE_PREFIX) or n.type in ('R_LAYERS', 'COMPOSITE', 'GROUP_OUTPUT')]:
        tree.nodes.remove(node)
    def add(kind, name, x):
        node = tree.nodes.new(kind)
        node.name = node.label = NODE_PREFIX + name
        node.location = (x, 0)
        return node
    layers = add('CompositorNodeRLayers', 'Render', -500)
    layers.location.y = -300
    plate = add('CompositorNodeMovieClip', 'Plate', -700)
    plate.clip = clip
    image = plate.outputs['Image']
    if undistort:
        distortion = add('CompositorNodeMovieDistortion', 'Undistort plate', -500)
        distortion.clip = clip
        if BLENDER_5_COMPOSITOR:
            distortion.inputs['Type'].default_value = 'Undistort'
        else:
            distortion.distortion_type = 'UNDISTORT'
        tree.links.new(image, distortion.inputs['Image'])
        image = distortion.outputs['Image']
    scale = add('CompositorNodeScale', 'Plate to render size', -300)
    if BLENDER_5_COMPOSITOR:  # node options became menu sockets in Blender 5
        scale.inputs['Type'].default_value, scale.inputs['Frame Type'].default_value = 'Render Size', 'Stretch'
    else:
        scale.space, scale.frame_method = 'RENDER_SIZE', 'STRETCH'
    tree.links.new(image, scale.inputs['Image'])
    over = add('CompositorNodeAlphaOver', 'Render over plate', -50)
    background, foreground = (over.inputs['Background'], over.inputs['Foreground']) if BLENDER_5_COMPOSITOR else (over.inputs[1], over.inputs[2])
    tree.links.new(scale.outputs['Image'], background)
    tree.links.new(layers.outputs['Image'], foreground)
    # Blender 5 has no Composite node: the group output is the final image.
    composite = add('NodeGroupOutput' if BLENDER_5_COMPOSITOR else 'CompositorNodeComposite', 'Composite', 200)
    tree.links.new(over.outputs['Image'], composite.inputs['Image'])
    viewer = add('CompositorNodeViewer', 'Viewer', 200)
    viewer.location.y = -200
    tree.links.new(over.outputs['Image'], viewer.inputs['Image'])
    return True


def setup_scene(scene):
    """Transparent film so Rendered view shows the plate behind CG; Standard view transform shows the plate unaltered."""
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
