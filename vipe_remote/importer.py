"""Import frozen camera transforms and native landmarks on Blender's main thread."""
import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
import numpy as np

from .bundle import load_bundle
from .lens_math import polynomial_from_mei, review_errors


CAMERA_AXES = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


def _linear_keys(block):
    animation = block.animation_data
    if not animation or not animation.action:
        return
    action = animation.action
    if bpy.app.version < (4, 4, 0):
        curves = action.fcurves
    else:
        curves = [curve for layer in action.layers for strip in layer.strips for bag in strip.channelbags for curve in bag.fcurves]
    for curve in curves:
        for key in curve.keyframe_points:
            key.interpolation = 'LINEAR'


def _make_cloud(collection, root, points, colors, radius):
    mesh = bpy.data.meshes.new("ViPE native points")
    mesh.from_pydata(points.tolist(), [], [])
    mesh.update()
    color = mesh.color_attributes.new(name="Color", type='FLOAT_COLOR', domain='POINT')
    rgba = np.ones((len(points), 4), dtype=np.float32)
    # Video RGB is display-encoded; Blender shader colors are linear.
    rgba[:, :3] = np.where(colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4)
    color.data.foreach_set('color', rgba.reshape(-1))
    obj = bpy.data.objects.new("Native cloud", mesh)
    collection.objects.link(obj)
    obj.parent = root
    obj["vipe_role"] = "native_cloud"
    obj["classification"] = "Post-BA native map; reconstructed hypotheses, not surveyed points"
    material = bpy.data.materials.new("ViPE point colors")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    attribute = nodes.new('ShaderNodeAttribute')
    attribute.attribute_name = 'Color'
    attribute.attribute_type = 'INSTANCER'
    emission = nodes.new('ShaderNodeEmission')
    output = nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(attribute.outputs['Color'], emission.inputs['Color'])
    material.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
    mesh.materials.append(material)
    group = bpy.data.node_groups.new("ViPE point display", 'GeometryNodeTree')
    group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    size = group.interface.new_socket(name="Point radius", in_out='INPUT', socket_type='NodeSocketFloat')
    size.default_value = radius
    size.min_value = 0.000001
    inputs = group.nodes.new('NodeGroupInput')
    output_node = group.nodes.new('NodeGroupOutput')
    sphere = group.nodes.new('GeometryNodeMeshIcoSphere')
    sphere.inputs['Subdivisions'].default_value = 1
    set_material = group.nodes.new('GeometryNodeSetMaterial')
    set_material.inputs['Material'].default_value = material
    instances = group.nodes.new('GeometryNodeInstanceOnPoints')
    group.links.new(inputs.outputs['Point radius'], sphere.inputs['Radius'])
    group.links.new(sphere.outputs['Mesh'], set_material.inputs['Geometry'])
    group.links.new(set_material.outputs['Geometry'], instances.inputs['Instance'])
    group.links.new(inputs.outputs['Geometry'], instances.inputs['Points'])
    group.links.new(instances.outputs['Instances'], output_node.inputs['Geometry'])
    modifier = obj.modifiers.new("Point display", 'NODES')
    modifier.node_group = group
    return obj



def _review_viewports():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.shading.type = 'SOLID'
                space.shading.color_type = 'MATERIAL'
                space.overlay.show_overlays = True
                space.overlay.show_extras = True
                space.overlay.show_relationship_lines = False
                space.region_3d.view_perspective = 'CAMERA'


def _child_collection(parent, name, role):
    child = bpy.data.collections.new(name)
    parent.children.link(child)
    child['vipe_role'] = role
    return child


def _make_original_witnesses(collection, marker_collection, root, camera, audit, manifest, start_frame, marker_count):
    from .artist_witnesses import (
        landmark_geometry, conditioned_landmarks, depth_layers,
        select_screen_safe_track_pins, load_directional_proxy_asset,
        create_supported_probe, create_depth_pin, DEFAULT_VERY_FOREGROUND_MAX_DEPTH_QUANTILE,
    )
    if audit is None:
        collection['status'] = 'Independent review witnesses were not included in this bundle.'
        return
    frames = camera['source_frame_indices']
    xyz = audit['landmark_xyz']
    depths, angles = landmark_geometry(xyz=xyz,
        observation_offsets=audit['observation_offsets'], observation_frames=audit['observation_frames'],
        frame_indices=frames, c2w=camera['c2w'])
    selected = np.flatnonzero(conditioned_landmarks(depths, angles))
    if not len(selected):
        collection['status'] = 'No conditioned independent landmarks support witness placement.'
        return
    layers = depth_layers(depths[selected])
    source = manifest['source']
    near = DEFAULT_VERY_FOREGROUND_MAX_DEPTH_QUANTILE
    chosen = select_screen_safe_track_pins(selected=selected, depths=depths, layers=layers,
        spans=audit['landmark_span_frames'], median_errors=audit['landmark_median_error_evaluation_px'],
        landmark_ids=audit['landmark_ids'], observation_offsets=audit['observation_offsets'],
        observation_frames=audit['observation_frames'], observed_xy=audit['observed_xy'],
        observation_roles=audit['observation_roles'], positive_depth=audit['positive_depth'],
        frame_indices=frames, plate_width=source['width'], plate_height=source['height'],
        very_foreground_max_depth_quantile=near)
    for index in (selected[np.linspace(0,len(selected)-1,min(marker_count,len(selected)),dtype=int)] if marker_count else []):
        marker = bpy.data.objects.new(f"Static track {int(audit['landmark_ids'][index])}", None)
        marker_collection.objects.link(marker)
        marker.parent = root
        marker.location = Vector(xyz[index].tolist())
        marker.empty_display_type = 'PLAIN_AXES'
        marker.empty_display_size = max(1e-6, float(depths[index]) / float(camera['intrinsics'][0,0]) * 15)
        marker.show_in_front = True
        marker['vipe_role'] = 'scene_marker'
        marker['landmark_id'] = str(int(audit['landmark_ids'][index]))
        marker['source'] = 'Independent CoTracker triangulation through the frozen camera'
    collection['status'] = 'Ready' if len(chosen) == 4 else f'Incomplete support: {len(chosen)} of four witnesses; no invented placements.'
    collection['witness_count'] = len(chosen)
    collection['conditioned_landmark_count'] = len(selected)
    if not chosen:
        return
    asset = load_directional_proxy_asset(Path(__file__).parent / 'assets/axis_shaped_cross.glb')
    scene = bpy.context.scene
    old_resolution = (scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage)
    scene.render.resolution_x = 1280
    scene.render.resolution_y = round(1280 * source['height'] / source['width'])
    scene.render.resolution_percentage = 100
    try:
        for item in chosen:
            index = item['source_index']
            ref = item['reference_frame']
            offset = int(np.flatnonzero(frames == ref)[0])
            pose = camera['c2w'][offset]
            depth = float(((xyz[index] - pose[:3,3]) @ pose[:3,:3])[2])
            fx, fy = camera['intrinsics'][offset,:2]
            pin = create_depth_pin(collection=marker_collection, parent=root,
                label='ViPE', slot=item['slot'], layer_name=item['layer_name'],
                point=xyz[index], landmark_id=int(audit['landmark_ids'][index]),
                proximity_class=item['proximity_class'], depth_quantile_0_1=item['depth_percentile'],
                very_foreground_depth_threshold=item['very_foreground_depth_threshold_candidate_units'],
                depth=float(depths[index]), angle=float(angles[index]),
                display_radius=float(depths[index] * 2 / np.sqrt(fx*fy)),
                reference_frame=ref, reference_xy=item['reference_xy'])
            pin['vipe_role'] = 'track_pin'
            probe = create_supported_probe(collection=collection, parent=root,
                label='ViPE', slot=item['slot'], layer_name=item['layer_name'],
                proximity_class=item['proximity_class'], point=xyz[index],
                landmark_id=int(audit['landmark_ids'][index]), reference_frame=ref,
                reference_c2w=pose, camera_depth=depth, depth_percentile=item['depth_percentile'],
                focal_x=float(fx), focal_y=float(fy), plate_width=source['width'], plate_height=source['height'],
                proxy_asset=asset, very_foreground_max_depth_quantile=near)
            probe['vipe_role'] = 'review_cross'
            probe['reference_frame'] = start_frame + offset
            probe['reference_observation_xy_px'] = item['reference_xy']
            probe['slot'] = item['slot']
            probe['track_pin'] = pin.name
            probe.hide_render = True
            for child in probe.children:
                child['vipe_role'] = 'review_cross_part'
                child.hide_render = True
    finally:
        scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = old_resolution
        for _, mesh in asset['parts']:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)


def _summary(collection, manifest, camera, cloud, focal_mm, sensor_width, errors=None):
    stages = manifest.get('processing', {}).get('lens_stages_original_pixels', {})
    lens = manifest['camera'].get('lens_distortion')
    witnesses = manifest.get('witnesses')
    summary = dict(job_id=manifest['job_id'], frames=manifest['selection']['count'], focal_mm=focal_mm,
        focal_px=float(camera['intrinsics'][0, 0]), sensor_width_mm=sensor_width, points=manifest['cloud']['point_count'],
        witness_landmarks=witnesses.get('landmark_count') if witnesses else None,
        reprojection_px_median=float(np.median(errors)) if errors is not None and len(errors) else None,
        reprojection_px_p90=float(np.percentile(errors, 90)) if errors is not None and len(errors) else None,
        distortion=json.loads(collection['vipe_distortion']) if lens and collection.get('vipe_distortion') else None,
        timings=manifest.get('timings_seconds', {}), lens_initial_px=(stages.get('initial') or [None])[0],
        lens_final_px=(stages.get('post_global_ba') or [None])[0], dense=bool(manifest.get('environment')), has_witnesses=bool(witnesses))
    collection['vipe_summary'] = json.dumps(summary)
    return summary


def _key_review_errors(cam, start_frame, frames, errors):
    """Animate per-frame median review error on the camera so it can be graphed against time."""
    if not len(errors):
        return
    order = np.argsort(frames, kind='stable')
    frames, errors = frames[order], errors[order]
    unique, starts = np.unique(frames, return_index=True)
    cam['vipe_reprojection_px'] = 0.0
    for frame, values in zip(unique, np.split(errors, starts[1:])):
        cam['vipe_reprojection_px'] = float(np.median(values))
        cam.keyframe_insert(data_path='["vipe_reprojection_px"]', frame=start_frame + int(frame))
    _linear_keys(cam)


def _merge_review(parent, manifest, camera, cloud, marker_count):
    """Add review witnesses from a review-only job to the already imported parent job."""
    if parent.get('vipe_source_sha256') != manifest['source']['sha256']:
        raise ValueError('The review bundle belongs to a different plate.')
    groups = {child.get('vipe_role'): child for child in parent.children}
    root = next(obj for obj in parent.objects if obj.get('vipe_role') == 'alignment')
    markers, crosses = groups['static_tracks_group'], groups['review_crosses_group']
    for obj in list(markers.objects) + list(crosses.all_objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    start_frame = int(parent.get('vipe_start_frame', 1))
    _make_original_witnesses(crosses, markers, root, camera, cloud.get('audit'), manifest, start_frame, marker_count)
    cam = bpy.data.objects.get(parent.get('vipe_camera', ''))
    frames, errors = review_errors(camera, cloud['audit'], manifest['selection']['first_source_frame'])
    if cam is not None:
        _key_review_errors(cam, start_frame, frames, errors)
    focal_mm = cam.data.lens if cam is not None else 0.0
    sensor = cam.data.sensor_width if cam is not None else 36.0
    _summary(parent, manifest, camera, cloud, focal_mm, sensor, errors)
    report = bpy.data.texts.get(parent.get('vipe_manifest_text', ''))
    if report is not None:
        report.clear()
        report.write(json.dumps(manifest, indent=2))
    parent['vipe_review_merged'] = True
    parent['vipe_review_job_id'] = manifest['job_id']
    return parent


def import_bundle(path, *, source_path="", start_frame=1, marker_count=100, scene_scale=1.0, expected_source_sha256=None, sensor_width=36.0, plate_path="", plate_first_frame=None):
    manifest, camera, cloud = load_bundle(path, expected_source_sha256)
    identifier = manifest['job_id']
    for existing in bpy.context.scene.collection.children:
        if existing.get('vipe_job_id') == identifier:
            return existing, False
    parent = next((c for c in bpy.context.scene.collection.children if manifest.get('review_of') and c.get('vipe_job_id') == manifest['review_of']), None)
    if parent is not None:
        if cloud.get('audit') is None:
            raise ValueError('The review bundle has no independent landmarks.')
        return _merge_review(parent, manifest, camera, cloud, marker_count), False
    if not math.isfinite(sensor_width) or sensor_width <= 0:
        raise ValueError("Sensor width must be positive and finite.")
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("Scene scale must be positive and finite.")
    source = manifest['source']
    width, height = source['width'], source['height']
    count = manifest['selection']['count']
    scene = bpy.context.scene
    old = (scene.camera, scene.frame_start, scene.frame_end, scene.frame_current, scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage, scene.render.fps, scene.render.fps_base, scene.render.pixel_aspect_x, scene.render.pixel_aspect_y)
    label = Path(source['name']).stem
    collection = bpy.data.collections.new(f"ViPE | {label} | {identifier[:8]}")
    scene.collection.children.link(collection)
    collection['vipe_job_id'] = identifier
    collection['vipe_result_bundle'] = str(Path(path).resolve())
    collection['vipe_source_sha256'] = source['sha256']
    try:
        camera_collection = _child_collection(collection, '01 Camera and source plate', 'camera_group')
        marker_collection = _child_collection(collection, '02 Static tracks', 'static_tracks_group')
        cross_collection = _child_collection(collection, '03 Review crosses', 'review_crosses_group')
        cloud_collection = _child_collection(collection, '04 Native point cloud', 'native_cloud_group')
        root = bpy.data.objects.new("ViPE alignment", None)
        collection.objects.link(root)
        root.rotation_euler.x = math.radians(-90)
        root.scale = (scene_scale,) * 3
        root['vipe_role'] = 'alignment'
        root['world_scale_note'] = 'Estimated scale. Adjust this parent to align the solve.'
        data = bpy.data.cameras.new(f"{label} solved camera")
        cam = bpy.data.objects.new(data.name, data)
        camera_collection.objects.link(cam)
        cam.parent = root
        cam.rotation_mode = 'QUATERNION'
        cam['vipe_role'] = 'camera'
        data.type = 'PERSP'
        data.sensor_fit = 'HORIZONTAL'
        data.sensor_width = sensor_width
        data.passepartout_alpha = 0.9
        data.clip_start = 0.001
        data.clip_end = 1000000
        intrinsic = camera['intrinsics']
        ratio = float(intrinsic[0, 0] / intrinsic[0, 1])
        scene.render.resolution_x, scene.render.resolution_y = width, height
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = max(1.0, 1.0 / ratio)
        scene.render.pixel_aspect_y = max(1.0, ratio)
        fps = source['fps_num'] / source['fps_den']
        scene.render.fps = round(fps)
        scene.render.fps_base = round(fps) / fps
        previous_quaternion = None
        # ViPE solves one lens for the shot; a constant lens must not look like a zoom to exporters.
        constant_lens = bool(np.allclose(intrinsic, intrinsic[0], rtol=1e-7, atol=1e-9))
        for offset, (pose, k) in enumerate(zip(camera['c2w'], intrinsic)):
            frame = start_frame + offset
            local = Matrix(pose.tolist()) @ CAMERA_AXES
            cam.location = local.to_translation()
            q = local.to_quaternion()
            if previous_quaternion is not None and previous_quaternion.dot(q) < 0:
                q.negate()
            cam.rotation_quaternion = q
            previous_quaternion = q.copy()
            cam.keyframe_insert(data_path='location', frame=frame)
            cam.keyframe_insert(data_path='rotation_quaternion', frame=frame)
            if constant_lens and offset:
                continue
            fx, fy, cx, cy = map(float, k)
            data.lens = fx / width * sensor_width
            data.shift_x = -(cx - width / 2) / width
            data.shift_y = (cy - height / 2) / width * fx / fy
            if not constant_lens:
                for key in ['lens', 'shift_x', 'shift_y']:
                    data.keyframe_insert(data_path=key, frame=frame)
        _linear_keys(cam)
        _linear_keys(data)
        points = cloud['xyz']
        # Choose an initial display size from the first camera, so distant
        # outliers cannot make every nearby point enormous. The modifier
        # exposes the radius for artist adjustment.
        depth = (points - camera['c2w'][0, :3, 3]) @ camera['c2w'][0, :3, 2]
        front = depth[depth > 0.001]
        radius = max(0.000001, float(np.percentile(front, 25)) / intrinsic[0, 0] * max(width / 1280, 1) * 1.5) if len(front) else 0.001
        if len(points):
            _make_cloud(cloud_collection, root, points, cloud['rgb'], radius)
            cloud_collection.hide_render = True
        if cloud.get('audit') is None and len(points):
            # Legacy bundles retain their native markers, but receive no fake witnesses.
            selected = np.linspace(0, len(points)-1, min(marker_count,len(points)), dtype=int) if marker_count else []
            for index in selected:
                marker = bpy.data.objects.new(f"Native point {int(cloud['ids'][index])}", None)
                marker_collection.objects.link(marker)
                marker.parent = root
                marker.location = Vector(points[index].tolist())
                marker.empty_display_type = 'PLAIN_AXES'
                marker.empty_display_size = radius * 10
                marker.show_in_front = True
                marker['vipe_role'] = 'scene_marker'
                marker['native_point_id'] = int(cloud['ids'][index])
        _make_original_witnesses(cross_collection, marker_collection, root, camera, cloud.get('audit'), manifest, start_frame, marker_count)
        dense_collection = None
        if cloud.get('environment'):
            from .environment import import_environment
            dense_collection = import_environment(collection, root, cloud['environment'])
        lens = manifest['camera'].get('lens_distortion')
        if lens:
            k123, fit_px = polynomial_from_mei(lens, width, height)
            collection['vipe_distortion'] = json.dumps({**lens, 'blender_polynomial_k': k123, 'polynomial_fit_max_px': fit_px})
        if source_path:
            plate = Path(source_path)
            if not plate.is_file():
                raise FileNotFoundError(f"Source plate is missing: {plate}")
            clip = bpy.data.movieclips.load(str(plate.resolve()), check_existing=False)
            clip.frame_start = start_frame - manifest['selection']['first_source_frame']
            collection['vipe_plate_path'] = plate_path or str(plate.resolve())
            tracking = clip.tracking.camera
            tracking.sensor_width = sensor_width
            tracking.focal_length = float(intrinsic[0, 0]) / width * sensor_width
            if lens:
                # The movie stays distorted; the clip's lens model undistorts it behind the pinhole camera.
                tracking.distortion_model = 'POLYNOMIAL'
                tracking.k1, tracking.k2, tracking.k3 = k123
                if hasattr(tracking, 'principal_point_pixels'):
                    tracking.principal_point_pixels = (lens['cx'], height - lens['cy'])
            data.show_background_images = True
            background = data.background_images.new()
            background.source = 'MOVIE_CLIP'
            background.clip = clip
            background.alpha = 1.0
            background.frame_method = 'STRETCH'
            background.display_depth = 'BACK'
            if lens:
                background.clip_user.use_render_undistorted = True
        errors = None
        if cloud.get('audit') is not None:
            frames, errors = review_errors(camera, cloud['audit'], manifest['selection']['first_source_frame'])
            _key_review_errors(cam, start_frame, frames, errors)
        _summary(collection, manifest, camera, cloud, data.lens, sensor_width, errors)
        if plate_path and not source_path:
            collection['vipe_plate_path'] = plate_path
        if plate_first_frame is not None or source.get('first_frame') is not None:
            collection['vipe_plate_first_frame'] = int(plate_first_frame if plate_first_frame is not None else source['first_frame'])
        report = bpy.data.texts.new(f"ViPE {label} {identifier[:8]} result")
        report.write(json.dumps(manifest, indent=2))
        collection['vipe_manifest_text'] = report.name
        collection['vipe_camera'] = cam.name
        collection['vipe_start_frame'] = start_frame
        collection['vipe_layout_version'] = 4
        scene.camera = cam
        scene.frame_start, scene.frame_end = start_frame, start_frame + count - 1
        scene.frame_set(start_frame)
        bpy.context.view_layer.update()
        for ob in bpy.context.selected_objects:
            ob.select_set(False)
        for layer in scene.view_layers:
            layer.update()
            layer.layer_collection.children[collection.name].children[cloud_collection.name].hide_viewport = True
            if dense_collection is not None:
                layer.layer_collection.children[collection.name].children[dense_collection.name].hide_viewport = True
        _review_viewports()
        cam.select_set(True)
        bpy.context.view_layer.objects.active = cam
        return collection, True
    except Exception:
        for obj in list(collection.all_objects):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0:
                if isinstance(data, bpy.types.Camera):
                    bpy.data.cameras.remove(data)
                elif isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
        children = list(collection.children)
        for child in children:
            children.extend(list(child.children))
        for child in reversed(children):
            bpy.data.collections.remove(child)
        bpy.data.collections.remove(collection)
        scene.camera, scene.frame_start, scene.frame_end, frame, scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage, scene.render.fps, scene.render.fps_base, scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = old
        scene.frame_set(frame)
        raise
