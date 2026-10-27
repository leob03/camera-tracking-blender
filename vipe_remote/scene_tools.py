"""Align a solved job by moving its alignment parent; camera solve and point layout stay rigid."""
import json
import math

import bmesh
import bpy
from bpy.props import FloatProperty
from mathutils import Matrix, Vector
import numpy as np


SKIP_ROLES = {'static_marker_geometry', 'alignment', 'camera', 'native_cloud', 'dense_samples', 'dense_surface', 'review_cross_part', 'gaussian_splat'}


def _record(props):
    try:
        record = json.loads(props.job_record) if props.job_record else {}
    except ValueError:
        return {}
    return record if isinstance(record, dict) else {}


def job_collection(context):
    """The scene collection of the selected job; a review job resolves to the parent it merged into."""
    props = context.scene.vipe_remote
    record = _record(props)
    candidates = [record.get('parent_job_id')] if record.get('mode') == 'review' and record.get('job_id') == props.job_id else []
    for identifier in candidates + [props.job_id]:
        if identifier:
            for collection in context.scene.collection.children:
                if collection.get('vipe_job_id') == identifier:
                    return collection
    return None


def alignment_root(context):
    collection = job_collection(context)
    if collection is None:
        return None
    return next((obj for obj in collection.objects if obj.get('vipe_role') == 'alignment'), None)


def selected_points(context):
    """World positions of selected markers (Object Mode) or vertices (Edit Mode); the active one is last."""
    if context.mode == 'EDIT_MESH':
        points, last = [], []
        for obj in context.objects_in_mode_unique_data:
            if obj.type != 'MESH':
                continue
            bm = bmesh.from_edit_mesh(obj.data)
            active = bm.select_history.active
            for vertex in bm.verts:
                if vertex.select:
                    (last if vertex is active else points).append(obj.matrix_world @ vertex.co)
        return points + last
    active = context.view_layer.objects.active
    objects = [obj for obj in context.selected_objects if obj.type != 'CAMERA' and obj.get('vipe_role') not in SKIP_ROLES]
    objects.sort(key=lambda obj: obj == active)
    return [obj.matrix_world.translation.copy() for obj in objects]


def _apply(context, root, transform):
    if 'vipe_original_matrix' not in root:
        root['vipe_original_matrix'] = [value for row in root.matrix_world for value in row]
    if 'vipe_scene_scale' not in root:
        root['vipe_scene_scale'] = float(_record(context.scene.vipe_remote).get('scene_scale', sum(root.scale) / 3))
    root.matrix_world = transform @ root.matrix_world
    context.view_layer.update()


class _AlignmentTool:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if alignment_root(context) is None:
            cls.poll_message_set('Import or load a ViPE job first.')
            return False
        return True

    def execute(self, context):
        root = alignment_root(context)
        points = selected_points(context)
        try:
            transform, message = self.transform(context, [Vector(p) for p in points])
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        _apply(context, root, transform)
        self.report({'INFO'}, message)
        return {'FINISHED'}


class VIPE_OT_set_floor(_AlignmentTool, bpy.types.Operator):
    bl_idname = 'vipe_remote.set_floor'
    bl_label = 'Set Floor'
    bl_description = 'Fit a plane through 3+ selected markers or vertices and make it the ground (Z = 0, centroid at the origin, camera above)'

    def transform(self, context, points):
        if len(points) < 3:
            raise ValueError('Select at least three markers, or three vertices in Edit Mode.')
        array = np.array([tuple(p) for p in points])
        centroid = array.mean(axis=0)
        _, spread, axes = np.linalg.svd(array - centroid)
        if spread[0] <= 1e-9 or spread[1] <= 1e-6 * spread[0]:
            raise ValueError('The selected points are collinear; choose points that span an area.')
        normal, centroid = Vector(axes[2].tolist()), Vector(centroid.tolist())
        collection = job_collection(context)
        camera = bpy.data.objects.get(collection.get('vipe_camera', '')) if collection else None
        camera = camera or context.scene.camera
        if camera is not None:
            if (camera.matrix_world.translation - centroid).dot(normal) < 0:
                normal.negate()
        elif normal.z < 0:
            normal.negate()
        rotation = normal.rotation_difference(Vector((0.0, 0.0, 1.0))).to_matrix().to_4x4()
        return rotation @ Matrix.Translation(-centroid), f'Floor set from {len(points)} points.'


class VIPE_OT_set_origin(_AlignmentTool, bpy.types.Operator):
    bl_idname = 'vipe_remote.set_origin'
    bl_label = 'Set Origin'
    bl_description = 'Move the solve so the centroid of the selected markers or vertices is the world origin'

    def transform(self, context, points):
        if not points:
            raise ValueError('Select one or more markers, or vertices in Edit Mode.')
        centroid = sum(points, Vector()) / len(points)
        return Matrix.Translation(-centroid), f"Origin set from {len(points)} point{'s' if len(points) > 1 else ''}."


class VIPE_OT_set_scale(_AlignmentTool, bpy.types.Operator):
    bl_idname = 'vipe_remote.set_scale'
    bl_label = 'Set Scale'
    bl_description = 'Scale the solve uniformly about the world origin so two selected points are a known distance apart'
    distance: FloatProperty(name='Distance', default=1.0, min=1e-6, subtype='DISTANCE', unit='LENGTH', options={'SKIP_SAVE'})

    def invoke(self, context, event):
        points = selected_points(context)
        if len(points) == 2:
            self.distance = max((points[1] - points[0]).length, 1e-6)
        return context.window_manager.invoke_props_dialog(self)

    def transform(self, context, points):
        if len(points) != 2:
            raise ValueError('Select exactly two markers or vertices.')
        current = (points[1] - points[0]).length
        if current <= 1e-9:
            raise ValueError('The two selected points coincide.')
        return Matrix.Scale(self.distance / current, 4), f'Scale set: points are {self.distance:g} apart.'


class VIPE_OT_set_axis(_AlignmentTool, bpy.types.Operator):
    bl_idname = 'vipe_remote.set_axis'
    bl_label = 'Set X Axis'
    bl_description = 'Rotate the solve about world Z so the direction to the active point (from the other selected point) is +X'

    def transform(self, context, points):
        if len(points) != 2:
            raise ValueError('Select exactly two markers or vertices; the active one sets the +X direction.')
        direction = points[1] - points[0]
        if math.hypot(direction.x, direction.y) <= 1e-9:
            raise ValueError('The two points are vertically aligned; choose points separated across the floor.')
        return Matrix.Rotation(-math.atan2(direction.y, direction.x), 4, 'Z'), 'X axis set.'


class VIPE_OT_reset_alignment(_AlignmentTool, bpy.types.Operator):
    bl_idname = 'vipe_remote.reset_alignment'
    bl_label = 'Reset Alignment'
    bl_description = 'Restore the imported alignment of this job'

    def execute(self, context):
        root = alignment_root(context)
        original = root.get('vipe_original_matrix')
        if original is not None and len(original) == 16:
            values = list(original)
            root.matrix_world = Matrix([values[0:4], values[4:8], values[8:12], values[12:16]])
        else:
            scale = float(root.get('vipe_scene_scale', _record(context.scene.vipe_remote).get('scene_scale', 1.0)))
            root.location, root.rotation_euler, root.scale = (0, 0, 0), (math.radians(-90), 0, 0), (scale,) * 3
        context.view_layer.update()
        return {'FINISHED'}


CLASSES = [VIPE_OT_set_floor, VIPE_OT_set_origin, VIPE_OT_set_scale, VIPE_OT_set_axis, VIPE_OT_reset_alignment]
