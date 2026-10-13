import hashlib
import json
from pathlib import Path
import queue
import textwrap
import threading
import time
import uuid

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from .bundle import load_bundle
from .client import Client, Detached, JobCancelled
from .importer import import_bundle


_active = None


class ViPERemotePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    worker_url: StringProperty(name="ComfyUI worker", default="http://127.0.0.1:8189")
    token: StringProperty(name="Bearer token", subtype='PASSWORD', options={'SKIP_SAVE'}, description="Optional gateway token for this Blender session")
    cache_directory: StringProperty(name="Download folder", subtype='DIR_PATH', description="Leave empty to use Blender's user data folder")

    def draw(self, context):
        self.layout.prop(self, 'worker_url')
        self.layout.prop(self, 'token')
        self.layout.prop(self, 'cache_directory')
        self.layout.label(text="GPU inference runs on the worker.")


class ViPERemoteProperties(bpy.types.PropertyGroup):
    source: StringProperty(name="Video plate", subtype='FILE_PATH')
    frame_start: IntProperty(name="First source frame", default=1, min=1)
    frame_end: IntProperty(name="Last source frame", default=0, min=0, description="0 processes to the end of the video; otherwise inclusive")
    max_dimension: IntProperty(name="Processing long edge", default=1280, min=0, description="Pixels; 0 keeps original resolution. Imported camera matches the original plate size")
    review_witnesses: BoolProperty(name="Independent review witnesses", default=True, description="Run CoTracker and place the original review crosses; assesses the frozen camera without improving its solve")
    dense_environment: BoolProperty(name="Dense environment reconstruction", default=False, description="Compute post-SLAM video depth and accumulate static colored surfaces; adds processing time")
    masking: BoolProperty(name="Exclude dynamic categories", default=True)
    mask_phrases: StringProperty(name="Categories", default="person,animal,vehicle,ball,balloon,gun,pet,car,bus")
    lens_mode: EnumProperty(name="Lens", items=[('auto', "Estimate and refine", "GeoCalib initialization followed by stock ViPE lens optimization"), ('fixed', "Known focal length (locked)", "Use a supplied focal length with a centered principal point and keep it fixed")], default='auto')
    focal_mm: FloatProperty(name="Focal length (mm)", default=35.0, min=0.1)
    sensor_mm: FloatProperty(name="Sensor width (mm)", default=36.0, min=0.1)
    max_points: IntProperty(name="Cloud point limit", default=30000, min=1, max=1000000)
    marker_count: IntProperty(name="3D scene markers", default=100, min=0, max=2000)
    start_frame: IntProperty(name="Blender start frame", default=1001, min=0, description="Timeline frame for the first selected source frame")
    scene_scale: FloatProperty(name="Scene scale", default=1.0, min=0.000001)
    auto_import: BoolProperty(name="Import when ready", default=True)
    show_connection: BoolProperty(name="Worker connection", default=False)
    show_advanced: BoolProperty(name="Import settings", default=False)
    job_id: StringProperty(name="Job ID")
    job_record: StringProperty(options={'HIDDEN'})
    result_path: StringProperty(name="Result bundle", subtype='FILE_PATH')
    status: StringProperty(default="Choose a plate and submit a tracking job.")
    state: StringProperty(default="IDLE")
    progress: FloatProperty(default=0, min=0, max=1, options={'SKIP_SAVE'})
    busy: BoolProperty(default=False, options={'SKIP_SAVE'})
    lens_summary: StringProperty(options={'HIDDEN'})


def preferences(context):
    return context.preferences.addons[__package__].preferences


def settings_from_properties(props):
    return {name: getattr(props, name) for name in ['frame_start', 'frame_end', 'max_dimension', 'masking', 'mask_phrases', 'review_witnesses', 'dense_environment', 'lens_mode', 'focal_mm', 'sensor_mm', 'max_points']}


def _notify_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _import(props, record):
    manifest, _, _ = load_bundle(props.result_path, record.get('source_sha256'))
    if manifest['job_id'] != record['job_id']:
        raise ValueError("The downloaded bundle belongs to a different job.")
    collection, created = import_bundle(props.result_path, source_path=record['source'], start_frame=record['start_frame'], marker_count=record['marker_count'], scene_scale=record['scene_scale'], expected_source_sha256=record.get('source_sha256'))
    props.state = 'COMPLETE'
    props.status = f"Imported {manifest['selection']['count']} camera frames and {manifest['cloud']['point_count']:,} native points." if created else "This job is already imported; its collection was preserved."
    if created:
        crosses = next((c for c in collection.children if c.get('vipe_role') == 'review_crosses_group'), None)
        if crosses is not None and manifest.get('witnesses'):
            props.status += f" Witnesses: {crosses.get('witness_count', 0)}/4. {crosses.get('status', '')}"
    if created and manifest.get('environment'):
        props.status += ' Dense reconstruction ready: use Show Dense Environment.'
    props.progress = 1
    stages = manifest.get('processing', {}).get('lens_stages_original_pixels', {})
    props.lens_summary = ' / '.join(f"{name}: {values[0]:.1f} px" for name, values in [('initial fx', stages.get('initial')), ('final fx', stages.get('post_global_ba'))] if values)
    if bpy.context.screen:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
    return collection


def _run_network(mode, record, url, token, events, cancel, detach):
    def emit(kind, value):
        events.put((kind, value))
    last_progress = [None, 0.0]
    def progress(message, fraction):
        now = time.monotonic()
        if last_progress[0] != message or now - last_progress[1] > 0.2:
            emit('progress', (message, fraction))
            last_progress[:] = [message, now]
    client = Client(url, token)
    try:
        if mode == 'check':
            client.capabilities()
            emit('connected', "Connected to the ViPE worker.")
            return
        if mode == 'log':
            emit('log', client.log(record['job_id']))
            return
        folder = Path(record['folder'])
        folder.mkdir(parents=True, exist_ok=True)
        if mode == 'submit':
            uploaded = client.upload(record['source'], progress, lambda: cancel.is_set() or detach.is_set())
            if cancel.is_set():
                raise JobCancelled("Upload stopped before submission.")
            if detach.is_set():
                raise Detached("Upload finished; no job submitted after monitoring stopped.")
            record['source_sha256'] = uploaded['sha256']
            (folder / 'job.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            emit('record', dict(record))
            workflow = client.submit(uploaded, Path(record['source']).name, record['settings'], record['job_id'])
            (folder / 'workflow_api.json').write_text(json.dumps(workflow, indent=2), encoding='utf-8')
        item = client.wait(record['job_id'], progress, cancel.is_set, detach.is_set)
        result = client.download(item, folder / 'result.zip', progress, lambda: detach.is_set() or cancel.is_set())
        manifest, _, _ = load_bundle(result, record.get('source_sha256'))
        if manifest['job_id'] != record['job_id']:
            raise ValueError("Downloaded result belongs to another job.")
        record['source_sha256'] = manifest['source']['sha256']
        record['result'] = str(result)
        (folder / 'job.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        if record['source']:
            h = hashlib.sha256()
            with Path(record['source']).open('rb') as plate:
                for chunk in iter(lambda: plate.read(1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != record['source_sha256']:
                raise ValueError('The local plate changed after submission. Restore that source version before importing.')
        emit('ready', record)
    except JobCancelled as exc:
        emit('cancelled', str(exc))
    except Detached as exc:
        emit('detached', str(exc))
    except Exception as exc:
        emit('error', str(exc))
    finally:
        emit('finished', None)


def _poll():
    global _active
    active = _active
    if active is None:
        return None
    try:
        props = active['scene'].vipe_remote
    except ReferenceError:
        active['detach'].set()
        _active = None
        return None
    finished = False
    for _ in range(100):
        try:
            kind, value = active['events'].get_nowait()
        except queue.Empty:
            break
        if kind == 'progress':
            props.status, props.progress = value
            props.state = 'UPLOADING' if value[0].startswith('Uploading') else 'DOWNLOADING' if value[0].startswith('Downloading') else 'RUNNING'
        elif kind == 'record':
            props.job_record = json.dumps(value)
        elif kind == 'ready':
            props.job_record = json.dumps(value)
            props.result_path = value['result']
            props.state, props.status, props.progress = 'READY', 'Result downloaded. Ready to import.', 1
            if value['auto_import'] and active['scene'] == bpy.context.scene:
                try:
                    _import(props, value)
                except Exception as exc:
                    props.state, props.status = 'ERROR', f"Import failed: {exc}"
        elif kind in ('cancelled', 'error', 'detached'):
            props.state = 'CANCELLED' if kind == 'cancelled' else 'ERROR' if kind == 'error' else 'READY'
            props.status = value
        elif kind == 'connected':
            props.status = value
        elif kind == 'log':
            text = bpy.data.texts.get('ViPE Job Log') or bpy.data.texts.new('ViPE Job Log')
            text.clear()
            text.write(value)
            props.status = "Log loaded into the 'ViPE Job Log' text block."
        elif kind == 'finished':
            finished = True
    _notify_redraw()
    if finished:
        props.busy = False
        _active = None
        return None
    return 0.25


def _launch(context, mode, record):
    global _active
    if _active is not None:
        raise RuntimeError("A request is already active in this Blender session.")
    prefs = preferences(context)
    events, cancel, detach = queue.Queue(), threading.Event(), threading.Event()
    url = record.get('server', prefs.worker_url)
    thread = threading.Thread(target=_run_network, args=(mode, record, url, prefs.token, events, cancel, detach), daemon=True, name='ViPE-HTTP')
    _active = {'thread': thread, 'events': events, 'cancel': cancel, 'detach': detach, 'scene': context.scene, 'mode': mode}
    context.scene.vipe_remote.busy = True
    thread.start()
    bpy.app.timers.register(_poll, first_interval=0.1)


class VIPE_OT_check(bpy.types.Operator):
    bl_idname = 'vipe_remote.check'
    bl_label = 'Check Connection'
    def execute(self, context):
        _launch(context, 'check', {})
        return {'FINISHED'}


class VIPE_OT_submit(bpy.types.Operator):
    bl_idname = 'vipe_remote.submit'
    bl_label = 'Submit Tracking Job'
    bl_description = 'Upload this plate to ComfyUI, solve on the worker, and retrieve camera and native points'
    @classmethod
    def poll(cls, context):
        return _active is None and bool(context.scene.vipe_remote.source)
    def execute(self, context):
        props, prefs = context.scene.vipe_remote, preferences(context)
        source = Path(bpy.path.abspath(props.source))
        if not source.is_file():
            self.report({'ERROR'}, "Choose an existing video plate.")
            return {'CANCELLED'}
        identifier = str(uuid.uuid4())
        cache = Path(bpy.path.abspath(prefs.cache_directory)) if prefs.cache_directory else Path(bpy.utils.user_resource('DATAFILES', path='vipe_remote', create=True))
        record = {'job_id': identifier, 'source': str(source.resolve()), 'server': prefs.worker_url, 'folder': str(cache / identifier), 'settings': settings_from_properties(props), 'start_frame': props.start_frame, 'marker_count': props.marker_count, 'scene_scale': props.scene_scale, 'auto_import': props.auto_import}
        props.job_id, props.job_record = identifier, json.dumps(record)
        props.result_path, props.state, props.status, props.progress = '', 'UPLOADING', 'Connecting to worker', 0
        props.lens_summary = ''
        _launch(context, 'submit', record)
        return {'FINISHED'}


def _has_remote_job(props):
    if not props.job_record:
        return False
    record = json.loads(props.job_record)
    return bool(record.get('server') and record.get('folder'))


class VIPE_OT_resume(bpy.types.Operator):
    bl_idname = 'vipe_remote.resume'
    bl_label = 'Resume Job'
    bl_description = 'Reconnect to the saved job without submitting it again'
    @classmethod
    def poll(cls, context):
        return _active is None and _has_remote_job(context.scene.vipe_remote)
    def execute(self, context):
        props = context.scene.vipe_remote
        record = json.loads(props.job_record)
        saved = Path(record['folder']) / 'job.json'
        if saved.is_file():
            record = json.loads(saved.read_text(encoding='utf-8'))
        props.status = 'Reconnecting to saved job'
        _launch(context, 'resume', record)
        return {'FINISHED'}


class VIPE_OT_cancel(bpy.types.Operator):
    bl_idname = 'vipe_remote.cancel'
    bl_label = 'Cancel Job'
    @classmethod
    def poll(cls, context):
        return _active is not None
    def execute(self, context):
        _active['cancel'].set()
        context.scene.vipe_remote.status = 'Requesting cancellation'
        return {'FINISHED'}


class VIPE_OT_import_cached(bpy.types.Operator):
    bl_idname = 'vipe_remote.import_cached'
    bl_label = 'Import Result'
    bl_options = {'REGISTER', 'UNDO'}
    @classmethod
    def poll(cls, context):
        props = context.scene.vipe_remote
        return _active is None and bool(props.result_path) and bool(props.job_record)
    def execute(self, context):
        try:
            _import(context.scene.vipe_remote, json.loads(context.scene.vipe_remote.job_record))
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class VIPE_OT_import_file(bpy.types.Operator, ImportHelper):
    bl_idname = 'vipe_remote.import_file'
    bl_label = 'Import Saved Bundle'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.zip'
    filter_glob: StringProperty(default='*.zip', options={'HIDDEN'})
    @classmethod
    def poll(cls, context):
        return _active is None
    def execute(self, context):
        props = context.scene.vipe_remote
        try:
            manifest, _, _ = load_bundle(self.filepath)
            # A selected local plate must match the bundle before it can be
            # shown behind this camera. An empty selection allows geometry-only import.
            source = ''
            if props.source.strip():
                plate = Path(bpy.path.abspath(props.source))
                if not plate.is_file():
                    raise ValueError('Select the original video plate, or clear Video plate for a geometry-only import.')
                digest = hashlib.sha256()
                with plate.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != manifest['source']['sha256']:
                    raise ValueError('Selected Video plate does not match this bundle. Select its original video, or clear the field.')
                source = str(plate.resolve())
            record = {'job_id': manifest['job_id'], 'source': source, 'start_frame': props.start_frame, 'marker_count': props.marker_count, 'scene_scale': props.scene_scale, 'source_sha256': manifest['source']['sha256']}
            props.result_path = self.filepath
            _import(props, record)
            props.job_id = record['job_id']
            props.job_record = json.dumps(record)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class VIPE_OT_log(bpy.types.Operator):
    bl_idname = 'vipe_remote.log'
    bl_label = 'Read Job Log'
    @classmethod
    def poll(cls, context):
        return _active is None and _has_remote_job(context.scene.vipe_remote)
    def execute(self, context):
        _launch(context, 'log', json.loads(context.scene.vipe_remote.job_record))
        return {'FINISHED'}


class VIPE_OT_view_environment(bpy.types.Operator):
    bl_idname = 'vipe_remote.view_environment'
    bl_label = 'Show Dense Environment'
    bl_description = 'Reveal the selected job’s dense samples in material preview; camera solve stays unchanged'
    def execute(self, context):
        identifier = context.scene.vipe_remote.job_id
        for parent in context.scene.collection.children:
            if parent.get('vipe_job_id') != identifier:
                continue
            for child in parent.children:
                if child.get('vipe_role') == 'dense_environment_group':
                    layer = context.view_layer.layer_collection.children[parent.name].children[child.name]
                    layer.hide_viewport = False
                    for obj in context.selected_objects: obj.select_set(False)
                    for obj in child.all_objects:
                        if obj.get('vipe_role') == 'dense_samples': obj.select_set(True)
                    context.space_data.shading.type = 'MATERIAL'
                    bpy.ops.view3d.view_selected(use_all_regions=False)
                    return {'FINISHED'}
        self.report({'INFO'}, 'This job has no dense environment. Enable Dense environment reconstruction before submitting.')
        return {'CANCELLED'}


class VIPE_PT_panel(bpy.types.Panel):
    bl_label = 'ViPE Remote Tracking'
    bl_idname = 'VIPE_PT_remote_tracking'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'ViPE Remote'
    def draw(self, context):
        layout, props, prefs = self.layout, context.scene.vipe_remote, preferences(context)
        wrap_width = max(20, int(context.region.width / 7) - 6)
        box = layout.box()
        box.prop(props, 'show_connection', icon='TRIA_DOWN' if props.show_connection else 'TRIA_RIGHT', emboss=False)
        if props.show_connection:
            connection = box.column()
            connection.enabled = not props.busy
            connection.prop(prefs, 'worker_url', text='Worker')
            connection.prop(prefs, 'token')
            connection.prop(prefs, 'cache_directory')
            connection.operator('vipe_remote.check', icon='CHECKMARK')
        fields = layout.column()
        fields.enabled = not props.busy
        fields.prop(props, 'source')
        frames = fields.box()
        frames.label(text='Source range (last 0 = end of video)')
        frames.prop(props, 'frame_start')
        frames.prop(props, 'frame_end')
        fields.prop(props, 'max_dimension')
        fields.prop(props, 'lens_mode')
        if props.lens_mode == 'fixed':
            fields.prop(props, 'focal_mm')
            fields.prop(props, 'sensor_mm')
        fields.prop(props, 'masking')
        if props.masking:
            fields.prop(props, 'mask_phrases')
        fields.prop(props, 'review_witnesses')
        fields.prop(props, 'dense_environment')
        fields.prop(props, 'show_advanced', icon='TRIA_DOWN' if props.show_advanced else 'TRIA_RIGHT', emboss=False)
        if props.show_advanced:
            fields.prop(props, 'max_points')
            fields.prop(props, 'marker_count')
            fields.prop(props, 'start_frame')
            fields.prop(props, 'scene_scale')
        fields.prop(props, 'auto_import')
        row = layout.row()
        row.scale_y = 1.4
        if props.busy:
            if _active and _active['mode'] in ('submit', 'resume'):
                row.operator('vipe_remote.cancel', icon='CANCEL', text='Stop Download' if props.state == 'DOWNLOADING' else 'Cancel Job')
            else:
                row.label(text='Contacting worker…')
        else:
            row.operator('vipe_remote.submit', icon='CAMERA_DATA')
        status = layout.box()
        for line in textwrap.wrap(props.status, width=wrap_width)[:5]:
            status.label(text=line)
        if props.lens_summary and props.state == 'COMPLETE':
            for line in textwrap.wrap(props.lens_summary, width=wrap_width):
                status.label(text=line)
        if props.busy:
            status.progress(factor=props.progress, type='BAR', text='Current stage')
        if props.job_id:
            status.label(text='Job ' + props.job_id[:8])
            row = layout.row(align=True)
            row.enabled = not props.busy
            row.operator('vipe_remote.resume', icon='FILE_REFRESH')
            row.operator('vipe_remote.log', icon='TEXT')
        row = layout.row(align=True)
        row.enabled = not props.busy
        row.operator('vipe_remote.import_cached', icon='IMPORT')
        row.operator('vipe_remote.import_file', text='Saved Bundle', icon='FILE_FOLDER')
        layout.operator('vipe_remote.view_environment', icon='MESH_GRID')
        layout.label(text='Native points are reconstruction hypotheses.', icon='INFO')


@persistent
def _before_load(_):
    global _active
    if _active is not None:
        _active['detach'].set()
        _active = None
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)


CLASSES = [ViPERemotePreferences, ViPERemoteProperties, VIPE_OT_check, VIPE_OT_submit, VIPE_OT_resume, VIPE_OT_cancel, VIPE_OT_import_cached, VIPE_OT_import_file, VIPE_OT_log, VIPE_OT_view_environment, VIPE_PT_panel]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.vipe_remote = PointerProperty(type=ViPERemoteProperties)
    bpy.app.handlers.load_pre.append(_before_load)


def unregister():
    _before_load(None)
    if _before_load in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_before_load)
    if hasattr(bpy.types.Scene, 'vipe_remote'):
        del bpy.types.Scene.vipe_remote
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
