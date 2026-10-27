import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import queue
import textwrap
import threading
import time
import uuid

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix

from .bundle import load_bundle
from .client import Client, Detached, JobCancelled
from .importer import _child_collection, import_bundle
from .job_logs import LogBuffer
from .scene_tools import CLASSES as SCENE_TOOL_CLASSES, job_collection
from . import plate_proxy
from .sequence_flow import plate_pattern, prepare_training_plate, prepare_upload


_active = None
_log_buffers = {}
JOB_MODES = ('submit', 'resume', 'review', 'splat')
STAGES = [('upload', "Upload"), ('queue', "Queue"), ('prepare', "Prepare"), ('solve', "Camera solve"), ('witnesses', "Review witnesses"), ('dense', "Dense"), ('splat', "Gaussian splat"), ('package', "Package"), ('download', "Download"), ('import', "Import")]
STAGE_ORDER = [key for key, _ in STAGES]
STAGE_MESSAGES = [('Preparing training plate', 'upload'), ('Uploading training plate', 'upload'), ('Preparing splat', 'prepare'), ('Training Gaussian', 'splat'), ('Rendering Gaussian', 'splat'), ('Gaussian splat and', 'splat'), ('Connecting to worker', 'upload'), ('Preparing plate proxy', 'upload'), ('Checking plate on worker', 'upload'), ('Plate already on worker', 'upload'), ('Uploading video', 'upload'), ('Waiting for worker', 'queue'), ('Starting ViPE worker', 'prepare'), ('Preparing video', 'prepare'), ('Preparing review', 'prepare'), ('Unpacking plate frames', 'prepare'), ('Estimating initial lens', 'solve'), ('Reusing camera solve', 'solve'), ('Tracking camera', 'solve'), ('Saving frozen camera', 'solve'), ('Generating independent CoTracker', 'witnesses'), ('Loading independent CoTracker', 'witnesses'), ('Triangulating independent tracks', 'witnesses'), ('Computing optional dense', 'dense'), ('Dense ', 'dense'), ('Packaging tracking results', 'package'), ('Tracking results ready', 'package'), ('Downloading result', 'download')]
# Published horizontal filmback widths (mm) of the recording area.
SENSOR_PRESETS = [
    ('CUSTOM', "Custom", "Enter the filmback width", 0.0),
    ('FULL_FRAME', "Full frame (36.00 mm)", "36 × 24 mm still-photo full frame", 36.0),
    ('SUPER35', "Super 35 (24.89 mm)", "Super 35 film, 4-perf full aperture, 24.89 × 18.66 mm", 24.89),
    ('ALEXA_35_OG', "ARRI ALEXA 35 4.6K 3:2 Open Gate", "27.99 × 19.22 mm", 27.99),
    ('ALEXA_35_4K', "ARRI ALEXA 35 4K 16:9", "24.88 × 14.00 mm", 24.88),
    ('ALEXA_LF_OG', "ARRI ALEXA LF Open Gate", "36.70 × 25.54 mm", 36.70),
    ('ALEXA_MINI_OG', "ARRI ALEXA Mini 3.4K Open Gate", "28.25 × 18.17 mm", 28.25),
    ('ALEXA_MINI_169', "ARRI ALEXA Mini 2.8K / HD 16:9", "23.76 × 13.37 mm", 23.76),
    ('RED_V_RAPTOR_8K_VV', "RED V-RAPTOR 8K VV", "40.96 × 21.60 mm", 40.96),
    ('MFT', "Micro Four Thirds (17.30 mm)", "17.3 × 13.0 mm", 17.3),
]
SENSOR_WIDTHS = {key: width for key, _, _, width in SENSOR_PRESETS}


def _follow_log_changed(props, context):
    if props.follow_job_log:
        buffer = _log_buffers.get(props.job_id)
        if buffer is not None:
            _display_log(props, buffer.snapshot()[1])


def _probe_plate(props, context):
    """Read plate size and timing once per path change; draw() only formats the stored result."""
    props.plate_info = ''
    if not props.source.strip():
        return
    path = Path(bpy.path.abspath(props.source))
    if not path.is_file():
        props.plate_info = json.dumps({'error': "Video file not found"})
        return
    if plate_proxy.is_sequence(path):
        # Image sequences: one directory listing and one EXR header; no pixels are read.
        try:
            seq = plate_proxy.describe(path)
        except Exception as exc:
            props.plate_info = json.dumps({'error': f"Cannot read this image sequence: {exc}"})
            return
        fps = seq['fps'][0] / seq['fps'][1] if seq['fps'] else context.scene.render.fps / context.scene.render.fps_base
        info = {'width': seq['width'], 'height': seq['height'], 'frames': seq['count'], 'fps': fps, 'fps_rational': list(seq['fps']) if seq['fps'] else None, 'sequence': True, 'first': seq['first'], 'last': seq['last'], 'pattern': seq['pattern'], 'gaps': seq['gaps'], 'pixel_aspect': seq['pixel_aspect']}
        props.plate_info = json.dumps(info)
        props.start_frame = seq['first'] + props.frame_start - 1
        return
    clip, info = None, {'error': "Blender cannot read this video"}
    try:
        clip = bpy.data.movieclips.load(str(path), check_existing=False)
        width, height = clip.size
        if width > 0 and height > 0 and clip.frame_duration > 0:
            info = {'width': int(width), 'height': int(height), 'frames': int(clip.frame_duration), 'fps': float(clip.fps)}
    except RuntimeError:
        pass
    finally:
        if clip is not None:
            bpy.data.movieclips.remove(clip)
    props.plate_info = json.dumps(info)


def _sensor_preset_changed(props, context):
    if props.sensor_preset != 'CUSTOM':
        props.sensor_mm = SENSOR_WIDTHS[props.sensor_preset]


def _sensor_changed(props, context):
    if props.sensor_preset != 'CUSTOM' and abs(SENSOR_WIDTHS[props.sensor_preset] - props.sensor_mm) > 1e-4:
        props.sensor_preset = 'CUSTOM'


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
    source: StringProperty(name="Plate", subtype='FILE_PATH', update=_probe_plate, description="A movie file, or any frame of an EXR/DPX/TIFF/PNG image sequence")
    plate_colorspace: EnumProperty(name="Plate colour space", items=[(key, key if key != 'RAW' else name, description) for key, name, description in plate_proxy.COLORSPACES], default='ACEScg', description="Colour space of the image-sequence pixels; the upload proxy is tone-mapped to AgX sRGB for the solver")
    plate_info: StringProperty(options={'HIDDEN'})
    frame_start: IntProperty(name="First source frame", default=1, min=1)
    frame_end: IntProperty(name="Last source frame", default=0, min=0, description="0 processes to the end of the video; otherwise inclusive")
    max_dimension: IntProperty(name="Processing long edge", default=1280, min=0, description="Pixels; 0 keeps original resolution. Imported camera matches the original plate size")
    review_witnesses: BoolProperty(name="Review witnesses", default=False, description="Run CoTracker and place the original review crosses; assesses the frozen camera without improving its solve")
    dense_environment: BoolProperty(name="Dense environment reconstruction", default=False, description="Compute post-SLAM video depth and accumulate static colored surfaces; adds processing time")
    dense_mode: EnumProperty(name="Dense depth", items=[('fast', "Sampled views (fast)", "Depth for the sampled dense views only"), ('video', "Full video depth (slow, original)", "Video depth for the whole selection, as in earlier versions")], default='fast')
    splat_preset: EnumProperty(name="Splat quality", items=[
        ('quick', "Quick preview", "About 5 minutes on a 16 GB GPU: 10k steps, 1536 px from the uploaded solve plate, up to 1M Gaussians"),
        ('quality', "Quality", "About 30-35 minutes for 300 frames: full 30k-step schedule, 2048 px (EXR plates upload a 2048 px training plate), up to 3M Gaussians")], default='quick')
    splat_custom: BoolProperty(name="Custom settings", default=False, description="Override the preset's training settings")
    splat_iterations: IntProperty(name="Training steps", default=30000, min=100, max=60000, description="The whole training schedule is scaled to this length")
    splat_max_dimension: IntProperty(name="Training long edge", default=2048, min=256, max=4096, description="Pixels; larger images need more GPU memory and, for EXR plates, a larger upload")
    splat_frame_step: IntProperty(name="Use every Nth frame", default=1, min=1, max=20, description="Shots longer than 400 frames are also subsampled evenly")
    splat_max_gaussians: IntProperty(name="Gaussian limit", default=3000000, min=50000, max=6000000, description="Detail versus GPU memory, file size (about 250 MB per million) and viewport speed")
    masking: BoolProperty(name="Exclude dynamic categories", default=True)
    mask_phrases: StringProperty(name="Categories", default="person,animal,vehicle,ball,balloon,gun,pet,car,bus")
    lens_mode: EnumProperty(name="Lens", items=[('auto', "Estimate and refine", "GeoCalib initialization followed by stock ViPE lens optimization"), ('fixed', "Known focal length (locked)", "Use a supplied focal length with a centered principal point and keep it fixed")], default='auto')
    lens_distortion: BoolProperty(name="Estimate lens distortion (experimental)", default=False, description="Solve a one-parameter unified (MEI) distortion and export undistort/redistort data; automatic lens mode only")
    focal_mm: FloatProperty(name="Focal length (mm)", default=35.0, min=0.1)
    sensor_preset: EnumProperty(name="Filmback", items=[(key, name, description) for key, name, description, _ in SENSOR_PRESETS], default='FULL_FRAME', update=_sensor_preset_changed)
    sensor_mm: FloatProperty(name="Sensor width (mm)", default=36.0, min=0.1, update=_sensor_changed, description="Filmback used for the imported camera and for known focal conversion; does not change the solve")
    max_points: IntProperty(name="Cloud point limit", default=30000, min=1, max=1000000, description="Native points exported and displayed. Native points are reconstruction hypotheses, not surveyed points")
    marker_count: IntProperty(name="3D scene markers", default=100, min=0, max=2000)
    start_frame: IntProperty(name="Blender start frame", default=1001, min=0, description="Timeline frame for the first selected source frame")
    scene_scale: FloatProperty(name="Scene scale", default=1.0, min=0.000001)
    auto_import: BoolProperty(name="Import when ready", default=True)
    job_id: StringProperty(name="Job ID")
    job_record: StringProperty(options={'HIDDEN'})
    result_path: StringProperty(name="Result bundle", subtype='FILE_PATH')
    status: StringProperty(default="Choose a plate and submit a tracking job.")
    state: StringProperty(default="IDLE")
    progress: FloatProperty(default=0, min=0, max=1, options={'SKIP_SAVE'})
    busy: BoolProperty(default=False, options={'SKIP_SAVE'})
    connection_state: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    connection_url: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    echo_job_logs: BoolProperty(name="Echo logs to terminal", default=True, description="Print live worker output in the terminal used to launch Blender")
    follow_job_log: BoolProperty(name="Follow live output", default=True, update=_follow_log_changed, description="Keep the log editor at the latest output; disable to pause the editor for inspection while the job continues")
    log_text_name: StringProperty(options={'HIDDEN'})
    lens_summary: StringProperty(options={'HIDDEN'})


class ViPEJobItem(bpy.types.PropertyGroup):
    date: StringProperty()
    frames: StringProperty()
    has_result: BoolProperty()
    job_id: StringProperty()
    record: StringProperty()


def preferences(context):
    return context.preferences.addons[__package__].preferences


def settings_from_properties(props):
    settings = {name: getattr(props, name) for name in ['frame_start', 'frame_end', 'max_dimension', 'masking', 'mask_phrases', 'review_witnesses', 'dense_environment', 'lens_mode', 'focal_mm', 'sensor_mm', 'max_points']}
    settings['lens_distortion'] = props.lens_distortion and props.lens_mode == 'auto'
    if props.dense_environment:
        settings['dense_mode'] = props.dense_mode
    return settings


def _loads(text):
    try:
        value = json.loads(text) if text else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _cache_dir(prefs, create=True):
    if prefs.cache_directory:
        return Path(bpy.path.abspath(prefs.cache_directory))
    return Path(bpy.utils.user_resource('DATAFILES', path='vipe_remote', create=create))


def _save_record(record):
    if record.get('folder'):
        try:
            (Path(record['folder']) / 'job.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        except OSError:
            pass  # The record in the scene stays usable; only the local history copy is stale.


def _parent_record(record, current=None):
    """The tracking job a review job belongs to; a tracking record is its own parent."""
    if record.get('mode') != 'review' and not record.get('parent_job_id'):
        return record
    parent_id = record['parent_job_id']
    folder = record.get('parent_folder') or (str(Path(record['folder']).parent / parent_id) if record.get('folder') else '')
    parent = {}
    if folder:
        try:
            parent = _loads(Path(folder, 'job.json').read_text(encoding='utf-8'))
        except OSError:
            pass
    if parent.get('job_id') != parent_id:
        parent = current if current and current.get('job_id') == parent_id else {key: record[key] for key in ('source', 'source_sha256', 'start_frame', 'marker_count', 'scene_scale', 'sensor_width', 'upload', 'background', 'plate') if key in record}
        parent = {**parent, 'job_id': parent_id}
    return parent


def read_job_history(folder, limit=40, scan_limit=2000):
    """Most recent `<folder>/*/job.json` records; immediate children only and bounded."""
    found = []
    try:
        with os.scandir(folder) as entries:
            for count, entry in enumerate(entries):
                if count >= scan_limit:
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stat = os.stat(os.path.join(entry.path, 'job.json'))
                        if stat.st_size <= 1024 * 1024:
                            found.append((stat.st_mtime, entry.path))
                except OSError:
                    pass
    except OSError:
        return []
    jobs = []
    for mtime, path in sorted(found, reverse=True)[:limit]:
        try:
            record = _loads(Path(path, 'job.json').read_text(encoding='utf-8'))
        except OSError:
            continue
        if not record.get('job_id'):
            continue
        record['folder'] = path
        result = Path(path, 'result.zip')
        if result.is_file():
            record['result'] = str(result)
        else:
            record.pop('result', None)
        jobs.append((mtime, record))
    return jobs


def _notify_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _job_log_text(props):
    name = 'ViPE Log ' + (props.job_id or 'connection')
    text = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    props.log_text_name = text.name
    return text


def _display_log(props, contents):
    text = _job_log_text(props)
    text.clear()
    text.write(contents)
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'TEXT_EDITOR' and area.spaces.active.text == text:
                region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                if region is not None:
                    try:
                        with bpy.context.temp_override(window=window, area=area, region=region):
                            bpy.ops.text.move(type='FILE_BOTTOM')
                    except RuntimeError:
                        pass  # A temporarily unavailable editor must not stop job monitoring.
                area.tag_redraw()


def _import(props, record):
    manifest, _, _ = load_bundle(props.result_path, record.get('source_sha256'))
    if manifest['job_id'] != record['job_id']:
        raise ValueError("The downloaded bundle belongs to a different job.")
    if manifest.get('splat'):
        from .splat import attach_splat
        scene_id = record.get('scene_job_id') or manifest.get('review_of')
        collection = next((c for c in bpy.context.scene.collection.children if c.get('vipe_job_id') == scene_id), None)
        if collection is not None:
            attach_splat(collection, props.result_path, manifest)
            parent = dict(record.get('scene_record') or _parent_record(record))
            parent.update(splat_job_id=record['job_id'], splat_result=props.result_path)
            _save_record(parent)
            props.job_id, props.job_record = scene_id, json.dumps(parent)
            props.result_path = parent.get('result', props.result_path)
            props.state, props.progress = 'COMPLETE', 1
            props.status = 'Gaussian splat ready. Open its previews or output folder in Results.'
            return collection
    background = record.get('background') or record.get('source', '')
    if background and not Path(background).is_file() and record.get('plate'):
        background = ''  # proxy frames were removed: geometry-only import
    collection, created = import_bundle(props.result_path, source_path=background, start_frame=record['start_frame'], marker_count=record['marker_count'], scene_scale=record['scene_scale'], expected_source_sha256=record.get('source_sha256'), sensor_width=record.get('sensor_width', 36.0), plate_path=plate_pattern(record) if record.get('plate') else '')
    props.state, props.progress = 'COMPLETE', 1
    parent_id = record.get('parent_job_id') or manifest.get('review_of')
    if parent_id and not created and collection.get('vipe_job_id') == parent_id:
        # Results stay on the parent job, which now carries the review witnesses.
        parent = _parent_record({**record, 'parent_job_id': parent_id}, _loads(props.job_record))
        parent['review_job_id'] = record['job_id']
        _save_record(parent)
        props.job_id, props.job_record, props.result_path = parent_id, json.dumps(parent), parent.get('result', '')
        added = [name for key, name in (('review_witnesses', 'review witnesses'), ('dense_environment', 'dense environment')) if record.get('extend', {'review_witnesses': True}).get(key)]
        props.status = f"{' and '.join(added).capitalize()} added to job {parent_id[:8]}"
        if record.get('extend', {}).get('dense_environment'):
            props.status += '. Use Show Dense Environment.'
        return collection
    props.status = f"Imported {manifest['selection']['count']} camera frames and {manifest['cloud']['point_count']:,} native points." if created else "This job is already imported; its collection was preserved."
    if created:
        crosses = next((c for c in collection.children if c.get('vipe_role') == 'review_crosses_group'), None)
        if crosses is not None and manifest.get('witnesses'):
            props.status += f" Witnesses: {crosses.get('witness_count', 0)}/4. {crosses.get('status', '')}"
    if created and manifest.get('environment'):
        props.status += ' Dense reconstruction ready: use Show Dense Environment.'
    stages = manifest.get('processing', {}).get('lens_stages_original_pixels', {})
    props.lens_summary = ' / '.join(f"{name}: {values[0]:.1f} px" for name, values in [('initial fx', stages.get('initial')), ('final fx', stages.get('post_global_ba'))] if values)
    if bpy.context.screen:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
    return collection


def _run_network(mode, record, url, token, events, cancel, detach, log_buffer):
    def emit(kind, value):
        events.put((kind, value))
    def log_output(value):
        clean = log_buffer.append(value)
        if clean and log_buffer.echo:
            try:
                print(clean, end='', flush=True)
            except OSError:
                log_buffer.echo = False
    last_progress = [None, 0.0]
    def progress(message, fraction):
        # Live details after ' · ' change every second; log only stage changes.
        now, base = time.monotonic(), message.split(' · ', 1)[0]
        if last_progress[0] != base:
            log_output(f"\n[ViPE {record.get('job_id', '')[:8]}] {base}\n")
        if last_progress[0] != base or now - last_progress[1] > 0.2:
            emit('progress', (message, fraction))
            last_progress[:] = [base, now]
    try:
        client = Client(url, token)
        if mode == 'check':
            client.capabilities()
            emit('connection', True)
            emit('connected', "Connected to the ViPE worker.")
            return
        if mode == 'log':
            emit('log', client.log(record['job_id']))
            return
        folder = Path(record['folder'])
        folder.mkdir(parents=True, exist_ok=True)
        if mode == 'submit':
            upload_path = prepare_upload(record, progress, lambda: cancel.is_set() or detach.is_set())
            if upload_path != record['source']:
                emit('record', dict(record))
            uploaded = client.upload(upload_path, progress, lambda: cancel.is_set() or detach.is_set())
            emit('connection', True)
            if uploaded.get('reused'):
                progress('Plate already on worker', 1)
            if cancel.is_set():
                raise JobCancelled("Upload stopped before submission.")
            if detach.is_set():
                raise Detached("Upload finished; no job submitted after monitoring stopped.")
            record['source_sha256'] = uploaded['sha256']
            (folder / 'job.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            emit('record', dict(record))
            workflow = client.submit(uploaded, record['plate']['pattern'] if record.get('plate') else Path(record['source']).name, record['settings'], record['job_id'])
            (folder / 'workflow_api.json').write_text(json.dumps(workflow, indent=2), encoding='utf-8')
        elif mode in ('review', 'splat'):
            (folder / 'job.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            emit('record', dict(record))
            extend = dict(record.get('extend', {}))
            if mode == 'splat' and record.get('training_long_edge'):
                stop = lambda: cancel.is_set() or detach.is_set()
                training = prepare_training_plate(record, record['training_long_edge'], progress, stop)
                if training:
                    uploaded = client.upload(training, lambda message, fraction: progress('Uploading training plate' if message == 'Uploading video' else message, fraction), stop)
                    extend.update(splat_training_video=uploaded['video'], splat_training_sha256=uploaded['sha256'])
                if cancel.is_set():
                    raise JobCancelled("Upload stopped before submission.")
            submit = client.submit_splat if mode == 'splat' else client.submit_review
            workflow = submit(record['parent_job_id'], record['job_id'], **extend)
            emit('connection', True)
            (folder / 'workflow_api.json').write_text(json.dumps(workflow, indent=2), encoding='utf-8')
        item = client.wait(record['job_id'], progress, cancel.is_set, detach.is_set, log=log_output)
        result = client.download(item, folder / 'result.zip', progress, lambda: detach.is_set() or cancel.is_set())
        manifest, _, _ = load_bundle(result, record.get('source_sha256'))
        if manifest['job_id'] != record['job_id']:
            raise ValueError("Downloaded result belongs to another job.")
        record['source_sha256'] = manifest['source']['sha256']
        record['result'] = str(result)
        (folder / 'job.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        if record['source']:
            h = hashlib.sha256()
            with Path(record.get('upload') or record['source']).open('rb') as plate:
                for chunk in iter(lambda: plate.read(1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != record['source_sha256']:
                raise ValueError('The local plate changed after submission. Restore that source version before importing.')
        emit('ready', record)
    except JobCancelled as exc:
        log_output(f'\n[ViPE] {exc}\n')
        emit('cancelled', str(exc))
    except Detached as exc:
        emit('detached', str(exc))
    except Exception as exc:
        log_output(f'\n[ViPE error] {exc}\n')
        if mode == 'check' or isinstance(exc, (OSError, http.client.HTTPException)):
            emit('connection', False)
        emit('error', str(exc))
    finally:
        emit('finished', None)


def _stage(message):
    base = message.split(' · ', 1)[0]
    return next((STAGE_ORDER.index(key) for prefix, key in STAGE_MESSAGES if base.startswith(prefix)), None)


def _job_stages(record):
    if record.get('mode') == 'splat':
        return {'upload', 'queue', 'prepare', 'splat', 'package', 'download', 'import'} if record.get('training_long_edge') else {'queue', 'prepare', 'splat', 'package', 'download', 'import'}
    if record.get('mode') == 'review':
        return {'queue', 'prepare', 'witnesses', 'package', 'download', 'import'}
    settings = record.get('settings') or {}
    keys = {'upload', 'queue', 'prepare', 'solve', 'package', 'download'}
    keys |= {'witnesses'} if settings.get('review_witnesses') else set()
    keys |= {'dense'} if settings.get('dense_environment') else set()
    return keys | ({'import'} if record.get('auto_import') else set())


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
    active['log_buffer'].echo = props.echo_job_logs
    finished = False
    for _ in range(100):
        try:
            kind, value = active['events'].get_nowait()
        except queue.Empty:
            break
        if kind == 'progress':
            props.status, props.progress = value
            props.state = 'UPLOADING' if value[0].startswith(('Uploading', 'Checking plate')) else 'DOWNLOADING' if value[0].startswith('Downloading') else 'RUNNING'
            stage = _stage(value[0])
            if stage is not None:
                active['stage'] = max(active.get('stage', -1), stage)
        elif kind == 'record':
            props.job_record = json.dumps(value)
        elif kind == 'ready':
            props.job_record = json.dumps(value)
            props.result_path = value['result']
            props.state, props.status, props.progress = 'READY', 'Result downloaded. Ready to import.', 1
            active['stage'] = STAGE_ORDER.index('import')
            if value['auto_import'] and active['scene'] == bpy.context.scene:
                try:
                    _import(props, value)
                except Exception as exc:
                    props.state, props.status = 'ERROR', f"Import failed: {exc}"
        elif kind in ('cancelled', 'error', 'detached'):
            props.state = 'CANCELLED' if kind == 'cancelled' else 'ERROR' if kind == 'error' else 'READY'
            props.status = value
        elif kind == 'connection':
            props.connection_state, props.connection_url = 'OK' if value else 'ERROR', active.get('url', '')
        elif kind == 'connected':
            props.status = value
        elif kind == 'log':
            _display_log(props, value)
            props.status = 'Log loaded. Use Open Live Log to view it.'
        elif kind == 'finished':
            finished = True
    revision, contents = active['log_buffer'].snapshot()
    if active['mode'] in JOB_MODES and props.follow_job_log and revision != active['log_revision']:
        _display_log(props, contents)
        active['log_revision'] = revision
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
    log_buffer = LogBuffer()
    log_buffer.echo = context.scene.vipe_remote.echo_job_logs
    if mode in JOB_MODES:
        _log_buffers.pop(record['job_id'], None)
        _log_buffers[record['job_id']] = log_buffer
        while len(_log_buffers) > 4:
            del _log_buffers[next(iter(_log_buffers))]
        log_buffer.append(f"ViPE {'review ' if record.get('mode') == 'review' else ''}job {record['job_id']}\nFollowing worker output…\n")
        _display_log(context.scene.vipe_remote, log_buffer.snapshot()[1])
    thread = threading.Thread(target=_run_network, args=(mode, record, url, prefs.token, events, cancel, detach, log_buffer), daemon=True, name='ViPE-HTTP')
    _active = {'thread': thread, 'events': events, 'cancel': cancel, 'detach': detach, 'scene': context.scene, 'mode': mode, 'log_buffer': log_buffer, 'log_revision': 0, 'record': record, 'url': url, 'started': time.monotonic(), 'stage': 0 if mode == 'submit' else -1}
    context.scene.vipe_remote.busy = True
    thread.start()
    bpy.app.timers.register(_poll, first_interval=0.1)


def _has_remote_job(props):
    record = _loads(props.job_record)
    return bool(record.get('server') and record.get('folder'))


def _summary(collection):
    summary = _loads(collection.get('vipe_summary', ''))
    if 'has_witnesses' not in summary:
        summary['has_witnesses'] = any(c.get('vipe_role') == 'review_crosses_group' and len(c.all_objects) for c in collection.children)
    if 'dense' not in summary:
        summary['dense'] = any(c.get('vipe_role') == 'dense_environment_group' for c in collection.children)
    return summary


class VIPE_OT_check(bpy.types.Operator):
    bl_idname = 'vipe_remote.check'
    bl_label = 'Check Connection'
    bl_description = 'Ask the worker for its ViPE capabilities'
    @classmethod
    def poll(cls, context):
        return _active is None
    def execute(self, context):
        _launch(context, 'check', {})
        return {'FINISHED'}


class VIPE_OT_submit(bpy.types.Operator):
    bl_idname = 'vipe_remote.submit'
    bl_label = 'Submit Tracking Job'
    bl_description = 'Upload this plate to ComfyUI, solve on the worker, and retrieve camera and native points (reconstruction hypotheses, not surveyed points)'
    @classmethod
    def poll(cls, context):
        return _active is None and bool(context.scene.vipe_remote.source)
    def execute(self, context):
        props, prefs = context.scene.vipe_remote, preferences(context)
        source = Path(bpy.path.abspath(props.source))
        if not source.is_file():
            self.report({'ERROR'}, "Choose an existing video plate.")
            return {'CANCELLED'}
        settings = settings_from_properties(props)
        if settings['lens_distortion'] and settings['dense_environment']:
            self.report({'ERROR'}, "Dense environment does not support lens distortion yet; disable one of them.")
            return {'CANCELLED'}
        identifier = str(uuid.uuid4())
        cache = _cache_dir(prefs)
        record = {'job_id': identifier, 'source': str(source.resolve()), 'server': prefs.worker_url, 'folder': str(cache / identifier), 'settings': settings, 'start_frame': props.start_frame, 'marker_count': props.marker_count, 'scene_scale': props.scene_scale, 'sensor_width': props.sensor_mm, 'auto_import': props.auto_import}
        if plate_proxy.is_sequence(source):
            info = _loads(props.plate_info)
            scene = context.scene
            record.update(plate_colorspace=props.plate_colorspace, ocio_config=os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles', 'colormanagement', 'config.ocio'),
                          plate_fps=info.get('fps_rational') or [round(scene.render.fps * 1000), round(scene.render.fps_base * 1000)])
        props.job_id, props.job_record = identifier, json.dumps(record)
        props.result_path, props.state, props.status, props.progress = '', 'UPLOADING', 'Connecting to worker', 0
        props.lens_summary = ''
        _launch(context, 'submit', record)
        return {'FINISHED'}


def _extension_poll(cls, context, key, message):
    props = context.scene.vipe_remote
    if _active is not None:
        return False
    if not _has_remote_job(props):
        cls.poll_message_set('Needs the worker job record; a saved bundle cannot be extended.')
        return False
    collection = job_collection(context)
    if collection is None or _summary(collection).get(key):
        cls.poll_message_set(message)
        return False
    return True


def _launch_extension(operator, context, extend, status):
    """Run optional stages on the worker for the current job, reusing its camera solve."""
    props, prefs = context.scene.vipe_remote, preferences(context)
    parent = _parent_record(_loads(props.job_record))
    if not (parent.get('server') and parent.get('folder')):
        operator.report({'ERROR'}, "The original job record is missing; load the job from Jobs first.")
        return {'CANCELLED'}
    identifier = str(uuid.uuid4())
    record = {'job_id': identifier, 'parent_job_id': parent['job_id'], 'mode': 'review', 'extend': extend, 'server': parent['server'], 'folder': str(_cache_dir(prefs) / identifier), 'parent_folder': parent['folder'], 'source': parent.get('source', ''), 'source_sha256': parent.get('source_sha256'), 'start_frame': parent.get('start_frame', props.start_frame), 'marker_count': parent.get('marker_count', props.marker_count), 'scene_scale': parent.get('scene_scale', props.scene_scale), 'sensor_width': parent.get('sensor_width', props.sensor_mm), 'auto_import': True, 'settings': parent.get('settings', {}), **{key: parent[key] for key in ('upload', 'background', 'plate') if key in parent}}
    parent['review_job_id'] = identifier
    _save_record(parent)
    props.job_id, props.job_record = identifier, json.dumps(record)
    props.result_path, props.state, props.status, props.progress = '', 'RUNNING', status, 0
    _launch(context, 'review', record)
    return {'FINISHED'}


class VIPE_OT_review(bpy.types.Operator):
    bl_idname = 'vipe_remote.review'
    bl_label = 'Run Review Witnesses'
    bl_description = 'Add independent CoTracker review witnesses to this job on the worker, reusing its solved camera; the solve is not changed'
    @classmethod
    def poll(cls, context):
        return _extension_poll(cls, context, 'has_witnesses', 'This job already has review witnesses.')
    def execute(self, context):
        return _launch_extension(self, context, {'review_witnesses': True}, 'Submitting review job')


class VIPE_OT_add_dense(bpy.types.Operator):
    bl_idname = 'vipe_remote.add_dense'
    bl_label = 'Add Dense Environment'
    bl_description = 'Compute dense geometry for this job on the worker, reusing its solved camera, and add it to this scene'
    @classmethod
    def poll(cls, context):
        if not _extension_poll(cls, context, 'dense', 'This job already has a dense environment.'):
            return False
        if _summary(job_collection(context)).get('distortion'):
            cls.poll_message_set('Dense environment does not support lens distortion yet.')
            return False
        return True
    def execute(self, context):
        return _launch_extension(self, context, {'review_witnesses': False, 'dense_environment': True, 'dense_mode': context.scene.vipe_remote.dense_mode}, 'Submitting dense job')


SPLAT_KEYS = ('splat_iterations', 'splat_max_dimension', 'splat_frame_step', 'splat_max_gaussians')


def _splat_settings(props):
    """The preset, plus explicit values only when the artist overrides it (the worker owns the preset values)."""
    return {'splat_preset': props.splat_preset, **({key: getattr(props, key) for key in SPLAT_KEYS} if props.splat_custom else {})}


def _training_long_edge(props):
    """Quality trains above the solve proxy; image-sequence plates then upload a larger proxy of the same frames."""
    if props.splat_preset != 'quality':
        return 0
    return props.splat_max_dimension if props.splat_custom else 2048


class VIPE_OT_train_splat(bpy.types.Operator):
    bl_idname = 'vipe_remote.train_splat'
    bl_label = 'Train Gaussian Environment'
    bl_description = 'Train a static Gaussian splat on the worker using this job’s solved cameras, dense points and motion masks'

    @classmethod
    def poll(cls, context):
        props = context.scene.vipe_remote
        collection = job_collection(context)
        if _active is not None or not _has_remote_job(props):
            cls.poll_message_set('Load the completed worker job first.')
            return False
        if collection is None or not _summary(collection).get('dense'):
            cls.poll_message_set('Add Dense Environment first.')
            return False
        return not _summary(collection).get('distortion')

    def execute(self, context):
        props, prefs = context.scene.vipe_remote, preferences(context)
        collection = job_collection(context)
        parent = _parent_record(_loads(props.job_record))
        dense_id = collection.get('vipe_dense_job_id')
        if not dense_id:
            # Scenes saved before the splat feature keep the latest extension ID here.
            dense_id = parent.get('review_job_id') or parent['job_id']
        identifier = str(uuid.uuid4())
        record = {**parent, 'job_id': identifier, 'parent_job_id': dense_id,
                  'scene_job_id': collection['vipe_job_id'], 'scene_record': dict(parent),
                  'parent_folder': parent.get('folder', ''), 'mode': 'splat',
                  'folder': str(_cache_dir(prefs) / identifier), 'auto_import': True,
                  'extend': _splat_settings(props), 'training_long_edge': _training_long_edge(props)}
        record.pop('result', None)
        props.job_id, props.job_record = identifier, json.dumps(record)
        props.result_path, props.state, props.status, props.progress = '', 'RUNNING', 'Submitting splat job', 0
        _launch(context, 'splat', record)
        return {'FINISHED'}


class VIPE_OT_view_splat(bpy.types.Operator):
    bl_idname = 'vipe_remote.view_splat'
    bl_label = 'Show Gaussian Splat'
    bl_description = 'Load the trained splat under this job\'s ViPE alignment (KIRI 3DGS Render on Blender 5.1+, native splats on Blender 5.3+), or toggle it'
    show: BoolProperty(default=True)

    @classmethod
    def poll(cls, context):
        collection = job_collection(context)
        return collection is not None and bool(collection.get('vipe_splat_ply'))

    def execute(self, context):
        from .splat import import_correction, load_splat
        collection = job_collection(context)
        objects = [bpy.data.objects.get(name) for name in json.loads(collection.get('vipe_splat_objects', '[]'))]
        objects = [o for o in objects if o is not None]
        if objects:
            for obj in objects:
                obj.hide_viewport = obj.hide_render = not self.show
            return {'FINISHED'}
        if not self.show:
            return {'FINISHED'}
        path = Path(collection['vipe_splat_ply'])
        if not path.is_file():
            self.report({'ERROR'}, f'The downloaded splat is missing: {path}')
            return {'CANCELLED'}
        root = next(o for o in collection.all_objects if o.get('vipe_role') == 'alignment')
        try:
            objects = load_splat(context, path)
            correction = import_correction(objects[0], path)
        except ValueError as exc:
            for obj in locals().get('objects') or []:
                bpy.data.objects.remove(obj, do_unlink=True)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        group = next((c for c in collection.children if c.get('vipe_role') == 'splat_group'), None) or _child_collection(collection, '06 Gaussian splat', 'splat_group')
        for obj in objects:
            for owner in list(obj.users_collection):
                owner.objects.unlink(obj)
            group.objects.link(obj)
            # The PLY is in the ViPE world, like the dense samples.
            obj.parent, obj.matrix_parent_inverse, obj.matrix_basis = root, Matrix.Identity(4), correction
            obj['vipe_role'] = 'gaussian_splat'
        objects[0].name = f"Gaussian splat {collection['vipe_job_id'][:8]}"
        collection['vipe_splat_objects'] = json.dumps([o.name for o in objects])
        self.report({'INFO'}, f'Loaded {objects[0].name}')
        return {'FINISHED'}


class VIPE_OT_open_splat(bpy.types.Operator):
    bl_idname = 'vipe_remote.open_splat'
    bl_label = 'Open Splat Output'
    kind: EnumProperty(items=[('camera', 'Camera preview', ''), ('nearby', 'Nearby view', ''), ('masks', 'Training masks', ''), ('folder', 'Output folder', '')])

    def execute(self, context):
        collection = job_collection(context)
        folder = Path(collection.get('vipe_splat_directory', '')) if collection else None
        names = {'camera': 'camera_preview.mp4', 'nearby': 'nearby_preview.mp4', 'masks': 'masks_preview.jpg'}
        path = folder / names[self.kind] if folder and self.kind in names else folder
        if path is None or not path.exists():
            self.report({'ERROR'}, 'The downloaded splat output is missing.')
            return {'CANCELLED'}
        return bpy.ops.wm.path_open(filepath=str(path))


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
            record = {'job_id': manifest['job_id'], 'source': source, 'start_frame': props.start_frame, 'marker_count': props.marker_count, 'scene_scale': props.scene_scale, 'sensor_width': props.sensor_mm, 'source_sha256': manifest['source']['sha256']}
            if manifest.get('review_of'):
                record.update(mode='review', parent_job_id=manifest['review_of'])
            props.result_path = self.filepath
            collection = _import(props, record)
            if collection.get('vipe_job_id') == record['job_id']:
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


class VIPE_OT_open_log(bpy.types.Operator):
    bl_idname = 'vipe_remote.open_log'
    bl_label = 'Open Live Log'
    bl_description = 'Show this job’s live output in a Text Editor beside the viewport'

    @classmethod
    def poll(cls, context):
        return bool(context.scene.vipe_remote.job_id) and context.area is not None

    def execute(self, context):
        text = _job_log_text(context.scene.vipe_remote)
        target = next((a for a in context.screen.areas if a.type == 'TEXT_EDITOR' and a.spaces.active.text == text), None)
        if target is None:
            before = {a.as_pointer() for a in context.screen.areas}
            bpy.ops.screen.area_split(direction='HORIZONTAL', factor=0.35)
            target = next((a for a in context.screen.areas if a.as_pointer() not in before), None)
            if target is None:
                self.report({'WARNING'}, 'Select the job log in any Blender Text Editor.')
                return {'CANCELLED'}
            target.type = 'TEXT_EDITOR'
        target.spaces.active.text = text
        target.spaces.active.show_word_wrap = True
        target.spaces.active.show_line_numbers = False
        region = next((r for r in target.regions if r.type == 'WINDOW'), None)
        if region is not None:
            with context.temp_override(area=target, region=region):
                bpy.ops.text.move(type='FILE_BOTTOM')
        return {'FINISHED'}


def _dense_layer(context):
    parent = job_collection(context)
    for child in parent.children if parent else []:
        if child.get('vipe_role') == 'dense_environment_group':
            return context.view_layer.layer_collection.children[parent.name].children[child.name], child
    return None, None


class VIPE_OT_view_environment(bpy.types.Operator):
    bl_idname = 'vipe_remote.view_environment'
    bl_label = 'Show Dense Environment'
    bl_description = 'Show or hide the selected job’s dense samples; the camera solve stays unchanged'
    show: BoolProperty(default=True, options={'SKIP_SAVE'})
    def execute(self, context):
        layer, child = _dense_layer(context)
        if layer is None:
            self.report({'INFO'}, 'This job has no dense environment. Use Add Dense Environment.')
            return {'CANCELLED'}
        layer.hide_viewport = not self.show
        space = context.space_data if context.space_data and context.space_data.type == 'VIEW_3D' else None
        if self.show:
            for obj in context.selected_objects: obj.select_set(False)
            for obj in child.all_objects:
                if obj.get('vipe_role') == 'dense_samples': obj.select_set(True)
            if space:
                space.shading.type = 'MATERIAL'
                bpy.ops.view3d.view_selected(use_all_regions=False)
        elif space:
            # Back to the lightweight camera review.
            space.shading.type = 'SOLID'
            space.region_3d.view_perspective = 'CAMERA'
        return {'FINISHED'}


class VIPE_OT_markers_in_render(bpy.types.Operator):
    bl_idname = 'vipe_remote.markers_in_render'
    bl_label = 'Markers in Renders'
    bl_description = 'Show or hide the static-track markers and review crosses in renders (the plate is always composited underneath)'
    bl_options = {'REGISTER', 'UNDO'}
    enable: BoolProperty(default=True, options={'SKIP_SAVE'})
    @classmethod
    def poll(cls, context):
        return job_collection(context) is not None
    def execute(self, context):
        from .review_render import set_markers_in_render
        set_markers_in_render(job_collection(context), self.enable)
        return {'FINISHED'}


class VIPE_OT_export_nuke(bpy.types.Operator):
    bl_idname = 'vipe_remote.export_nuke'
    bl_label = 'Export to Nuke'
    bl_description = 'Write a Nuke script with this job’s camera, points and lens data to a folder'
    directory: StringProperty(subtype='DIR_PATH')
    filter_folder: BoolProperty(default=True, options={'HIDDEN'})
    @classmethod
    def poll(cls, context):
        return _active is None and job_collection(context) is not None
    def invoke(self, context, event):
        if not self.directory:
            folder = _parent_record(_loads(context.scene.vipe_remote.job_record)).get('folder')
            if folder and Path(folder).is_dir():
                self.directory = str(Path(folder)) + os.sep
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    def execute(self, context):
        if not self.directory:
            self.report({'ERROR'}, "Choose an export folder.")
            return {'CANCELLED'}
        try:
            from . import nuke_export
            result = nuke_export.export_nuke(context, job_collection(context), bpy.path.abspath(self.directory))
        except (ValueError, ImportError, OSError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        for warning in result.get('warnings', []):
            self.report({'WARNING'}, warning)
        context.scene.vipe_remote.status = f"Exported {Path(result['nk']).name} with {len(result.get('files', []))} files to {Path(result['nk']).parent}"
        self.report({'INFO'}, context.scene.vipe_remote.status)
        return {'FINISHED'}


class VIPE_OT_full_clip(bpy.types.Operator):
    bl_idname = 'vipe_remote.full_clip'
    bl_label = 'Full Clip'
    bl_description = 'Process the whole video: first frame 1, last frame 0 (end)'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        props = context.scene.vipe_remote
        props.frame_start, props.frame_end = 1, 0
        return {'FINISHED'}


def _load_record(props, record):
    props.job_id, props.job_record = record['job_id'], json.dumps(record)
    props.result_path = record.get('result', '')
    props.state, props.progress, props.lens_summary = 'READY' if props.result_path else 'IDLE', 0, ''
    props.status = f"Loaded job {record['job_id'][:8]}. " + ("Import or resume it." if props.result_path else "Resume it to reconnect.")


def _selected_job(context):
    wm = context.window_manager
    return wm.vipe_jobs[wm.vipe_jobs_index] if 0 <= wm.vipe_jobs_index < len(wm.vipe_jobs) else None


class VIPE_OT_refresh_jobs(bpy.types.Operator):
    bl_idname = 'vipe_remote.refresh_jobs'
    bl_label = 'Refresh'
    bl_description = 'List the 40 most recent jobs in the download folder'
    def execute(self, context):
        wm = context.window_manager
        wm.vipe_jobs.clear()
        for mtime, record in read_job_history(_cache_dir(preferences(context), create=False)):
            item = wm.vipe_jobs.add()
            settings = record.get('settings') or {}
            if record.get('mode') == 'review':
                item.name = f"Review of {record.get('parent_job_id', '')[:8]}"
            else:
                item.name = Path(record.get('source') or '').name or 'Saved job'
                item.frames = f"{settings.get('frame_start', 1)}–{settings.get('frame_end') or 'end'}" if settings else ''
            item.date = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
            item.has_result, item.job_id, item.record = bool(record.get('result')), record['job_id'], json.dumps(record)
        wm.vipe_jobs_index = min(wm.vipe_jobs_index, max(len(wm.vipe_jobs) - 1, 0))
        self.report({'INFO'}, f"{len(wm.vipe_jobs)} saved jobs")
        return {'FINISHED'}


class VIPE_OT_load_job(bpy.types.Operator):
    bl_idname = 'vipe_remote.load_job'
    bl_label = 'Load'
    bl_description = 'Select this saved job so it can be resumed, imported or reviewed'
    @classmethod
    def poll(cls, context):
        return _active is None and _selected_job(context) is not None
    def execute(self, context):
        _load_record(context.scene.vipe_remote, json.loads(_selected_job(context).record))
        return {'FINISHED'}


class VIPE_OT_import_job(bpy.types.Operator):
    bl_idname = 'vipe_remote.import_job'
    bl_label = 'Import'
    bl_description = 'Import the downloaded result of this saved job'
    bl_options = {'REGISTER', 'UNDO'}
    @classmethod
    def poll(cls, context):
        item = _selected_job(context)
        return _active is None and item is not None and item.has_result
    def execute(self, context):
        props, record = context.scene.vipe_remote, json.loads(_selected_job(context).record)
        _load_record(props, record)
        try:
            _import(props, record)
        except Exception as exc:
            props.state, props.status = 'ERROR', f"Import failed: {exc}"
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class VIPE_UL_jobs(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=item.name, icon='CHECKMARK' if item.has_result else 'TIME')
        date = row.row()
        date.alignment, date.active = 'RIGHT', False
        date.label(text=item.date)


def _wrap_width(context):
    return max(20, int((context.region.width if context.region else 300) / 7) - 6)


def _hint(layout, text, width):
    column = layout.column(align=True)
    column.active = False
    for line in textwrap.wrap(text, width):
        column.label(text=line, icon='BLANK1')


def _number(value):
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _clock(seconds):
    hours, rest = divmod(int(round(seconds)), 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


def _summary_lines(summary):
    lines = []
    if isinstance(summary.get('frames'), int):
        lines.append(f"Frames: {summary['frames']:,}")
    focal, sensor, pixels = (_number(summary.get(key)) for key in ('focal_mm', 'sensor_width_mm', 'focal_px'))
    if focal:
        lines.append(f"Focal: {focal:.2f} mm @ {sensor:.2f} mm filmback" if sensor else f"Focal: {focal:.2f} mm")
    if pixels:
        lines.append(f"Focal: {pixels:.1f} px")
    initial, final = _number(summary.get('lens_initial_px')), _number(summary.get('lens_final_px'))
    if initial and final:
        lines.append(f"Lens refinement: {initial:.1f} → {final:.1f} px")
    distortion = summary.get('distortion')
    if isinstance(distortion, dict):
        xi = _number(distortion.get('xi'))
        lines.append(f"Distortion: {str(distortion.get('model', 'unified')).upper()}" + (f" ξ {xi:.4f}" if xi is not None else ''))
    if isinstance(summary.get('points'), int):
        lines.append(f"Native points: {summary['points']:,}")
    if isinstance(summary.get('witness_landmarks'), int):
        lines.append(f"Witness landmarks: {summary['witness_landmarks']:,}")
    median, p90 = _number(summary.get('reprojection_px_median')), _number(summary.get('reprojection_px_p90'))
    if median is not None:
        lines.append(f"Review error: median {median:.2f} px" + (f", p90 {p90:.2f} px" if p90 is not None else ''))
    timings = summary.get('timings')
    if isinstance(timings, dict):
        parts = [f"{name.replace('_', ' ')} {_clock(_number(value))}" for name, value in timings.items() if _number(value) is not None]
        if parts:
            lines.append('Time: ' + ', '.join(parts))
    return lines


def _plate_lines(props):
    info = _loads(props.plate_info)
    if 'error' in info:
        return [(info['error'], 'ERROR')]
    if not info.get('frames'):
        return []
    fps = info.get('fps') or 0
    rate = (f"{fps:.3f}".rstrip('0').rstrip('.') + ' fps') if fps > 0 else 'unknown frame rate'
    lines = [(f"{info['width']} × {info['height']}, {rate}", 'FILE_IMAGE' if info.get('sequence') else 'FILE_MOVIE'), (f"{info['frames']} frames" + (f", {info['frames'] / fps:.1f} s" if fps > 0 else ''), 'BLANK1')]
    if info.get('sequence'):
        lines.append((f"Sequence {info['pattern']} {info['first']}–{info['last']}", 'SEQUENCE'))
        if info.get('gaps'):
            lines.append((f"{info['gaps']} missing frames in the sequence", 'ERROR'))
        if abs(info.get('pixel_aspect', 1) - 1) > 1e-3:
            lines.append(("Non-square pixels are not supported yet", 'ERROR'))
        if not info.get('fps_rational'):
            lines.append(("No frame rate in the file; the scene frame rate is used", 'INFO'))
    return lines


def _range_text(props):
    info = _loads(props.plate_info)
    total, fps = info.get('frames') or 0, info.get('fps') or 0
    first, last = props.frame_start, props.frame_end or total
    if not last:
        return f"Frames {first}–end", 'PREVIEW_RANGE'
    if last < first:
        return "Last frame is before the first frame", 'ERROR'
    if total and last > total:
        return f"Range exceeds the clip ({total} frames)", 'ERROR'
    count = last - first + 1
    plate = f", plate {info['first'] + first - 1}–{info['first'] + last - 1}" if info.get('sequence') else ''
    return f"Frames {first}–{last} ({count} frames" + (f", {count / fps:.1f} s" if fps > 0 else "") + plate + ")", 'PREVIEW_RANGE'


def _draw_stages(layout, props, active, width):
    keys = _job_stages(active.get('record') or {})
    current = active.get('stage', -1)
    detail = props.status.split(' · ', 1)[1] if ' · ' in props.status else ''
    column = layout.column(align=True)
    for index, (key, label) in enumerate(STAGES):
        if key not in keys:
            continue
        row = column.row()
        row.active = index <= current
        row.label(text=label, icon='CHECKMARK' if index < current else 'PLAY' if index == current else 'RADIOBUT_OFF')
        if index == current and detail:
            _hint(column, detail, width - 4)


class VIPE_PT_panel(bpy.types.Panel):
    bl_label = 'ViPE Remote Tracking'
    bl_idname = 'VIPE_PT_remote_tracking'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'ViPE Remote'
    def draw(self, context):
        layout, props, prefs = self.layout, context.scene.vipe_remote, preferences(context)
        width = _wrap_width(context)
        state = props.connection_state if props.connection_url == prefs.worker_url else ''
        text, icon = {'OK': ("Worker connected", 'CHECKMARK'), 'ERROR': ("Worker check failed", 'ERROR')}.get(state, ("Worker not checked", 'QUESTION'))
        row = layout.row(align=True)
        row.label(text=text, icon=icon)
        row.operator('vipe_remote.check', text='', icon='FILE_REFRESH')
        row = layout.row()
        row.scale_y = 1.4
        if props.busy:
            if _active and _active['mode'] in JOB_MODES:
                row.operator('vipe_remote.cancel', icon='CANCEL', text='Stop Download' if props.state == 'DOWNLOADING' else 'Cancel Job')
            else:
                row.label(text='Contacting worker…')
        else:
            row.operator('vipe_remote.submit', icon='CAMERA_DATA')
        status = layout.box()
        for line in textwrap.wrap(props.status.split(' · ', 1)[0] if props.busy else props.status, width=width)[:5]:
            status.label(text=line)
        if props.lens_summary and props.state == 'COMPLETE':
            for line in textwrap.wrap(props.lens_summary, width=width):
                status.label(text=line)
        if props.busy:
            status.progress(factor=props.progress, type='BAR', text=f"{props.progress:.0%}")
            if _active:
                status.label(text='Elapsed ' + _clock(time.monotonic() - _active.get('started', time.monotonic())), icon='TIME')
                if _active['mode'] in JOB_MODES:
                    _draw_stages(status, props, _active, width)


class _SubPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'ViPE Remote'
    bl_parent_id = 'VIPE_PT_remote_tracking'


class VIPE_PT_worker(_SubPanel, bpy.types.Panel):
    bl_label = 'Worker'
    bl_options = {'DEFAULT_CLOSED'}
    def draw(self, context):
        layout, prefs = self.layout, preferences(context)
        layout.enabled = not context.scene.vipe_remote.busy
        layout.prop(prefs, 'worker_url', text='Worker')
        layout.prop(prefs, 'token')
        layout.prop(prefs, 'cache_directory')
        layout.operator('vipe_remote.check', icon='CHECKMARK')


class VIPE_PT_plate(_SubPanel, bpy.types.Panel):
    bl_label = 'Plate'
    def draw(self, context):
        layout, props = self.layout, context.scene.vipe_remote
        layout.enabled = not props.busy
        layout.prop(props, 'source')
        info = layout.column(align=True)
        for text, icon in _plate_lines(props):
            info.label(text=text, icon=icon)
        row = layout.row(align=True)
        row.prop(props, 'frame_start', text='First')
        row.prop(props, 'frame_end', text='Last')
        row.operator('vipe_remote.full_clip', text='Full Clip')
        text, icon = _range_text(props)
        layout.label(text=text, icon=icon)
        if _loads(props.plate_info).get('sequence'):
            layout.prop(props, 'plate_colorspace')
            layout.label(text="Sequences keep their own frame numbers on the timeline", icon='INFO')
        else:
            layout.prop(props, 'start_frame')


class VIPE_PT_lens(_SubPanel, bpy.types.Panel):
    bl_label = 'Lens'
    def draw(self, context):
        layout, props = self.layout, context.scene.vipe_remote
        layout.enabled = not props.busy
        layout.prop(props, 'lens_mode')
        if props.lens_mode == 'fixed':
            layout.prop(props, 'focal_mm')
        filmback = layout.column(align=True)
        filmback.prop(props, 'sensor_preset')
        filmback.prop(props, 'sensor_mm')
        if props.lens_mode == 'auto':
            layout.prop(props, 'lens_distortion')
            if props.lens_distortion:
                hint = layout.column(align=True)
                hint.active = False
                # Full-shot A/B (2026-10-19): equal review error with a very different focal on one shot.
                hint.label(text="Focal and distortion can trade off: compare", icon='INFO')
                hint.label(text="the review error with a pinhole solve first", icon='BLANK1')


class VIPE_PT_solve(_SubPanel, bpy.types.Panel):
    bl_label = 'Solve'
    bl_options = {'DEFAULT_CLOSED'}
    def draw(self, context):
        layout, props = self.layout, context.scene.vipe_remote
        layout.enabled = not props.busy
        layout.prop(props, 'max_dimension')
        layout.prop(props, 'masking')
        if props.masking:
            layout.prop(props, 'mask_phrases')


class VIPE_PT_outputs(_SubPanel, bpy.types.Panel):
    bl_label = 'Outputs'
    def draw(self, context):
        layout, props = self.layout, context.scene.vipe_remote
        width = _wrap_width(context)
        layout.enabled = not props.busy
        layout.prop(props, 'review_witnesses')
        _hint(layout, 'About 2.5–3× total time; can also run later from Results', width)
        layout.prop(props, 'dense_environment')
        if props.dense_environment:
            layout.prop(props, 'dense_mode', text='')
        _hint(layout, 'Fast mode: about 1–2 min extra' if props.dense_mode == 'fast' else 'Full video depth adds much more time', width)
        if props.dense_environment and props.lens_distortion and props.lens_mode == 'auto':
            row = layout.row()
            row.alert = True
            row.label(text='Dense does not support lens distortion yet', icon='ERROR')
        layout.separator()
        layout.prop(props, 'max_points')
        layout.prop(props, 'marker_count')
        layout.prop(props, 'scene_scale')
        layout.prop(props, 'auto_import')


class VIPE_PT_results(_SubPanel, bpy.types.Panel):
    bl_label = 'Results'
    @classmethod
    def poll(cls, context):
        return job_collection(context) is not None
    def draw(self, context):
        layout, props = self.layout, context.scene.vipe_remote
        width = _wrap_width(context)
        summary = _summary(job_collection(context))
        lines = _summary_lines(summary) or [props.lens_summary or 'Summary unavailable; reimport with this add-on version.']
        column = layout.column(align=True)
        for line in lines:
            for wrapped in textwrap.wrap(line, width):
                column.label(text=wrapped)
        if summary.get('has_witnesses'):
            _hint(layout, 'Per-frame review error (plate pixels, lower is better): Graph Editor › camera › vipe_reprojection_px', width)
        column = layout.column(align=True)
        column.operator('vipe_remote.review', icon='TRACKER')
        if summary.get('dense'):
            layer, _ = _dense_layer(context)
            visible = layer is not None and not layer.hide_viewport
            column.operator('vipe_remote.view_environment', text='Hide Dense Environment' if visible else 'Show Dense Environment', icon='HIDE_ON' if visible else 'MESH_GRID').show = not visible
        else:
            column.operator('vipe_remote.add_dense', icon='MESH_GRID')
        collection = job_collection(context)
        shown = bool(collection.get('vipe_markers_in_render', False))
        column.operator('vipe_remote.markers_in_render', text='Hide Markers in Renders' if shown else 'Show Markers in Renders', icon='RESTRICT_RENDER_ON' if shown else 'RESTRICT_RENDER_OFF').enable = not shown
        column.operator('vipe_remote.export_nuke', text='Export to Nuke…', icon='EXPORT')
        box = layout.box()
        box.label(text='Gaussian environment (optional)', icon='OUTLINER_DATA_POINTCLOUD')
        box.prop(props, 'splat_preset', text='')
        box.prop(props, 'splat_custom')
        if props.splat_custom:
            for key in SPLAT_KEYS:
                box.prop(props, key)
        box.operator('vipe_remote.train_splat', icon='RENDER_STILL')
        if collection.get('vipe_splat_directory'):
            names = json.loads(collection.get('vipe_splat_objects', '[]'))
            loaded = [bpy.data.objects.get(name) for name in names if bpy.data.objects.get(name)]
            visible = bool(loaded) and not loaded[0].hide_viewport
            box.operator('vipe_remote.view_splat', text='Hide Gaussian Splat' if visible else 'Show Gaussian Splat', icon='HIDE_ON' if visible else 'OUTLINER_OB_POINTCLOUD').show = not visible
            row = box.row(align=True)
            row.operator('vipe_remote.open_splat', text='Camera preview', icon='PLAY').kind = 'camera'
            row.operator('vipe_remote.open_splat', text='Nearby view', icon='PLAY').kind = 'nearby'
            row = box.row(align=True)
            row.operator('vipe_remote.open_splat', text='Masks', icon='IMAGE_DATA').kind = 'masks'
            row.operator('vipe_remote.open_splat', text='Output folder', icon='FILE_FOLDER').kind = 'folder'
        _hint(box, 'Static scene only; add Dense Environment first. Quick is a preview; Quality takes longer and, for EXR plates, uploads a larger training plate. View the splat with KIRI 3DGS Render (Blender 5.1+) or Blender 5.3+.', width - 2)
        box = layout.box()
        box.label(text='Scene alignment', icon='ORIENTATION_GLOBAL')
        grid = box.grid_flow(columns=2, even_columns=True, align=True)
        grid.operator('vipe_remote.set_floor', icon='MESH_PLANE')
        grid.operator('vipe_remote.set_origin', icon='OBJECT_ORIGIN')
        grid.operator('vipe_remote.set_scale', icon='DRIVER_DISTANCE')
        grid.operator('vipe_remote.set_axis', icon='EMPTY_SINGLE_ARROW')
        box.operator('vipe_remote.reset_alignment', icon='LOOP_BACK')
        _hint(box, 'Select markers, or cloud vertices in Edit Mode. Floor 3+, scale and axis 2 (active point is +X).', width - 2)


class VIPE_PT_jobs(_SubPanel, bpy.types.Panel):
    bl_label = 'Jobs'
    bl_options = {'DEFAULT_CLOSED'}
    def draw(self, context):
        layout, props, wm = self.layout, context.scene.vipe_remote, context.window_manager
        record = _loads(props.job_record)
        if props.job_id:
            layout.label(text=f"Extension {props.job_id[:8]} of job {record.get('parent_job_id', '')[:8]}" if record.get('mode') == 'review' else f"Job {props.job_id[:8]}")
        row = layout.row(align=True)
        row.operator('vipe_remote.resume', icon='FILE_REFRESH')
        row.operator('vipe_remote.log', icon='TEXT')
        row = layout.row(align=True)
        row.operator('vipe_remote.import_cached', icon='IMPORT')
        row.operator('vipe_remote.import_file', text='Saved Bundle', icon='FILE_FOLDER')
        row = layout.row()
        row.label(text='Recent jobs')
        row.operator('vipe_remote.refresh_jobs', text='', icon='FILE_REFRESH')
        layout.template_list('VIPE_UL_jobs', '', wm, 'vipe_jobs', wm, 'vipe_jobs_index', rows=3)
        item = _selected_job(context)
        if item is not None:
            _hint(layout, f"{item.job_id[:8]}" + (f" · frames {item.frames}" if item.frames else '') + (" · result downloaded" if item.has_result else " · no result yet"), _wrap_width(context))
        row = layout.row(align=True)
        row.operator('vipe_remote.load_job', icon='FILE_TICK')
        row.operator('vipe_remote.import_job', icon='IMPORT')


class VIPE_PT_logs(_SubPanel, bpy.types.Panel):
    bl_label = 'Logs'
    bl_options = {'DEFAULT_CLOSED'}
    def draw(self, context):
        layout, props = self.layout, context.scene.vipe_remote
        layout.operator('vipe_remote.open_log', icon='TEXT')
        layout.prop(props, 'echo_job_logs')
        layout.prop(props, 'follow_job_log')


@persistent
def _before_load(_):
    global _active
    if _active is not None:
        _active['detach'].set()
        _active = None
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    _log_buffers.clear()


CLASSES = [ViPERemotePreferences, ViPERemoteProperties, ViPEJobItem, VIPE_OT_check, VIPE_OT_submit, VIPE_OT_review, VIPE_OT_add_dense, VIPE_OT_train_splat, VIPE_OT_view_splat, VIPE_OT_open_splat, VIPE_OT_resume, VIPE_OT_cancel, VIPE_OT_import_cached, VIPE_OT_import_file, VIPE_OT_log, VIPE_OT_open_log, VIPE_OT_view_environment, VIPE_OT_markers_in_render, VIPE_OT_export_nuke, VIPE_OT_full_clip, VIPE_OT_refresh_jobs, VIPE_OT_load_job, VIPE_OT_import_job, *SCENE_TOOL_CLASSES, VIPE_UL_jobs, VIPE_PT_panel, VIPE_PT_worker, VIPE_PT_plate, VIPE_PT_lens, VIPE_PT_solve, VIPE_PT_outputs, VIPE_PT_results, VIPE_PT_jobs, VIPE_PT_logs]
PANELS = [cls for cls in CLASSES if issubclass(cls, bpy.types.Panel)]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.vipe_remote = PointerProperty(type=ViPERemoteProperties)
    bpy.types.WindowManager.vipe_jobs = CollectionProperty(type=ViPEJobItem)
    bpy.types.WindowManager.vipe_jobs_index = IntProperty(min=0)
    bpy.app.handlers.load_pre.append(_before_load)


def unregister():
    _before_load(None)
    if _before_load in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_before_load)
    for owner, name in [(bpy.types.Scene, 'vipe_remote'), (bpy.types.WindowManager, 'vipe_jobs'), (bpy.types.WindowManager, 'vipe_jobs_index')]:
        if hasattr(owner, name):
            delattr(owner, name)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
