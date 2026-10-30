"""Run: blender -b --factory-startup --python-exit-code 1 --python tests/blender_ui_register.py

Registration, panels (mocked layout), plate probe, job history and the review flow with a scripted
client; no network, inference or real bundle."""
from pathlib import Path
import json, os, sys, tempfile, time, types
import bpy, addon_utils
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import vipe_remote
vipe_remote.register()
from vipe_remote import addon, scene_tools
assert all(cls.is_registered for cls in addon.CLASSES)
assert hasattr(bpy.types.Scene, 'vipe_remote') and hasattr(bpy.types.WindowManager, 'vipe_jobs')
vipe_remote.unregister()
assert not any(cls.is_registered for cls in addon.CLASSES) and not hasattr(bpy.types.WindowManager, 'vipe_jobs')
addon_utils.enable('vipe_remote', default_set=True, persistent=True)
assert all(cls.is_registered for cls in addon.CLASSES)
ICONS = set(bpy.types.UILayout.bl_rna.functions['label'].parameters['icon'].enum_items.keys())
props, prefs, wm = bpy.context.scene.vipe_remote, addon.preferences(bpy.context), bpy.context.window_manager
temp = tempfile.TemporaryDirectory(); TMP = Path(temp.name)
cache = TMP / 'cache'; cache.mkdir(); prefs.cache_directory = str(cache)
prefs.worker_url = 'http://127.0.0.1:9'  # Never contacted: every network test uses a scripted client.


class Layout:
    """Records draw calls and validates property, operator and icon names against Blender."""
    def __init__(self, log): self.__dict__['log'] = log
    def __setattr__(self, name, value): self.__dict__[name] = value
    def _child(self, *args, **kwargs): return Layout(self.log)
    row = column = box = split = grid_flow = column_flow = _child
    def _icon(self, icon): assert icon in ICONS, icon
    def label(self, text='', icon='NONE', **kwargs): self._icon(icon); self.log.append(text)
    def prop(self, data, name, text=None, icon='NONE', **kwargs):
        self._icon(icon); assert name in data.bl_rna.properties, name; self.log.append('prop:' + name)
    def operator(self, idname, text=None, icon='NONE', **kwargs):
        self._icon(icon); module, name = idname.split('.')
        getattr(getattr(bpy.ops, module), name).get_rna_type()
        self.log.append('op:' + idname)
        if text: self.log.append('optext:' + text)
        return types.SimpleNamespace()
    def separator(self, **kwargs): pass
    def progress(self, factor=0, type='BAR', text=''): assert 0 <= factor <= 1; self.log.append('progress:' + text)
    def template_list(self, listtype, list_id, data, name, active_data, active_name, **kwargs):
        assert getattr(bpy.types, listtype).is_registered and name in data.bl_rna.properties and active_name in active_data.bl_rna.properties
        self.log.append('list:' + name)


class Context:
    region = types.SimpleNamespace(width=320)
    def __getattr__(self, name): return getattr(bpy.context, name)


def draw_all():
    log, context = [], Context()
    for cls in addon.PANELS:
        if 'poll' in cls.__dict__ and not cls.poll(context):
            continue
        cls.draw(types.SimpleNamespace(layout=Layout(log)), context)
    return log


def expect_error(op, text, **kwargs):
    try:
        op(**kwargs)
    except RuntimeError as exc:
        assert text in str(exc), exc
    else:
        raise AssertionError(f'{op} should fail with: {text}')


# Panel structure.
panels = {cls.__name__: cls for cls in addon.PANELS}
assert all(cls.bl_parent_id == 'VIPE_PT_remote_tracking' for name, cls in panels.items() if name != 'VIPE_PT_panel')
assert all('DEFAULT_CLOSED' in cls.bl_options for name, cls in panels.items() if name != 'VIPE_PT_panel')  # every sub-panel starts closed
assert not props.review_witnesses and props.sensor_preset == 'FULL_FRAME' and props.sensor_mm == 36.0
log = draw_all()
assert 'Worker not checked' in log and 'op:vipe_remote.submit' in log and 'list:vipe_jobs' in log
assert 'op:vipe_remote.set_floor' not in log  # Results only with an imported job.
assert not any('reconstruction hypotheses' in line for line in log)
assert 'prop:lens_distortion' in log and 'prop:dense_mode' not in log
props.lens_mode = 'fixed'; assert 'prop:lens_distortion' not in draw_all() and 'prop:focal_mm' in draw_all()
props.lens_mode = 'auto'

# Filmback presets and solve settings.
props.sensor_preset = 'ALEXA_LF_OG'; assert abs(props.sensor_mm - 36.70) < 1e-5
props.sensor_mm = 30.0; assert props.sensor_preset == 'CUSTOM'
props.sensor_preset = 'SUPER35'; assert abs(props.sensor_mm - 24.89) < 1e-5 and props.sensor_preset == 'SUPER35'
settings = addon.settings_from_properties(props)
assert settings['lens_distortion'] is False and 'dense_mode' not in settings and settings['sensor_mm'] == props.sensor_mm
props.lens_distortion = True; props.dense_environment = True
settings = addon.settings_from_properties(props)
assert settings['lens_distortion'] is True and settings['dense_mode'] == 'fast'
assert 'Dense does not support lens distortion yet' in draw_all()
props.lens_mode = 'fixed'; assert addon.settings_from_properties(props)['lens_distortion'] is False
props.lens_mode = 'auto'

# Plate probe runs in the update callback, never leaks a clip and tolerates bad paths.
clips = len(bpy.data.movieclips)
props.source = str(TMP / 'missing plate.mp4')
assert json.loads(props.plate_info) == {'error': 'Video file not found'} and len(bpy.data.movieclips) == clips
(TMP / 'not a video.mp4').write_bytes(b'not a movie')
props.source = str(TMP / 'not a video.mp4'); assert 'error' in json.loads(props.plate_info) and len(bpy.data.movieclips) == clips
image = bpy.data.images.new('probe', 64, 32); image.file_format = 'PNG'
for name in ('plate.png', 'seq.1001.png', 'seq.1002.png'):
    image.filepath_raw = str(TMP / name); image.save()
props.source = str(TMP / 'plate.png')
assert 'numbered image sequence' in json.loads(props.plate_info)['error'], props.plate_info
props.source = str(TMP / 'seq.1001.png')
info = json.loads(props.plate_info)
assert (info['width'], info['height'], info['frames'], info['first'], info['sequence']) == (64, 32, 2, 1001, True) and len(bpy.data.movieclips) == clips, info
assert props.start_frame == 1001 and 'plate 1001–1002' in addon._range_text(props)[0], addon._range_text(props)
props.source = ''; assert props.plate_info == ''
props.plate_info = json.dumps(dict(width=1920, height=1080, frames=1059, fps=23.976))
assert addon._range_text(props) == ('Frames 1–1059 (1059 frames, 44.2 s)', 'PREVIEW_RANGE')
assert addon._plate_lines(props)[0][0] == '1920 × 1080, 23.976 fps' and addon._plate_lines(props)[1][0] == '1059 frames, 44.2 s'
props.frame_start, props.frame_end = 100, 2000; assert addon._range_text(props)[1] == 'ERROR'
assert bpy.ops.vipe_remote.full_clip() == {'FINISHED'} and (props.frame_start, props.frame_end) == (1, 0)
assert 'Frames 1–1059 (1059 frames, 44.2 s)' in draw_all()

# Submit refuses an option combination the worker rejects, before any network request.
props.source = str(TMP / 'plate.png')
expect_error(bpy.ops.vipe_remote.submit, 'lens distortion')
assert addon._active is None and not props.busy
props.lens_distortion = props.dense_environment = False

# Stage mapping uses the base message only.
S = addon.STAGE_ORDER.index
for message, stage in [('Checking plate on worker', 'upload'), ('Plate already on worker', 'upload'), ('Uploading video', 'upload'), ('Waiting for worker', 'queue'), ('Preparing review', 'prepare'),
                       ('Tracking camera and reconstructing native points · SLAM Pass (1/2) 45%', 'solve'), ('Triangulating independent tracks through the frozen camera', 'witnesses'),
                       ('Computing optional dense video depth', 'dense'), ('Dense view 3/12', 'dense'), ('Packaging tracking results', 'package'), ('Downloading result', 'download')]:
    assert addon._stage(message) == S(stage), message
assert addon._stage('Cancelling job') is None

# Job history: empty folder, then bounded reads of immediate job folders.
empty = TMP / 'empty'; empty.mkdir(); prefs.cache_directory = str(empty)
assert bpy.ops.vipe_remote.refresh_jobs() == {'FINISHED'} and len(wm.vipe_jobs) == 0
prefs.cache_directory = str(TMP / 'does not exist')
assert bpy.ops.vipe_remote.refresh_jobs() == {'FINISHED'} and len(wm.vipe_jobs) == 0
prefs.cache_directory = str(cache)

# A completed parent job imported in the scene, without witnesses yet.
PARENT = '00000000-0000-4000-8000-0000000000bb'
parent_folder = cache / PARENT; parent_folder.mkdir()
(parent_folder / 'result.zip').write_bytes(b'parent')
parent = dict(job_id=PARENT, source='', server=prefs.worker_url, folder=str(parent_folder), source_sha256='abc', start_frame=1001, marker_count=50, scene_scale=1.5, sensor_width=27.99, auto_import=True,
              settings=dict(frame_start=1, frame_end=0, review_witnesses=False, dense_environment=False), result=str(parent_folder / 'result.zip'))
(parent_folder / 'job.json').write_text(json.dumps(parent))
collection = bpy.data.collections.new('ViPE | plate | 00000000'); bpy.context.scene.collection.children.link(collection)
collection['vipe_job_id'] = PARENT
summary = dict(job_id=PARENT, frames=1059, focal_mm=31.2, focal_px=1664.0, sensor_width_mm=27.99, points=30000, witness_landmarks=None, reprojection_px_median=None, reprojection_px_p90=None,
               distortion=None, timings=dict(camera=133.0, worker_total=200.5), lens_initial_px=1500.0, lens_final_px=1664.0, dense=False, has_witnesses=False)
collection['vipe_summary'] = json.dumps(summary)
root = bpy.data.objects.new('ViPE alignment', None); collection.objects.link(root); root['vipe_role'] = 'alignment'
props.job_id, props.job_record = PARENT, json.dumps(parent)
assert scene_tools.job_collection(bpy.context) == collection
log = draw_all()
for text in ['Frames: 1,059', 'Focal: 31.20 mm @ 27.99 mm filmback', 'Focal: 1664.0 px', 'Native points: 30,000', 'op:vipe_remote.set_floor', 'op:vipe_remote.export_nuke']:
    assert text in log, text
assert any(line.startswith('Time: camera 2:13') for line in log)
assert bpy.ops.vipe_remote.review.poll()
collection['vipe_summary'] = json.dumps({**summary, 'has_witnesses': True})
assert not bpy.ops.vipe_remote.review.poll() and any('vipe_reprojection_px' in line for line in draw_all())
collection['vipe_summary'] = json.dumps(summary)

# Export delegates to nuke_export.export_nuke and reports its warnings.
calls = []
def export_nuke(context, target, directory):
    calls.append((target.name, directory)); return dict(nk=str(Path(directory) / 'shot.nk'), files=[str(Path(directory) / 'shot.nk')], warnings=['Synthetic warning'])
saved = sys.modules.get('vipe_remote.nuke_export'), getattr(vipe_remote, 'nuke_export', None)
sys.modules['vipe_remote.nuke_export'] = vipe_remote.nuke_export = types.SimpleNamespace(export_nuke=export_nuke)
try:
    assert bpy.ops.vipe_remote.export_nuke(directory=str(TMP)) == {'FINISHED'}
finally:
    for name, value in [('vipe_remote.nuke_export', saved[0])]:
        sys.modules.pop(name) if value is None else sys.modules.__setitem__(name, value)
    if saved[1] is None: del vipe_remote.nuke_export
    else: vipe_remote.nuke_export = saved[1]
assert calls == [(collection.name, str(TMP))] and 'shot.nk' in props.status

# Review on demand: scripted worker, real thread, real event loop, mocked bundle import.
submitted, imported = [], []
class ScriptedClient:
    def __init__(self, url, token): assert url == prefs.worker_url
    def submit_review(self, parent_job_id, identifier, **extend):
        submitted.append((parent_job_id, identifier, extend)); return {'1': {'class_type': 'ViPEReviewJob'}}
    def wait(self, identifier, progress, cancel, detach, log=None):
        progress('Preparing review', .05)
        for i in range(4):
            progress(f'Generating independent CoTracker tracks · CoTracker window {i + 1} (to frame {60 * i})', .5); time.sleep(.21)
        progress('Packaging tracking results', .95)
        return dict(type='output', filename='result.zip')
    def download(self, item, destination, progress, stop):
        progress('Downloading result', 1); Path(destination).write_bytes(b'review'); return Path(destination)
def fake_load_bundle(path, sha=None):
    job = Path(path).parent.name
    return dict(job_id=job, source=dict(sha256='abc'), selection=dict(count=1059), cloud=dict(point_count=30000), **({'review_of': PARENT} if job != PARENT else {})), None, None
def fake_import(path, **kwargs):
    imported.append(kwargs); collection['vipe_review_merged'] = True; return collection, False
original = addon.Client, addon.load_bundle, addon.import_bundle
addon.Client, addon.load_bundle, addon.import_bundle = ScriptedClient, fake_load_bundle, fake_import
try:
    assert bpy.ops.vipe_remote.review() == {'FINISHED'}
    review_id = props.job_id
    assert review_id != PARENT and props.busy and addon._active['mode'] == 'review'
    assert scene_tools.job_collection(bpy.context) == collection  # Results stay visible during the review.
    deadline, seen, drawn = time.monotonic() + 10, set(), 0
    while addon._active is not None:
        addon._poll(); time.sleep(.02); assert time.monotonic() < deadline
        if addon._active: seen.add(addon._active.get('stage'))
        if props.busy and ' · ' in props.status:
            log = draw_all()
            assert 'Review witnesses' in log and 'Camera solve' not in log and props.status.split(' · ', 1)[1] in log
            assert any(line.startswith('Elapsed ') for line in log) and 'op:vipe_remote.cancel' in log
            drawn += 1
    assert S('witnesses') in seen and drawn, (seen, drawn)
    assert submitted == [(PARENT, review_id, {'review_witnesses': True})], submitted
    assert props.job_id == PARENT and props.status == f'Review witnesses added to job {PARENT[:8]}', props.status
    assert json.loads(props.job_record)['review_job_id'] == review_id and props.result_path == str(parent_folder / 'result.zip')
    assert imported[0]['sensor_width'] == 27.99 and imported[0]['start_frame'] == 1001 and imported[0]['marker_count'] == 50
    review = json.loads((cache / review_id / 'job.json').read_text())
    assert review['mode'] == 'review' and review['parent_job_id'] == PARENT and review['source_sha256'] == 'abc' and review['auto_import'] and review['settings'] == parent['settings']
    assert (cache / review_id / 'workflow_api.json').is_file() and (cache / review_id / 'result.zip').is_file()
    assert json.loads((parent_folder / 'job.json').read_text())['review_job_id'] == review_id
    text = bpy.data.texts['ViPE Log ' + review_id].as_string()
    assert text.count('Generating independent CoTracker tracks') == 1 and 'CoTracker window' not in text, text
    assert props.connection_state == 'OK' and 'Worker connected' in draw_all()

    # History lists both jobs, newest first (the parent record was rewritten by the merge).
    os.utime(cache / review_id / 'job.json', (time.time() - 60,) * 2)
    assert bpy.ops.vipe_remote.refresh_jobs() == {'FINISHED'} and [item.job_id for item in wm.vipe_jobs] == [PARENT, review_id]
    assert wm.vipe_jobs[1].name == f'Review of {PARENT[:8]}' and wm.vipe_jobs[1].has_result
    assert wm.vipe_jobs[0].name == 'Saved job' and wm.vipe_jobs[0].frames == '1–end' and wm.vipe_jobs[0].has_result
    for i, item in enumerate(wm.vipe_jobs):
        addon.VIPE_UL_jobs.draw_item(None, Context(), Layout([]), wm, item, 0, wm, 'vipe_jobs_index', i)
    wm.vipe_jobs_index = 0
    assert any(line.startswith(PARENT[:8] + ' · frames 1–end') for line in draw_all())
    assert bpy.ops.vipe_remote.load_job() == {'FINISHED'} and props.job_id == PARENT and props.state == 'READY'
    assert bpy.ops.vipe_remote.import_job() == {'FINISHED'} and props.status == 'This job is already imported; its collection was preserved.'
    assert imported[-1]['sensor_width'] == 27.99

    # Add Dense Environment runs the same extension path with dense flags and merges into the job.
    submitted.clear()
    collection['vipe_summary'] = json.dumps({**json.loads(collection['vipe_summary']), 'dense': False, 'distortion': None})
    assert 'op:vipe_remote.add_dense' in draw_all()
    assert bpy.ops.vipe_remote.add_dense() == {'FINISHED'}
    dense_id = props.job_id
    deadline = time.monotonic() + 10
    while addon._active is not None:
        addon._poll(); time.sleep(.02); assert time.monotonic() < deadline
    assert submitted == [(PARENT, dense_id, {'review_witnesses': False, 'dense_environment': True, 'dense_mode': props.dense_mode})], submitted
    assert props.job_id == PARENT and props.status.startswith(f'Dense environment added to job {PARENT[:8]}'), props.status
    # Show / Hide Dense Environment toggles the job's dense layer.
    group = bpy.data.collections.new('05 Dense environment'); group['vipe_role'] = 'dense_environment_group'; collection.children.link(group)
    collection['vipe_summary'] = json.dumps({**json.loads(collection['vipe_summary']), 'dense': True})
    layer = bpy.context.view_layer.layer_collection.children[collection.name].children[group.name]
    layer.hide_viewport = True
    log = draw_all(); assert 'optext:Show Dense Environment' in log and 'op:vipe_remote.add_dense' not in log
    assert bpy.ops.vipe_remote.view_environment(show=True) == {'FINISHED'} and not layer.hide_viewport
    assert 'optext:Hide Dense Environment' in draw_all()
    assert bpy.ops.vipe_remote.view_environment(show=False) == {'FINISHED'} and layer.hide_viewport
finally:
    addon.Client, addon.load_bundle, addon.import_bundle = original
    addon._before_load(None)

addon_utils.disable('vipe_remote')
assert not any(cls.is_registered for cls in addon.CLASSES) and not hasattr(bpy.types.Scene, 'vipe_remote')
temp.cleanup()
print(json.dumps(dict(state='PASS', blender=bpy.app.version_string, classes=len(addon.CLASSES), panels=len(addon.PANELS), review_flow=True, job_history=True, plate_probe=True, mocked_draw=True)))
