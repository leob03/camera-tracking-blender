"""Export a ViPE job collection for Nuke: node script, chan file, Alembic and lens STMaps."""
from functools import partial
import json
import math
from pathlib import Path
import re

import bpy
from mathutils import Matrix
import numpy as np

from .exr_writer import write_exr


# Blender world (Z-up) to Nuke world (Y-up): (x, y, z) -> (x, z, -y). Camera axes agree.
TO_NUKE = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
MAX_POINTS = 50000
POINT_FORMAT_NOTE = "BakedPointCloud serialization ('N x y z ...') was written without Nuke and is unverified; the Alembic file also carries the cloud."


def _num(value):
    return f'{value:.10g}'


def _quote(text):
    for old, new in (('\\', '\\\\'), ('"', '\\"'), ('[', '\\['), (']', '\\]'), ('$', '\\$'), ('{', '\\{'), ('}', '\\}'), ('\n', '\\n')):
        text = text.replace(old, new)
    return f'"{text}"'


def _path(path):
    return _quote(str(path).replace('\\', '/'))


def _value(values, start):
    if max(values) - min(values) <= 1e-12 * max(1.0, abs(values[0])):
        return _num(values[0])
    return _curve(values, start)


def _curve(values, start):
    return '{curve x%d %s}' % (start, ' '.join(map(_num, values)))


def _node(kind, *knobs):
    return '\n'.join([kind + ' {'] + [f' {k} {v}' for k, v in knobs if v is not None] + ['}'])


def mei_project(lens, x, y, z):
    """MEI projection of camera-space points to OpenCV pixels (pixel centers at integers, v down)."""
    d = z + lens['xi'] * np.sqrt(x * x + y * y + z * z)
    valid = d > 1e-12
    d = np.where(valid, d, 1.0)
    return lens['fx'] * x / d + lens['cx'], lens['fy'] * y / d + lens['cy'], valid


def mei_unproject(lens, u, v):
    mx, my, xi = (u - lens['cx']) / lens['fx'], (v - lens['cy']) / lens['fy'], lens['xi']
    r2 = mx * mx + my * my
    s = 1 + (1 - xi * xi) * r2
    factor = (xi + np.sqrt(np.maximum(s, 0))) / (1 + r2)
    return mx * factor, my * factor, factor - xi, s >= 0


def stmaps(lens, width, height, band=256):
    """Nuke STMaps (rows from the top): undistort samples the plate, redistort samples the pinhole render.
    Returns the two {R,G,B} maps and their invalid-pixel counts; invalid pixels are set to -1."""
    f, cx, cy = lens['undistorted_focal_px'], lens['cx'], lens['cy']
    maps = [{k: np.zeros((height, width), np.float32) for k in 'RGB'} for _ in range(2)]
    invalid = [0, 0]
    for top in range(0, height, band):
        u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(top, min(height, top + band), dtype=np.float64))
        ud, vd, ok_u = mei_project(lens, (u - cx) / f, (v - cy) / f, np.ones_like(u))
        x, y, z, ok_r = mei_unproject(lens, u, v)
        ok_r &= z > 1e-9
        z = np.where(ok_r, z, 1.0)
        for index, (a, b, ok) in enumerate(((ud, vd, ok_u), (f * x / z + cx, f * y / z + cy, ok_r))):
            maps[index]['R'][top:top + len(u)] = np.where(ok, (a + 0.5) / width, -1)
            maps[index]['G'][top:top + len(u)] = np.where(ok, (height - b - 0.5) / height, -1)
            invalid[index] += int((~ok).sum())
    return maps[0], maps[1], invalid


def _filmback(data, width, height):
    """Horizontal aperture (mm) and the sensor size Blender's shift is measured in."""
    fit = data.sensor_fit if data.sensor_fit != 'AUTO' else ('HORIZONTAL' if width >= height else 'VERTICAL')
    if fit == 'HORIZONTAL':
        return data.sensor_width, data.sensor_width
    size = data.sensor_width if data.sensor_fit == 'AUTO' else data.sensor_height
    return size * width / height, size


def _camera_track(context, camera, frames, width, height, warnings):
    scene, rows, previous, notes = context.scene, [], None, set()
    for frame in frames:
        scene.frame_set(frame)
        evaluated = camera.evaluated_get(context.evaluated_depsgraph_get())
        matrix = TO_NUKE @ evaluated.matrix_world
        scale = matrix.to_scale()
        rotation = matrix.to_3x3().normalized()
        if max(scale) - min(scale) > 1e-6 * max(scale):
            notes.add("The camera has non-uniform scale; it was removed and the rotation may be approximate.")
        if rotation.determinant() < 0:
            notes.add("The camera matrix is mirrored (negative scale); Nuke rotations cannot represent it.")
        euler = rotation.to_euler('ZXY', previous) if previous is not None else rotation.to_euler('ZXY')
        previous = euler
        data = evaluated.data
        aperture, fit = _filmback(data, width, height)
        rows.append(dict(frame=frame, translate=tuple(matrix.to_translation()), rotate=tuple(math.degrees(a) for a in euler),
            focal=data.lens, haperture=aperture, vaperture=aperture * height / width,
            win=(2 * data.shift_x * fit / aperture, 2 * data.shift_y * fit / aperture), near=data.clip_start, far=data.clip_end))
    warnings.extend(sorted(notes))
    return rows


def _cloud(context, collection, eye, warnings):
    cloud = next((o for o in collection.all_objects if o.get('vipe_role') == 'native_cloud' and o.type == 'MESH'), None)
    if cloud is None or not len(cloud.data.vertices):
        warnings.append("No native point cloud in this job; the point cloud node was skipped.")
        return None, None
    mesh = cloud.data
    count = len(mesh.vertices)
    co = np.empty(count * 3, np.float32)
    mesh.vertices.foreach_get('co', co)
    colors = np.ones((count, 4), np.float32)
    attribute = mesh.color_attributes.get('Color')
    if attribute is not None and attribute.domain == 'POINT':
        attribute.data.foreach_get('color', colors.reshape(-1))
    else:
        warnings.append("The native cloud has no per-point 'Color' attribute; points are exported white.")
    keep = np.arange(count)
    if count > MAX_POINTS:
        keep = np.unique(np.linspace(0, count - 1, MAX_POINTS).round().astype(int))
        warnings.append(f"Native cloud reduced from {count} to {len(keep)} points for the Nuke script.")
    matrix = np.array(TO_NUKE @ cloud.evaluated_get(context.evaluated_depsgraph_get()).matrix_world)
    points = co.reshape(-1, 3)[keep].astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    normals = np.asarray(eye) - points
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return cloud, (points, normals, colors[keep, :3].astype(np.float64))


def _layer_path(layer_collection, collection):
    if layer_collection.collection == collection:
        return [layer_collection]
    for child in layer_collection.children:
        path = _layer_path(child, collection)
        if path:
            return [layer_collection] + path
    return []


def _alembic(context, path, camera, cloud, start, end, width, height, warnings):
    scene, layer = context.scene, context.view_layer
    selected, active, undo = [o for o in layer.objects if o.select_get()], layer.objects.active, []
    render = scene.render
    try:
        objects = [o for o in (camera, cloud) if o is not None]
        for obj in objects:
            for collection in obj.users_collection:
                for item in _layer_path(layer.layer_collection, collection):
                    if item.hide_viewport:
                        undo.append(partial(setattr, item, 'hide_viewport', True))
                        item.hide_viewport = False
            if obj.hide_get(view_layer=layer):
                undo.append(partial(obj.hide_set, True, view_layer=layer))
                obj.hide_set(False, view_layer=layer)
        # Export raw cloud vertices, not the display instances.
        for modifier in (cloud.modifiers if cloud is not None else []):
            if modifier.show_viewport:
                undo.append(partial(setattr, modifier, 'show_viewport', True))
                modifier.show_viewport = False
        for key, value in (('resolution_x', width), ('resolution_y', height), ('resolution_percentage', 100), ('pixel_aspect_x', 1.0), ('pixel_aspect_y', 1.0)):
            undo.append(partial(setattr, render, key, getattr(render, key)))
            setattr(render, key, value)
        layer.update()
        missing = [o.name for o in objects if not o.visible_get(view_layer=layer)]
        objects = [o for o in objects if o.visible_get(view_layer=layer)]
        if missing:
            warnings.append(f"Alembic skipped hidden objects: {', '.join(missing)}.")
        if camera not in objects:
            raise RuntimeError("the camera is not visible in the current view layer")
        for obj in selected:
            obj.select_set(False)
        for obj in objects:
            obj.select_set(True)
        layer.objects.active = camera
        allowed = bpy.ops.wm.alembic_export.get_rna_type().properties.keys()
        if not {'selected', 'start', 'end'} <= set(allowed):
            raise RuntimeError("this Blender's Alembic exporter lacks selection/frame-range options")
        options = dict(filepath=str(path), start=start, end=end, selected=True, flatten=True, vcolors=True, uvs=False, orcos=False,
            export_custom_properties=False, evaluation_mode='VIEWPORT', as_background_job=False, init_scene_frame_range=False)
        with context.temp_override(selected_objects=objects, selected_editable_objects=objects, active_object=camera, object=camera):
            result = bpy.ops.wm.alembic_export(**{k: v for k, v in options.items() if k in allowed})
        if 'FINISHED' not in result or not Path(path).is_file():
            raise RuntimeError(f"exporter returned {sorted(result)}")
        return True
    except Exception as error:
        warnings.append(f"Alembic export failed ({error}); the .nk and .chan files are unaffected.")
        return False
    finally:
        for obj in layer.objects:
            if obj.select_get():
                obj.select_set(False)
        for restore in reversed(undo):
            restore()
        for obj in selected:
            if obj.name in layer.objects:
                obj.select_set(True)
        layer.objects.active = active


def _frame_file(pattern, frame):
    pattern = re.sub(r'#+', lambda m: f'{frame:0{len(m.group())}d}', pattern)
    return re.sub(r'%0?(\d*)d', lambda m: f'{frame:0{m.group(1) or 1}d}', pattern)


def _plate(path, plate_first, first_source, start, count, fmt, warnings):
    if not path:
        warnings.append("No local plate path is stored on this job; a black Constant stands in for the plate.")
        return _node('Constant', ('inputs', 0), ('channels', 'rgb'), ('format', fmt), ('name', 'ViPE_Plate_missing'), ('xpos', 0), ('ypos', -300))
    text = str(path).replace('\\', '/')
    sequence = re.search(r'#+|%0?\d*d', text)
    knobs = [('inputs', 0), ('file', _path(text)), ('format', fmt)]
    if sequence:
        if plate_first is None:
            plate_first = start - first_source
            warnings.append("Plate sequence first frame is unknown; assumed the file numbers match the timeline.")
        first = int(plate_first) + first_source
        last = first + count - 1
        path = _frame_file(text, first)
        knobs += [('first', first), ('last', last), ('origfirst', first), ('origlast', last), ('origset', 'true')]
        if first != start:
            knobs += [('frame_mode', 'offset'), ('frame', start - first)]
    else:
        # Movie frame 1 is source frame 0; place the selected range at the job's start frame.
        last = first_source + count
        knobs += [('first', 1), ('last', last), ('origfirst', 1), ('origlast', last), ('origset', 'true'), ('frame_mode', '"start at"'), ('frame', start - first_source)]
    if not Path(path).is_file():
        warnings.append(f"The plate file was not found on this machine: {path}")
    return _node('Read', *knobs, ('name', 'ViPE_Plate'), ('xpos', 0), ('ypos', -300))


def _distortion(collection, width, height, warnings):
    text = collection.get('vipe_distortion')
    if not text:
        return None
    try:
        lens = json.loads(text)
        lens = {k: float(lens[k]) for k in ('fx', 'fy', 'cx', 'cy', 'xi')} | {'model': lens.get('model'),
            'undistorted_focal_px': float(lens.get('undistorted_focal_px') or lens['fx'] / (1 + lens['xi'])),
            'width': int(lens.get('width', width)), 'height': int(lens.get('height', height))}
    except (ValueError, KeyError, TypeError) as error:
        warnings.append(f"Lens distortion data is unreadable ({error}); STMaps were not written.")
        return None
    if lens['model'] != 'mei':
        warnings.append(f"Lens model {lens['model']!r} is not supported; STMaps were not written.")
        return None
    if (lens['width'], lens['height']) != (width, height):
        warnings.append(f"Lens data is for {lens['width']}x{lens['height']}, not the {width}x{height} plate; STMaps were not written.")
        return None
    return lens


def _readme(label, job, rows, width, height, fps, lens, have_abc):
    first, last = rows[0]['frame'], rows[-1]['frame']
    lines = [f"ViPE Remote Tracking export for Nuke: job {job}, frames {first}-{last}, {width}x{height}, {fps:.6g} fps.",
        f"{label}_track.nk: File > Import Script (or paste). Plate Read, ViPE_Camera (Camera2, rot_order ZXY), native point cloud, Scene + ScanlineRender check render.",
        f"{label}_camera.chan: frame tx ty tz rx ry rz vfov. Import onto a Camera with rot_order ZXY, haperture {_num(rows[0]['haperture'])} mm, vaperture {_num(rows[0]['vaperture'])} mm.",
        f"{label}_camera.abc: Alembic camera and native points (Y-up)." + ("" if have_abc else " NOT WRITTEN - see warnings."),
        "Units are ViPE's estimated scene scale (not surveyed); Blender alignment edits on the job's parent are baked in."]
    if lens:
        lines += [f"stmap_undistort.exr: STMap (uv rgb) from the distorted plate to the pinhole view ViPE_Camera matches (MEI xi={lens['xi']:.6g}, pinhole focal {lens['undistorted_focal_px']:.6g} px).",
            "stmap_redistort.exr: STMap (uv rgb) applying the lens back to CG rendered through ViPE_Camera. No overscan: corners outside the pinhole frame render black.",
            "STMap input wiring (src = image, stmap = map) was generated without Nuke; if an STMap looks wrong, swap its inputs (Shift+X)."]
    lines.append("File paths in the script are absolute; repath the Read nodes if this folder moves. See docs/NUKE.md in the add-on for conventions.")
    return '\n'.join(lines) + '\n'


def export_nuke(context, collection, directory):
    """Write the Nuke package for a ViPE job collection; returns {'nk', 'files', 'warnings'}."""
    if collection is None or not collection.get('vipe_job_id'):
        raise ValueError("Choose a ViPE job collection to export.")
    if not str(directory or '').strip():
        raise ValueError("Choose an output folder for the Nuke export.")
    text = bpy.data.texts.get(collection.get('vipe_manifest_text', ''))
    if text is None:
        raise ValueError("The job's manifest text block is missing; re-import the result bundle.")
    try:
        manifest = json.loads(text.as_string())
        source, selection = manifest['source'], manifest['selection']
        width, height, count = int(source['width']), int(source['height']), int(selection['count'])
        fps = source['fps_num'] / source['fps_den']
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        raise ValueError(f"The job's manifest text block is not a valid ViPE manifest ({error}).") from error
    camera = bpy.data.objects.get(collection.get('vipe_camera', ''))
    if camera is None or camera.type != 'CAMERA':
        camera = next((o for o in collection.all_objects if o.get('vipe_role') == 'camera' and o.type == 'CAMERA'), None)
    if camera is None:
        raise ValueError("The job's solved camera is missing from the scene.")
    if count < 1 or width < 1 or height < 1:
        raise ValueError("The job manifest has no frames or an invalid plate size.")
    folder = Path(bpy.path.abspath(str(directory))).resolve()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(f"Cannot create the output folder {folder}: {error}") from error
    job = str(collection['vipe_job_id'])
    start = int(collection.get('vipe_start_frame', context.scene.frame_start))
    first_source = int(selection.get('first_source_frame', 0))
    label = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(source.get('name', '')).stem).strip('_.') or f'vipe_{job[:8]}'
    warnings, files = [], []
    render = context.scene.render
    if abs(render.pixel_aspect_x - render.pixel_aspect_y) > 1e-6:
        warnings.append("The scene uses non-square pixels; the export assumes square pixels.")
    scene, old_frame = context.scene, context.scene.frame_current
    try:
        rows = _camera_track(context, camera, range(start, start + count), width, height, warnings)
        scene.frame_set(start)
        cloud, points = _cloud(context, collection, rows[0]['translate'], warnings)
    finally:
        scene.frame_set(old_frame)
    lens = _distortion(collection, width, height, warnings)
    if lens:
        first = rows[0]
        focal_px = first['focal'] / first['haperture'] * width
        if abs(focal_px / lens['undistorted_focal_px'] - 1) > 1e-3:
            warnings.append(f"Camera focal ({focal_px:.2f} px) differs from the lens model's pinhole focal ({lens['undistorted_focal_px']:.2f} px); CG will not match the undistorted plate.")
        cx, cy = width / 2 - first['win'][0] / 2 * width, height / 2 + first['win'][1] / 2 * width
        if max(abs(cx - lens['cx']), abs(cy - lens['cy'])) > 0.5:
            warnings.append(f"Camera principal point ({cx:.2f}, {cy:.2f}) differs from the lens model's ({lens['cx']:.2f}, {lens['cy']:.2f}) px.")
        undistort, redistort, invalid = stmaps(lens, width, height)
        for name, image, bad in (('stmap_undistort.exr', undistort, invalid[0]), ('stmap_redistort.exr', redistort, invalid[1])):
            files.append(str(write_exr(folder / name, image)))
            if bad:
                warnings.append(f"{name}: {bad} pixels have no valid lens ray and are set to -1 (black).")
    fmt = f'"{width} {height} 0 0 {width} {height} 1 "'
    focal_text = _num(rows[0]['focal']) + (' (animated)' if _value([r['focal'] for r in rows], start)[0] == '{' else '')
    info = (f"ViPE job {job}\nFrames {start}-{start + count - 1}, {width}x{height}\nFocal {focal_text} mm, filmback {_num(rows[0]['haperture'])} x {_num(rows[0]['vaperture'])} mm\n"
        + (f"Lens: MEI xi={lens['xi']:.6g}, pinhole focal {lens['undistorted_focal_px']:.6g} px (STMaps)" if lens else "Lens: no distortion model (pinhole)"))
    nodes = ['set cut_paste_input [stack 0]',
        _node('StickyNote', ('inputs', 0), ('name', 'ViPE_Info'), ('label', _quote(info)), ('note_font_size', 16), ('xpos', -520), ('ypos', -320)),
        _plate(collection.get('vipe_plate_path'), collection.get('vipe_plate_first_frame'), first_source, start, count, fmt, warnings),
        'set NViPEPlate [stack 0]']
    background = 'NViPEPlate'
    if lens:
        nodes += [_node('Read', ('inputs', 0), ('file', _path(folder / 'stmap_undistort.exr')), ('format', fmt), ('first', 1), ('last', 1), ('origfirst', 1), ('origlast', 1), ('origset', 'true'), ('raw', 'true'), ('name', 'ViPE_UndistortMap'), ('xpos', 160), ('ypos', -300)),
            'push $NViPEPlate',
            _node('STMap', ('inputs', 2), ('channels', 'rgba'), ('uv', 'rgb'), ('name', 'ViPE_Undistort'), ('label', _quote('src: plate, stmap: undistort map')), ('xpos', 0), ('ypos', -200)),
            'set NViPEUndistorted [stack 0]']
        background = 'NViPEUndistorted'
    win = [r['win'] for r in rows]
    has_win = max(abs(c) for w in win for c in w) > 1e-9
    nodes += [_node('Camera2', ('inputs', 0),
            ('translate', '{%s}' % ' '.join(_curve([r['translate'][i] for r in rows], start) for i in range(3))),
            ('rotate', '{%s}' % ' '.join(_curve([r['rotate'][i] for r in rows], start) for i in range(3))),
            ('rot_order', 'ZXY'), ('focal', _value([r['focal'] for r in rows], start)),
            ('haperture', _num(rows[0]['haperture'])), ('vaperture', _num(rows[0]['vaperture'])),
            ('near', _num(rows[0]['near'])), ('far', _num(rows[0]['far'])),
            ('win_translate', '{%s %s}' % (_value([w[0] for w in win], start), _value([w[1] for w in win], start)) if has_win else None),
            ('name', 'ViPE_Camera'), ('xpos', -320), ('ypos', -60)),
        'set NViPECamera [stack 0]']
    if points is not None:
        count_points = len(points[0])
        serial = lambda values: _quote(f'{count_points} ' + ' '.join(f'{v:.7g}' for v in values.reshape(-1)))
        nodes += [_node('BakedPointCloud', ('inputs', 0), ('serializePoints', serial(points[0])), ('serializeNormals', serial(points[1])),
            ('serializeColors', serial(points[2])), ('name', 'ViPE_NativeCloud'), ('xpos', -160), ('ypos', -160)),
            _node('Scene', ('inputs', 1), ('name', 'ViPE_Scene'), ('xpos', -150), ('ypos', -60))]
        warnings.append(POINT_FORMAT_NOTE)
    else:
        nodes.append(_node('Scene', ('inputs', 0), ('name', 'ViPE_Scene'), ('xpos', -150), ('ypos', -60)))
    nodes += ['set NViPEScene [stack 0]', 'push $NViPECamera', 'push $NViPEScene', f'push ${background}',
        _node('ScanlineRender', ('inputs', 3), ('name', 'ViPE_Render'), ('label', _quote(f"bg: {'undistorted ' if lens else ''}plate, obj/scn: scene, cam: ViPE_Camera")), ('xpos', 0), ('ypos', 0)),
        'set NViPERender [stack 0]']
    if lens:
        nodes += [_node('Read', ('inputs', 0), ('file', _path(folder / 'stmap_redistort.exr')), ('format', fmt), ('first', 1), ('last', 1), ('origfirst', 1), ('origlast', 1), ('origset', 'true'), ('raw', 'true'), ('name', 'ViPE_RedistortMap'), ('xpos', 160), ('ypos', 0)),
            'push $NViPERender',
            _node('STMap', ('inputs', 2), ('channels', 'rgba'), ('uv', 'rgb'), ('name', 'ViPE_Redistort'), ('label', _quote('src: render, stmap: redistort map')), ('xpos', 0), ('ypos', 100))]
    nk = folder / f'{label}_track.nk'
    nk.write_text('\n'.join(nodes) + '\n')
    chan = folder / f'{label}_camera.chan'
    chan.write_text(''.join('%d %s\n' % (r['frame'], ' '.join(map(_num, (*r['translate'], *r['rotate'],
        math.degrees(2 * math.atan(r['vaperture'] / (2 * r['focal']))))))) for r in rows))
    abc = folder / f'{label}_camera.abc'
    have_abc = _alembic(context, abc, camera, cloud, start, start + count - 1, width, height, warnings)
    readme = folder / 'README.txt'
    readme.write_text(_readme(label, job, rows, width, height, fps, lens, have_abc))
    files = [str(nk), str(chan)] + ([str(abc)] if have_abc else []) + files + [str(readme)]
    return {'nk': str(nk), 'files': files, 'warnings': warnings}
