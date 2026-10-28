"""GUI Blender: draw a splat with the viewport viewer, time redraws and screenshot it through the solved camera.

blender --window-geometry 0 0 1920 1080 <empty.blend> --python tests/blender_splat_viewer_gui.py -- result.zip plate.mp4 scene.ply out_dir [frame]
Needs a display (vglrun on a private Xvnc works). Writes out_dir/timings.json and PNG screenshots.
"""
import json
import math
import sys
import time
from pathlib import Path
import bpy
from mathutils import Euler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
args = sys.argv[sys.argv.index('--') + 1:]
bundle, plate, ply, out = args[:4]
frame = int(args[4]) if len(args) > 4 else 96
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
results = {}


def window():
    return bpy.context.window_manager.windows[0]


def area():
    return max((a for a in window().screen.areas if a.type == 'VIEW_3D'), key=lambda a: a.width * a.height)


def override():
    a = area()
    return bpy.context.temp_override(window=window(), area=a, region=next(r for r in a.regions if r.type == 'WINDOW'))


def shot(name):
    with bpy.context.temp_override(window=window(), area=area()):
        bpy.ops.screen.screenshot_area(filepath=str(out / f'{name}.png'))


def redraw_ms(count=10):
    with override():
        start = time.perf_counter()
        bpy.ops.wm.redraw_timer(type='DRAW', iterations=count)
    return round((time.perf_counter() - start) * 1000 / count, 1)


def steps():
    import addon_utils
    addon_utils.enable('vipe_remote', default_set=True)
    from vipe_remote import splat_viewer
    from vipe_remote.importer import import_bundle
    for obj in [o for o in bpy.data.objects if o.name == 'Cube']:
        bpy.data.objects.remove(obj)
    collection, _ = import_bundle(bundle, source_path=plate, start_frame=1)
    collection['vipe_splat_ply'], collection['vipe_splat_directory'] = ply, str(Path(ply).parent)
    scene = bpy.context.scene
    scene.vipe_remote.job_id = collection['vipe_job_id']
    scene.frame_set(frame)
    space = area().spaces.active
    space.region_3d.view_perspective = 'CAMERA'
    space.shading.type = 'SOLID'
    with override():
        bpy.ops.view3d.view_center_camera()
    yield 1
    start = time.perf_counter()
    with override():
        assert bpy.ops.vipe_remote.view_splat(show=True) == {'FINISHED'}
    while (splat_viewer.status(ply) or not splat_viewer.gaussian_count(ply)) and time.perf_counter() - start < 300:
        redraw_ms(1)
        yield 0.2
    assert not splat_viewer.status(ply), splat_viewer.status(ply)
    results['load_seconds'] = round(time.perf_counter() - start, 1)
    results['gaussians'] = splat_viewer.gaussian_count(ply)
    yield 1
    results['camera_redraw_ms'] = redraw_ms()
    shot('viewer_camera_plate')
    scene.camera.data.show_background_images = False
    yield 0.5
    shot('viewer_camera_splat')
    for degree in (0, 3):
        scene.vipe_remote.splat_sh_degree = degree
        results[f'redraw_ms_sh{degree}'] = redraw_ms()
    scene.vipe_remote.splat_display_percent = 30
    results['redraw_ms_30_percent'] = redraw_ms()
    scene.vipe_remote.splat_display_percent = 100
    scene.vipe_remote.splat_occlusion = False
    results['redraw_ms_no_occlusion'] = redraw_ms()
    scene.vipe_remote.splat_occlusion = True
    splat = bpy.data.objects[collection['vipe_splat_viewer']]
    splat.hide_viewport = True
    results['redraw_ms_splat_hidden'] = redraw_ms()
    splat.hide_viewport = False
    # Orbit around a point 4 m in front of the solved camera, up to 15 degrees either side; each step re-sorts.
    from mathutils import Vector
    r3d = space.region_3d
    r3d.view_perspective = 'PERSP'
    cam = scene.camera.matrix_world
    times = []
    for i in range(31):
        yaw = math.radians(i - 15)
        r3d.view_location = cam.translation + cam.to_3x3() @ Vector((0, 0, -4))
        r3d.view_rotation = (cam.to_3x3() @ Euler((0, yaw, 0)).to_matrix()).to_quaternion()
        r3d.view_distance = 4
        times.append(redraw_ms(1))
    results['orbit_redraw_ms_mean'] = round(sum(times) / len(times), 1)
    results['orbit_redraw_ms_max'] = max(times)
    yield 1
    redraw_ms(2)
    shot('viewer_orbit')
    results['viewer_errors'] = getattr(splat_viewer._draw, 'last_error', None)
    (out / 'timings.json').write_text(json.dumps(results, indent=1))
    print('VIEWER_GUI', json.dumps(results), flush=True)
    bpy.ops.wm.quit_blender()


gen = steps()


def tick():
    try:
        return next(gen)
    except StopIteration:
        return None
    except Exception:
        import traceback
        traceback.print_exc()
        bpy.ops.wm.quit_blender()
        return None


bpy.app.timers.register(tick, first_interval=3)
