"""Materialize only the declared Gaussian outputs; never execute downloaded content."""

import hashlib
from pathlib import Path
import zipfile

SPLAT_FILES = {
    "splat/scene.ply",
    "splat/camera_preview.mp4",
    "splat/nearby_preview.mp4",
    "splat/masks_preview.jpg",
    "splat/dataset.json",
}


def materialize_splat(bundle, manifest):
    destination = Path(bundle).parent / "splat"
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(bundle) as z:
        for name in sorted(SPLAT_FILES):
            info = manifest["files"][name]
            target = destination / Path(name).name
            temporary = target.with_suffix(target.suffix + ".part")
            h = hashlib.sha256()
            count = 0
            try:
                with z.open(name) as src, temporary.open("wb") as dst:
                    for data in iter(lambda: src.read(1024 * 1024), b""):
                        count += len(data)
                        if count > info["bytes"]:
                            raise ValueError("Splat file exceeds declared size.")
                        dst.write(data)
                        h.update(data)
                if count != info["bytes"] or h.hexdigest() != info["sha256"]:
                    raise ValueError("Corrupt splat output: " + name)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


def check_splat_camera(collection, manifest):
    if collection.get("vipe_source_sha256") != manifest["source"]["sha256"]:
        raise ValueError("The splat belongs to a different plate.")
    import bpy
    import json

    report = bpy.data.texts.get(collection.get("vipe_manifest_text", ""))
    if report is not None:
        original = json.loads(report.as_string())
        if (
            original["files"]["camera.npz"]["sha256"]
            != manifest["splat"]["camera_sha256"]
        ):
            raise ValueError("The splat uses a different camera solve from this scene.")


def attach_splat(collection, bundle, manifest):
    check_splat_camera(collection, manifest)
    folder = materialize_splat(bundle, manifest)
    collection["vipe_splat_directory"] = str(folder.resolve())
    collection["vipe_splat_ply"] = str((folder / "scene.ply").resolve())
    collection["vipe_splat_job_id"] = manifest["job_id"]
    collection["vipe_splat_bundle"] = str(Path(bundle).resolve())
    collection["vipe_splat_coordinates"] = manifest["splat"]["coordinates"]


def _ply_positions(path, count=2000):
    """The first vertex positions of a binary little-endian Gaussian PLY (x, y, z are its first float properties)."""
    import numpy as np
    with open(path, 'rb') as stream:
        header = []
        while not header or header[-1] != 'end_header':
            header.append(stream.readline().decode('ascii', 'replace').strip())
            if len(header) > 400:
                raise ValueError('Unreadable PLY header.')
        if 'format binary_little_endian 1.0' not in header:
            raise ValueError('Expected a binary little-endian Gaussian PLY.')
        properties = [line.split() for line in header if line.startswith('property')]
        if [p[2] for p in properties[:3]] != ['x', 'y', 'z'] or any(p[1] != 'float' for p in properties):
            raise ValueError('Unexpected Gaussian PLY layout.')
        data = np.fromfile(stream, dtype='<f4', count=count * len(properties))
    return data.reshape(-1, len(properties))[:, :3]


def _object_positions(obj, count):
    import numpy as np
    data = obj.data
    points = data.vertices if obj.type == 'MESH' else data.points
    co = np.empty(len(points) * 3, np.float32)
    points.foreach_get('co', co)
    return co.reshape(-1, 3)[:count]


def import_correction(obj, path):
    """Local matrix that puts imported splat positions back in the PLY's ViPE world, whichever importer converted axes."""
    import itertools
    import numpy as np
    from mathutils import Matrix
    raw = _ply_positions(path)
    got = _object_positions(obj, len(raw)).astype(np.float64)
    raw = raw[:len(got)]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            m[range(3), perm] = signs  # got = m @ raw
            if np.allclose(got, raw @ m.T, atol=1e-4 * (1 + np.abs(raw).max())):
                return Matrix(m.T.tolist()).to_4x4()  # undo it under the alignment parent
    raise ValueError('The imported splat does not match its PLY positions; it was not placed.')


def kiri_available():
    import bpy
    return hasattr(bpy.ops, 'sna') and 'dgs_render_import_ply_e0a3a' in dir(bpy.ops.sna)


def kiri_import(context, path):
    """Import a Gaussian PLY with KIRI 3DGS Render; returns the new objects, the splat first."""
    import bpy
    before = set(bpy.data.objects)
    area = next((a for a in context.window.screen.areas if a.type == 'VIEW_3D'), None) if context.window else None
    override = dict(area=area, region=next(r for r in area.regions if r.type == 'WINDOW')) if area else {}
    with context.temp_override(**override):
        if bpy.ops.sna.dgs_render_import_ply_e0a3a('EXEC_DEFAULT', filepath=str(path)) != {'FINISHED'}:
            raise ValueError('KIRI 3DGS Render could not import the splat.')
    obj = context.view_layer.objects.active
    created = [o for o in bpy.data.objects if o not in before]
    return [obj] + [o for o in created if o is not obj]


def open_kiri_tab(context):
    """Show KIRI 3DGS Render's sidebar tab in Edit mode at its Import step, as after KIRI's own Import PLY."""
    props = getattr(context.scene, 'sna_dgs_scene_properties', None)
    if props is not None:
        if getattr(props, 'active_mode', 'Edit') != 'Edit':
            props.active_mode = 'Edit'
        if hasattr(props, 'edit_mode_menu'):
            props.edit_mode_menu = 'Import'
    import bpy
    areas = [a for a in (context.screen.areas if context.screen else []) if a.type == 'VIEW_3D']
    for area in areas:
        area.spaces.active.show_region_ui = True
        area.tag_redraw()
    tries = [20]

    def switch_tab():
        # Blender accepts a sidebar tab only once the sidebar has drawn its tab list, so retry briefly.
        pending = False
        for area in areas:
            region = next((r for r in area.regions if r.type == 'UI'), None)
            try:
                if region is not None and region.active_panel_category != '3DGS Render':
                    region.active_panel_category = '3DGS Render'
                    area.tag_redraw()
            except (AttributeError, ReferenceError, TypeError, ValueError):
                pending = True
        tries[0] -= 1
        return 0.1 if pending and tries[0] > 0 else None
    if areas:
        bpy.app.timers.register(switch_tab, first_interval=0.1)
