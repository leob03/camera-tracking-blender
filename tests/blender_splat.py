"""Real parent + splat bundles; verify attachment preserves the imported scene.
BLENDER_SPLAT_ROOT points to the bounded experiment directory with full_result/result.zip.
"""

import json, os, sys
from pathlib import Path
import bpy, addon_utils

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
addon_utils.enable("vipe_remote", default_set=True, persistent=True)
from vipe_remote import addon
from vipe_remote.importer import import_bundle
from vipe_remote.bundle import load_bundle

root = Path(os.environ["BLENDER_SPLAT_ROOT"])
parent_bundle = Path(os.environ["BLENDER_SPLAT_PARENT"])
source = os.environ.get("BLENDER_SPLAT_SOURCE", "")
m, _, _ = load_bundle(parent_bundle)
collection, created = import_bundle(parent_bundle, source_path=source, start_frame=1001)
assert created and collection["vipe_dense_job_id"] == m["job_id"]
props = bpy.context.scene.vipe_remote
folder = root / "blender_cache" / m["job_id"]
folder.mkdir(parents=True, exist_ok=True)
parent = dict(
    job_id=m["job_id"],
    source=source,
    source_sha256=m["source"]["sha256"],
    server="http://127.0.0.1:8191",
    folder=str(folder),
    result=str(parent_bundle),
    start_frame=1001,
    marker_count=100,
    scene_scale=1.0,
    sensor_width=36.0,
    auto_import=True,
)
addon._save_record(parent)
props.job_id, props.job_record = m["job_id"], json.dumps(parent)
assert bpy.ops.vipe_remote.train_splat.poll()
objects = len(bpy.data.objects)
camera = bpy.context.scene.camera
bpy.context.scene.frame_set(1318)
before = tuple(tuple(row) for row in camera.matrix_world)
full = root / "full_result/result.zip"
sm, _, _ = load_bundle(full)
record = {
    **parent,
    "mode": "splat",
    "job_id": sm["job_id"],
    "parent_job_id": m["job_id"],
    "scene_job_id": m["job_id"],
    "scene_record": parent,
    "folder": str(full.parent),
    "result": str(full),
}
props.job_id, props.job_record, props.result_path = (
    sm["job_id"],
    json.dumps(record),
    str(full),
)
attached = addon._import(props, record)
assert (
    attached == collection
    and len(bpy.data.objects) == objects
    and bpy.context.scene.camera == camera
)
assert tuple(tuple(row) for row in camera.matrix_world) == before
assert props.job_id == m["job_id"] and props.state == "COMPLETE"
assert (
    Path(collection["vipe_splat_ply"]).stat().st_size
    == sm["files"]["splat/scene.ply"]["bytes"]
)
assert bpy.ops.vipe_remote.train_splat.poll()
# Repeat attachment is idempotent and does not duplicate scene geometry.
props.result_path = str(full)
addon._import(props, record)
assert len(bpy.data.objects) == objects
bpy.context.scene.frame_set(1001)
bpy.ops.wm.save_as_mainfile(
    filepath=str(root / "review/skateboard_alley_splat_review.blend")
)
receipt = dict(
    state="PASS",
    blender=bpy.app.version_string,
    camera_preserved=True,
    geometry_not_duplicated=True,
    points=len(bpy.data.objects),
    ply=collection["vipe_splat_ply"],
    scene=str(root / "review/skateboard_alley_splat_review.blend"),
)
(root / "blender_validation.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
