"""Real Blender operator -> ComfyUI -> download -> scene attachment test.
Needs the saved full review scene and isolated worker prepared by the experiment.
"""

import json, os, sys, time
from pathlib import Path
import bpy, addon_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
addon_utils.enable("vipe_remote", default_set=True, persistent=True)
from vipe_remote import addon

root = Path(os.environ["BLENDER_SPLAT_ROOT"])
bpy.ops.wm.open_mainfile(
    filepath=str(root / "review/skateboard_alley_splat_review.blend")
)
props = bpy.context.scene.vipe_remote
prefs = addon.preferences(bpy.context)
prefs.cache_directory = str(root / "blender_api_cache")
prefs.worker_url = "http://127.0.0.1:8191"
props.splat_iterations = 100
props.splat_max_dimension = 512
props.splat_frame_step = 6
props.splat_max_gaussians = 50000
collection = addon.job_collection(bpy.context)
old_count = len(bpy.data.objects)
camera = bpy.context.scene.camera
assert bpy.ops.vipe_remote.train_splat() == {"FINISHED"}
identifier = props.job_id
assert json.loads(props.job_record)["parent_job_id"] == collection["vipe_dense_job_id"]
deadline = time.monotonic() + 180
while addon._active is not None and time.monotonic() < deadline:
    addon._poll()
    time.sleep(0.15)
assert addon._active is None and props.state == "COMPLETE", (props.state, props.status)
assert collection["vipe_splat_job_id"] == identifier
assert bpy.context.scene.camera == camera and len(bpy.data.objects) == old_count
assert props.job_id == collection["vipe_job_id"]
receipt = dict(
    state="PASS",
    blender=bpy.app.version_string,
    job_id=identifier,
    operator_submission=True,
    download_and_attachment=True,
    log_text=props.log_text_name,
    status=props.status,
)
(root / "blender_api_validation.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
