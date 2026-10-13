"""Exercise the live log UI with synthetic output; no inference or HTTP required."""
from pathlib import Path
import json, queue, sys, threading
import bpy, addon_utils
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
addon_utils.enable('vipe_remote',default_set=True,persistent=True)
from vipe_remote import addon
from vipe_remote.job_logs import LogBuffer
props=bpy.context.scene.vipe_remote
props.job_id='00000000-0000-4000-8000-000000000099'
assert props.echo_job_logs and props.follow_job_log
buffer=LogBuffer();buffer.echo=True;buffer.append('first live line\n')
events=queue.Queue();events.put(('progress',('Tracking camera',.2)))
addon._log_buffers[props.job_id]=buffer
addon._active=dict(scene=bpy.context.scene,events=events,detach=threading.Event(),mode='resume',log_buffer=buffer,log_revision=0)
assert addon._poll()==.25
text=bpy.data.texts[props.log_text_name]
assert text.as_string()=='first live line\n'
assert props.status=='Tracking camera'
window=bpy.context.window_manager.windows[0]
area=next(a for a in window.screen.areas if a.type=='VIEW_3D')
region=next(r for r in area.regions if r.type=='WINDOW')
with bpy.context.temp_override(window=window,area=area,region=region):
 assert bpy.ops.vipe_remote.open_log()=={'FINISHED'}
count=len(window.screen.areas)
area=next(a for a in window.screen.areas if a.type=='VIEW_3D')
region=next(r for r in area.regions if r.type=='WINDOW')
with bpy.context.temp_override(window=window,area=area,region=region):
 assert bpy.ops.vipe_remote.open_log()=={'FINISHED'}
assert len(window.screen.areas)==count
assert any(a.type=='TEXT_EDITOR' and a.spaces.active.text==text for a in window.screen.areas)
props.follow_job_log=False
buffer.append('last live line\n')
addon._poll()
assert 'last live line' not in text.as_string()
events.put(('finished',None));assert addon._poll() is None
assert addon._active is None
props.follow_job_log=True
assert text.as_string()=='first live line\nlast live line\n'
addon._before_load(None)
assert not addon._log_buffers
# Exercise the actual network-thread -> display-buffer -> main-thread path.
import contextlib, io, tempfile, time
original_client=addon.Client
first_line=threading.Event();finish_job=threading.Event()
class ScriptedClient:
 def __init__(self,*args):pass
 def wait(self,identifier,progress,cancel,detach,log=None):
  log('threaded first line\n');progress('Tracking test job',.4);first_line.set()
  if not finish_job.wait(5):raise RuntimeError('Test failed to release worker')
  log('threaded final traceback\n')
  raise RuntimeError('Expected test worker failure')
addon.Client=ScriptedClient
captured=io.StringIO()
try:
 with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(captured):
  addon._launch(bpy.context,'resume',dict(job_id=props.job_id,folder=folder))
  assert first_line.wait(5)
  addon._poll()
  assert props.busy and props.state=='RUNNING'
  assert 'threaded first line' in bpy.data.texts[props.log_text_name].as_string()
  finish_job.set()
  deadline=time.monotonic()+5
  while addon._active is not None:
   addon._poll();time.sleep(.02)
   assert time.monotonic()<deadline
  assert props.state=='ERROR'
  assert 'threaded final traceback' in bpy.data.texts[props.log_text_name].as_string()
  assert 'Expected test worker failure' in props.status
 assert 'threaded first line' in captured.getvalue() and 'threaded final traceback' in captured.getvalue()
finally:
 finish_job.set();addon.Client=original_client;addon._before_load(None)
print(json.dumps(dict(state='PASS',blender=bpy.app.version_string,live_updates=True,split_log_editor=True,no_duplicate_editor=True,pause_and_resume_after_completion=True,default_terminal_echo=True,threaded_output_before_job_completion=True,terminal_echo_and_final_traceback=True)))
