"""Run: blender -b --factory-startup --python-exit-code 1 --python tests/blender_import.py -- result.zip [source.mp4]"""
from pathlib import Path
import sys
import bpy,numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vipe_remote.bundle import load_bundle
from vipe_remote.importer import import_bundle
args=sys.argv[sys.argv.index('--')+1:]
m,c,cloud=load_bundle(args[0])
collection,created=import_bundle(args[0],source_path=args[1] if len(args)>1 else '',start_frame=1001)
assert created
assert bpy.context.scene.frame_end==1000+len(c['c2w'])
assert import_bundle(args[0])[1] is False
errors=[]
for item in cloud.get('environment',[]):
 v,d=item['view'],item['arrays']
 root=next(o for o in collection.objects if o.get('vipe_role')=='alignment')
 bpy.context.scene.frame_set(1001+v['frame_index']);bpy.context.view_layer.update()
 for j in np.linspace(0,len(d['points'])-1,min(30,len(d['points'])),dtype=int):
  uv=world_to_camera_view(bpy.context.scene,bpy.context.scene.camera,root.matrix_world@Vector(d['points'][j]))
  errors.append(np.linalg.norm(np.array([uv.x*v['width'],(1-uv.y)*v['height']])-d['pixels'][j]))
assert not errors or max(errors)<.05,errors
print('PASS: frames',len(c['c2w']),'dense views',len(cloud.get('environment',[])),'maximum dense projection error',max(errors,default=0))
