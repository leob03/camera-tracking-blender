"""Exercise the same transport used by Blender, without a DCC."""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vipe_remote.client import Client
p=argparse.ArgumentParser()
p.add_argument('video',type=Path)
p.add_argument('--url',default='http://127.0.0.1:8189')
p.add_argument('--output',type=Path,default=Path('artifacts'))
p.add_argument('--first',type=int,default=1)
p.add_argument('--last',type=int,default=0)
p.add_argument('--max-dimension',type=int,default=1280)
p.add_argument('--no-witnesses',action='store_true')
p.add_argument('--dense',action='store_true')
a=p.parse_args()
client=Client(a.url)
identifier=str(uuid.uuid4());folder=a.output/identifier;folder.mkdir(parents=True)
settings=dict(frame_start=a.first,frame_end=a.last,max_dimension=a.max_dimension,masking=True,mask_phrases='person,animal,vehicle,ball,balloon,gun,pet,car,bus',lens_mode='auto',focal_mm=35.,sensor_mm=36.,max_points=30000,review_witnesses=not a.no_witnesses,dense_environment=a.dense)
record=dict(job_id=identifier,server=a.url,source=str(a.video.resolve()),folder=str(folder.resolve()),settings=settings,started=time.time())
receipt=folder/'job.json'
def save():receipt.write_text(json.dumps(record,indent=2))
last=[None]
def progress(message,value):
 if message!=last[0]:print(message,flush=True);last[0]=message
save();print('Job',identifier,flush=True)
try:
 upload=client.upload(a.video,progress)
 record['source_sha256']=upload['sha256'];save()
 workflow=client.submit(upload,a.video.name,settings,identifier)
 (folder/'workflow_api.json').write_text(json.dumps(workflow,indent=2))
 item=client.wait(identifier,progress)
 result=client.download(item,folder/'result.zip',progress)
 record.update(state='COMPLETE',result=str(result.resolve()),elapsed_seconds=time.time()-record['started']);save()
 print('Bundle:',result)
except KeyboardInterrupt:
 client.cancel(identifier)
 record['state']='CANCELLATION_REQUESTED';save()
 raise
except Exception as exc:
 record.update(state='ERROR',error=str(exc));save()
 raise
