"""Exercise the same transport used by Blender, without a DCC."""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vipe_remote.client import Client


def run_job(client,video,folder,settings=None,review_of=None,live_log=True,quiet=False):
 """Submit one job (or a review of an existing job), wait, download; return the receipt."""
 identifier=str(uuid.uuid4());folder=Path(folder)/identifier;folder.mkdir(parents=True)
 record=dict(job_id=identifier,server=client.base_url,source=str(Path(video).resolve()) if video else '',folder=str(folder.resolve()),settings=settings,review_of=review_of,started=time.time(),stages={})
 receipt=folder/'job.json'
 def save():receipt.write_text(json.dumps(record,indent=2))
 last=[None]
 def progress(message,value):
  base=message.split(' · ')[0]
  if base!=last[0]:
   record['stages'][base]=time.time()-record['started'];last[0]=base
   if not quiet:print(base,flush=True)
 save();print('Job',identifier,flush=True)
 try:
  if review_of:
   workflow=client.submit_review(review_of,identifier)
  else:
   upload=client.upload(video,progress)
   record.update(source_sha256=upload['sha256'],upload_reused=upload.get('reused',False),upload_seconds=time.time()-record['started']);save()
   workflow=client.submit(upload,Path(video).name,settings,identifier)
  (folder/'workflow_api.json').write_text(json.dumps(workflow,indent=2))
  item=client.wait(identifier,progress,log=lambda text: print(text,end='',flush=True) if live_log else None)
  result=client.download(item,folder/'result.zip',progress)
  record.update(state='COMPLETE',result=str(result.resolve()),elapsed_seconds=time.time()-record['started']);save()
  print('Bundle:',result,flush=True)
 except KeyboardInterrupt:
  client.cancel(identifier)
  record['state']='CANCELLATION_REQUESTED';save()
  raise
 except Exception as exc:
  record.update(state='ERROR',error=str(exc),elapsed_seconds=time.time()-record['started']);save()
  raise
 return record


if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('video',type=Path,nargs='?')
 p.add_argument('--url',default='http://127.0.0.1:8189')
 p.add_argument('--output',type=Path,default=Path('artifacts'))
 p.add_argument('--first',type=int,default=1)
 p.add_argument('--last',type=int,default=0)
 p.add_argument('--max-dimension',type=int,default=1280)
 p.add_argument('--witnesses',action='store_true',help='Run review witnesses in the same job')
 p.add_argument('--no-witnesses',action='store_true',help='Accepted for compatibility; witnesses are off by default')
 p.add_argument('--dense',action='store_true')
 p.add_argument('--dense-mode',choices=['fast','video'],default='fast')
 p.add_argument('--distortion',action='store_true',help='Estimate unified (MEI) lens distortion')
 p.add_argument('--review-of',help='Add review witnesses to this existing job ID instead of submitting a plate')
 p.add_argument('--no-live-log',action='store_true',help='Hide worker stdout/stderr while still showing stage progress')
 a=p.parse_args()
 if not a.review_of and not a.video:p.error('a video is required unless --review-of is given')
 settings=None if a.review_of else dict(frame_start=a.first,frame_end=a.last,max_dimension=a.max_dimension,masking=True,mask_phrases='person,animal,vehicle,ball,balloon,gun,pet,car,bus',lens_mode='auto',focal_mm=35.,sensor_mm=36.,max_points=30000,review_witnesses=a.witnesses and not a.no_witnesses,dense_environment=a.dense,dense_mode=a.dense_mode,lens_distortion=a.distortion)
 run_job(Client(a.url),a.video,a.output,settings,a.review_of,not a.no_live_log)
