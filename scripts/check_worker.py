"""Bounded preflight for an already-installed reviewed worker environment."""
import argparse, importlib, json, os, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
c=json.loads(a.config.read_text());sys.path.insert(0,c['vipe_root'])
for name in ['vipe_root','vipe_python','ffmpeg','ffprobe','torch_home','hf_home']:
 if not Path(c[name]).exists():raise SystemExit('Missing '+name+': '+c[name])
for name in ['numpy','torch','cv2','scipy','PIL','zstandard','hydra','omegaconf']:
 importlib.import_module(name)
from vipe.utils.io import save_slam_stage_artifacts
from vipe.streams.disk_store import frame_store, DiskArrayStore
from vipe.slam.interface import SLAMOutput
from vipe.pipeline.processors import AdaptiveDepthProcessor
for name in ['initial_intrinsics','pre_global_ba','post_global_ba']:
 if name not in SLAMOutput.__dataclass_fields__:raise SystemExit('ViPE is missing '+name)
import torch
if not torch.cuda.is_available():raise SystemExit('CUDA unavailable in worker Python')
print('Core worker interfaces available; CUDA:',torch.cuda.get_device_name(0))
for name in ['cotracker_repository','cotracker_checkpoint']:
 print(name, 'available' if c.get(name) and Path(c[name]).exists() else 'not configured (required for witnesses)')
print('Model completeness and offline inference still require real job tests.')
