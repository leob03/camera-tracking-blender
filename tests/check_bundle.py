"""Validate a downloaded bundle without importing Blender (requires NumPy)."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vipe_remote.bundle import load_bundle
p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);a=p.parse_args()
m,c,cloud=load_bundle(a.bundle)
print(json.dumps(dict(version=m['version'],frames=len(c['c2w']),native_points=len(cloud['xyz']),witnesses='audit' in cloud,dense_views=len(cloud.get('environment',[]))),indent=2))
