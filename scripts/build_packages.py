"""Build the two installable packages without collecting caches or model files."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    'vipe_remote': ['__init__.py', 'addon.py', 'bundle.py', 'client.py', 'importer.py', 'artist_witnesses.py', 'environment.py', 'assets/axis_shaped_cross.glb'],
    'comfyui_vipe_remote': ['__init__.py', 'worker.py', 'evidence_stage.py', 'dense_stage.py', 'web/vipe.js', 'camera_evidence/__init__.py', 'camera_evidence/audit.py', 'camera_evidence/evaluate.py', 'camera_evidence/cameras.py', 'camera_evidence/tapir_tracks.py', 'camera_evidence/cotracker_tracks.py', 'camera_evidence/SOURCE_RECEIPT.json'],
}

def build():
    output = ROOT / 'dist'
    output.mkdir(exist_ok=True)
    receipt = {}
    for package, names in PACKAGES.items():
        version = '0.3.0' if package == 'comfyui_vipe_remote' else '0.4.0'
        destination = output / (package + '-' + version + '.zip')
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                source = ROOT / package / name
                if source.suffix == '.py':
                    compile(source.read_text(encoding='utf-8'), str(source), 'exec')
                archive.write(source, package + '/' + name)
            archive.write(ROOT / 'README.md', package + '/README.md')
            archive.write(ROOT / 'THIRD_PARTY.md', package + '/THIRD_PARTY.md')
            for doc in ['BLENDER.md', 'WORKER_SETUP.md', 'BRIDGE_HANDOFF.md', 'OFFLINE.md', 'BUNDLE.md', 'VALIDATION.md']:
                archive.write(ROOT / 'docs' / doc, package + '/docs/' + doc)
            for receipt_name in ['offline-dependencies.json', 'worker-runtime.json', 'benchmarks.json', 'validation.json']:
                archive.write(ROOT / 'provenance' / receipt_name, package + '/provenance/' + receipt_name)
            if package == 'comfyui_vipe_remote':
                archive.write(ROOT / 'worker-config.example.json', package + '/worker-config.example.json')
                archive.write(ROOT / 'workflows/ViPE Camera and Native Cloud.json', package + '/workflows/ViPE Camera and Native Cloud.json')
        receipt[destination.name] = {'bytes': destination.stat().st_size, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
    (output / 'PACKAGES.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    build()
