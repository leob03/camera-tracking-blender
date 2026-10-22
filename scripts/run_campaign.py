"""Run a sequential A/B plan against one worker and write a JSON + Markdown report.

Plan: JSON list of {"name", "video", "settings": {...}} or {"name", "review_of": "<earlier name>", "extend": {...}}.
Jobs run strictly one after another. Requires NumPy (use the worker-side Comfy Python).
"""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vipe_remote.bundle import load_bundle
from vipe_remote.client import Client
from vipe_remote.lens_math import review_errors
from submit_video import run_job

DEFAULTS = dict(frame_start=1, frame_end=0, max_dimension=1280, masking=True, mask_phrases='person,animal,vehicle,ball,balloon,gun,pet,car,bus', lens_mode='auto', focal_mm=35., sensor_mm=36., max_points=30000, review_witnesses=False, dense_environment=False, dense_mode='fast', lens_distortion=False)


def metrics(bundle):
    manifest, camera, cloud = load_bundle(bundle)
    out = dict(frames=manifest['selection']['count'], focal_px=float(camera['intrinsics'][0, 0]), points=manifest['cloud']['point_count'], timings=manifest.get('timings_seconds', {}), camera_model=manifest['camera']['model'])
    lens = manifest['camera'].get('lens_distortion')
    if lens:
        out.update(xi=lens['xi'], distorted_fx=lens['fx'])
    if cloud.get('audit') is not None:
        frames, errors = review_errors(camera, cloud['audit'], manifest['selection']['first_source_frame'])
        per_frame = [float(np.median(errors[frames == f])) for f in np.unique(frames)]
        out.update(landmarks=int(len(cloud['audit']['landmark_ids'])), observations=int(len(errors)), review_px_median=float(np.median(errors)), review_px_p90=float(np.percentile(errors, 90)), review_px_rms=float(np.sqrt(np.mean(errors**2))), worst_frame_median_px=float(max(per_frame)) if per_frame else None)
    if cloud.get('environment'):
        out.update(dense_points=manifest['environment']['point_count'], dense_mode=manifest['environment'].get('mode', 'video'), dense_seconds=manifest['environment'].get('elapsed_seconds'))
    with zipfile.ZipFile(bundle) as archive:
        out['bundle_bytes'] = sum(x.compress_size for x in archive.infolist())
    return out


def report(results, path):
    lines = ['# ViPE A/B campaign', '', f'Generated {time.strftime("%Y-%m-%d %H:%M")}. Worker wall times exclude upload/download unless noted. Review px = held-out reprojection residuals of independent CoTracker landmarks through the frozen camera, in plate pixels (lower is better; compare only within the same plate).', '',
             '| Job | State | Worker s | Elapsed s | Upload | Focal px | xi | Review median px | p90 px | RMS px | Worst frame px | Dense pts | Notes |', '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        m = r.get('metrics', {})
        t = m.get('timings', {})
        worker = t.get('review_worker_before_packaging') or t.get('worker_before_packaging')
        f = lambda v, d=2: '' if v is None else f'{v:.{d}f}'
        lines.append(f"| {r['name']} | {r.get('state')} | {f(worker,1)} | {f(r.get('elapsed_seconds'),1)} | {'reused' if r.get('upload_reused') else ('sent' if r.get('upload_seconds') is not None else '—')} | {f(m.get('focal_px'),1)} | {f(m.get('xi'),4)} | {f(m.get('review_px_median'),3)} | {f(m.get('review_px_p90'),3)} | {f(m.get('review_px_rms'),3)} | {f(m.get('worst_frame_median_px'),2)} | {m.get('dense_points','')} | {r.get('error','')[:120]} |")
    lines += ['', 'Stage timings (seconds):', '']
    for r in results:
        if r.get('metrics', {}).get('timings'):
            lines.append(f"- **{r['name']}**: " + ', '.join(f'{k} {v:.1f}' for k, v in r['metrics']['timings'].items()))
    path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:8189')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recompute', action='store_true', help='Recompute metrics and the report from saved bundles without running jobs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.recompute:
        results = json.loads((args.output / 'results.json').read_text())
        for entry in results:
            if entry.get('result'):
                entry['metrics'] = metrics(entry['result'])
        (args.output / 'results.json').write_text(json.dumps(results, indent=2))
        report(results, args.output / 'REPORT.md')
        print('Recomputed', args.output / 'REPORT.md')
        raise SystemExit(0)
    client = Client(args.url)
    jobs, results = {}, []
    for step in json.loads(args.plan.read_text()):
        print(f"\n===== {step['name']} =====", flush=True)
        entry = {'name': step['name']}
        try:
            review_of = jobs[step['review_of']]['job_id'] if step.get('review_of') else None
            record = run_job(client, step.get('video'), args.output / 'jobs', None if review_of else {**DEFAULTS, **step.get('settings', {})}, review_of, live_log=False, extend=step.get('extend'))
            entry.update({k: record.get(k) for k in ('job_id', 'state', 'elapsed_seconds', 'upload_reused', 'upload_seconds', 'result', 'stages')})
            entry['metrics'] = metrics(record['result'])
            jobs[step['name']] = record
        except Exception as exc:
            entry.update(state='ERROR', error=str(exc))
            print('FAILED:', exc, flush=True)
        results.append(entry)
        (args.output / 'results.json').write_text(json.dumps(results, indent=2))
        report(results, args.output / 'REPORT.md')
    print('\nReport:', args.output / 'REPORT.md', flush=True)
