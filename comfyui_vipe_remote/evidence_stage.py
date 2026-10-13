"""Reproduce the study's independent tracks and triangulate through a frozen camera."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
from worker import digest, write_status, probe_video
from camera_evidence.cotracker_tracks import (
    CoTrackerTrackConfig, generate_cotracker_evidence,
    COTRACKER_SCALED_OFFLINE_SHA256, PINNED_COTRACKER_COMMIT,
)
from camera_evidence.tapir_tracks import DEEPLABV3_MOBILENET_V3_LARGE_SHA256
from camera_evidence.cameras import CameraSequence
from camera_evidence.evaluate import EvaluationConfig, _load_windows
from camera_evidence.audit import AuditConfig, _collect_landmarks, _write_landmark_npz


def run(request_path, config_path):
    started = time.monotonic()
    request = json.loads(request_path.read_text())
    cfg = json.loads(config_path.read_text())
    output, work = Path(request['output']), request_path.parent
    manifest = json.loads((output / 'result.json').read_text())
    source = Path(request['source'])
    camera_path = output / 'camera.npz'
    frozen_digest = digest(camera_path)
    if digest(source) != request['source_sha256']:
        raise ValueError('Source changed before independent tracking')
    recipe = CoTrackerTrackConfig()
    cache = Path(cfg['evidence_cache_directory']) / request['source_sha256']
    reused = (cache / 'MANIFEST.json').is_file()
    write_status(work, 'running', 'Loading independent CoTracker evidence' if reused else 'Generating independent CoTracker tracks', 70)
    if not reused:
        import torch
        import cv2
        torch.set_num_threads(cfg.get('cpu_threads', 2))
        cv2.setNumThreads(cfg.get('cpu_threads', 2))
        torch.cuda.set_per_process_memory_fraction(cfg.get('gpu_memory_fraction', .9))
        info = probe_video(source, cfg['ffprobe'])
        if int(info['nb_frames']) > cfg.get('evidence_max_frames', 1500):
            raise ValueError('Independent review currently requires a source clip of at most 1500 frames; trim the source video first.')
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(cache.name + '.building-' + uuid.uuid4().hex)
        generate_cotracker_evidence(source_video=source, output_dir=temporary,
            cotracker_repository=Path(cfg['cotracker_repository']),
            checkpoint=Path(cfg['cotracker_checkpoint']), config=recipe, device='cuda')
        temporary.rename(cache)
    evidence = json.loads((cache / 'MANIFEST.json').read_text())
    if (evidence.get('candidate_camera_inputs_used') is not False
        or evidence['source']['sha256'] != request['source_sha256']
        or evidence['config'] != asdict(recipe)
        or evidence['tracker']['cotracker_commit'] != PINNED_COTRACKER_COMMIT
        or evidence['tracker']['checkpoint_sha256'] != COTRACKER_SCALED_OFFLINE_SHA256
        or evidence['semantic_static_mask']['checkpoint_sha256'] != DEEPLABV3_MOBILENET_V3_LARGE_SHA256):
        raise ValueError('Cached evidence does not match the original independent tracking recipe')
    for key, expected in [('native_width', manifest['source']['width']), ('native_height', manifest['source']['height']), ('frame_count', manifest['source']['frame_count'])]:
        if evidence['source'][key] != expected:
            raise ValueError('Independent evidence dimensions or source frame count differ')
    # Keep the source-frame indices for trimmed solves. The full-source evidence
    # cache is camera-free and reusable across different selections and solves.
    with np.load(camera_path, allow_pickle=False) as arrays:
        camera = CameraSequence(candidate_id=request['job_id'],
            frame_indices=arrays['source_frame_indices'], c2w=arrays['c2w'],
            intrinsics=arrays['intrinsics'], plate_width=manifest['source']['width'],
            plate_height=manifest['source']['height'], adapter_metadata={'type':'frozen_comfy_bundle'})
    write_status(work, 'running', 'Triangulating independent tracks through the frozen camera', 85)
    _, windows = _load_windows(cache, 'routing')
    evaluation_config, audit_config = EvaluationConfig(), AuditConfig()
    landmarks = _collect_landmarks(camera=camera, windows=windows,
        native_pixels_per_evaluation_pixel=max(evidence['source']['evaluation_to_native_scale_x'], evidence['source']['evaluation_to_native_scale_y']),
        evaluation_config=evaluation_config, audit_config=audit_config)
    destination = output / 'audit_landmarks.npz'
    _write_landmark_npz(destination, landmarks)
    if digest(camera_path) != frozen_digest:
        raise RuntimeError('Independent review unexpectedly changed the solved camera')
    receipt = {'method':'study_schema_8_cotracker_triangulated_landmarks',
        'landmark_count':len(landmarks), 'file':destination.name,
        'evidence_reused':reused, 'evidence_source_sha256':request['source_sha256'],
        'evidence_manifest_sha256':digest(cache/'MANIFEST.json'),
        'evidence_frame_scope':'full original source; triangulation uses selected camera frames',
        'evidence_recipe':asdict(recipe), 'tracker':evidence['tracker'],
        'semantic_mask':evidence['semantic_static_mask'],
        'evaluation_config':asdict(evaluation_config), 'audit_config':asdict(audit_config),
        'partition':'routing', 'camera_modified':False,
        'frozen_camera_sha256':frozen_digest, 'elapsed_seconds':time.monotonic()-started,
        'status':'ready_for_witness_selection' if landmarks else 'no_supported_landmarks',
        'near_depth_quantile':.15, 'target_review_extent_px':80.0,
        'review_resolution':[1280, round(1280*manifest['source']['height']/manifest['source']['width'])],
        'quality_status':'human_review_required'}
    manifest['version'] = 2
    manifest['witnesses'] = receipt
    manifest['files'][destination.name] = {'sha256':digest(destination),'bytes':destination.stat().st_size}
    (output/'result.json').write_text(json.dumps(manifest,indent=2))
    (work/'INDEPENDENT_WITNESSES.json').write_text(json.dumps(receipt,indent=2))
    print('INDEPENDENT_WITNESSES', json.dumps({'landmarks':len(landmarks),'cached':reused,'seconds':receipt['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    try:
        run(args.request,args.config)
    except Exception as exc:
        write_status(args.request.parent, 'failed', str(exc)[-1500:], 0)
        raise
