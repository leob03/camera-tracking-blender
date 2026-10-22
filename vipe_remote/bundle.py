"""Read the data-only camera bundle without extracting or executing server files."""
import hashlib
import io
import json
import math
import zipfile

import numpy as np


def load_bundle(path, expected_source_sha256=None):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = {x.filename for x in members}
        if len(members) != len(names) or not {"result.json", "camera.npz", "cloud.npz"}.issubset(names):
            raise ValueError("Camera bundle has unexpected or duplicate files.")
        if sum(x.file_size for x in members) > 512 * 1024**2:
            raise ValueError("Camera bundle is too large to import.")
        if archive.getinfo("result.json").file_size > 1024**2:
            raise ValueError("Camera manifest is too large.")
        manifest = json.loads(archive.read("result.json"))
        if manifest.get("schema") != "vipe-camera-bundle" or manifest.get("version") not in (1, 2, 3):
            raise ValueError("Unsupported camera bundle version.")
        if manifest["camera"]["convention"] != "opencv_camera_to_world" or manifest["camera"]["model"] not in ("PINHOLE", "PINHOLE_UNDISTORTED"):
            raise ValueError("This importer requires an OpenCV camera-to-world PINHOLE camera.")
        if manifest["camera"]["model"] == "PINHOLE_UNDISTORTED":
            lens = manifest["camera"].get("lens_distortion") or {}
            values = [lens.get(k) for k in ("fx", "fy", "cx", "cy", "xi", "undistorted_focal_px")]
            if lens.get("model") != "mei" or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) or min(values[0], values[1], values[5]) <= 0 or not -0.5 < lens["xi"] < 2:
                raise ValueError("Invalid lens distortion description.")
        if expected_source_sha256 and manifest["source"]["sha256"] != expected_source_sha256:
            raise ValueError("This camera was solved from a different video.")
        expected_names = {"result.json", "camera.npz", "cloud.npz"}
        if manifest['version'] == 2 or manifest.get('witnesses'):
            expected_names.add('audit_landmarks.npz')
        views = manifest.get('environment', {}).get('views', [])
        if manifest['version'] == 3:
            if not 1 <= len(views) <= 12:
                raise ValueError('Dense environment must contain one to twelve views.')
            used_frames = set()
            for view in views:
                index = view['frame_index']
                if type(index) is not int or not 0 <= index < manifest['selection']['count'] or index in used_frames:
                    raise ValueError('Invalid dense view frame.')
                used_frames.add(index)
                if view['file'] != f'environment/view_{index:06d}.npz' or view['texture'] != f'environment/plate_{index:06d}.png':
                    raise ValueError('Unexpected dense environment member name.')
                if not 1 <= view['width'] <= 1280 or not 1 <= view['height'] <= 1280:
                    raise ValueError('Dense texture exceeds the supported dimensions.')
                expected_names.update((view['file'], view['texture']))
        elif views:
            raise ValueError('Dense environment requires bundle version 3.')
        if set(manifest['files']) != expected_names - {'result.json'}:
            raise ValueError('Manifest file list does not match the declared outputs.')
        if names != expected_names:
            raise ValueError('Bundle version and independent landmark members disagree.')
        arrays = {}
        expanded_bytes = 0
        for filename in sorted(expected_names - {"result.json"}):
            raw = archive.read(filename)
            entry = manifest["files"][filename]
            if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"Corrupt bundle member: {filename}")
            if filename.endswith('.png'):
                import struct
                view = next(v for v in views if v['texture'] == filename)
                if (raw[:8] != b'\x89PNG\r\n\x1a\n' or raw[12:16] != b'IHDR'
                    or struct.unpack('>II', raw[16:24]) != (view['width'], view['height'])):
                    raise ValueError('Dense texture dimensions do not match its manifest.')
                arrays[filename] = raw
                continue
            with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                expanded_bytes += sum(x.file_size for x in inner.infolist())
                if expanded_bytes > 512 * 1024**2:
                    raise ValueError("Camera arrays expand beyond the import limit.")
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                arrays[filename] = {name: data[name] for name in data.files}
    camera, cloud = arrays["camera.npz"], arrays["cloud.npz"]
    n = manifest["selection"]["count"]
    poses, intrinsic, indices = camera["c2w"], camera["intrinsics"], camera["source_frame_indices"]
    if poses.shape != (n, 4, 4) or intrinsic.shape != (n, 4) or indices.shape != (n,):
        raise ValueError("Camera arrays do not match the frame count.")
    expected = np.arange(manifest["selection"]["first_source_frame"], manifest["selection"]["last_source_frame"] + 1)
    if not np.array_equal(indices, expected) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Camera frame mapping is incomplete.")
    if not np.isfinite(poses).all() or not np.isfinite(intrinsic).all() or np.any(intrinsic[:, :2] <= 0):
        raise ValueError("Camera contains non-finite values or an invalid focal length.")
    rotation = poses[:, :3, :3]
    if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-3) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-3) or not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Camera transforms are not rigid camera-to-world matrices.")
    ratio = intrinsic[:, 0] / intrinsic[:, 1]
    if not np.allclose(ratio, ratio[0], rtol=1e-6):
        raise ValueError("Animated pixel aspect is not supported by this importer.")
    m = manifest["cloud"]["point_count"]
    if cloud["xyz"].shape != (m, 3) or cloud["rgb"].shape != (m, 3) or cloud["ids"].shape != (m,):
        raise ValueError("Point cloud arrays have invalid shapes.")
    if not np.isfinite(cloud["xyz"]).all() or not np.isfinite(cloud["rgb"]).all():
        raise ValueError("Point cloud contains non-finite values.")
    source = manifest["source"]
    if min(source[k] for k in ["width", "height", "fps_num", "fps_den"]) <= 0:
        raise ValueError("Source dimensions and frame rate must be positive.")
    if 'audit_landmarks.npz' in arrays:
        audit = arrays['audit_landmarks.npz']
        required = {'landmark_ids', 'landmark_xyz', 'landmark_span_frames',
                    'landmark_median_error_evaluation_px', 'observation_offsets',
                    'observation_frames', 'observed_xy', 'observation_roles', 'positive_depth'}
        if not required.issubset(audit):
            raise ValueError('Independent landmark arrays are incomplete.')
        ids = audit['landmark_ids']
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Invalid landmark IDs.')
        a = len(ids)
        offsets = audit['observation_offsets']
        frames = audit['observation_frames']
        if (a > 5000 or len(np.unique(ids)) != a or frames.ndim != 1
            or offsets.shape != (a + 1,) or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != len(frames) or np.any(np.diff(offsets) < 0)):
            raise ValueError('Invalid landmark observation offsets.')
        for key, shape in [('landmark_xyz',(a,3)), ('landmark_span_frames',(a,)),
                           ('landmark_median_error_evaluation_px',(a,)),
                           ('observed_xy',(len(frames),2)), ('observation_roles',(len(frames),)),
                           ('positive_depth',(len(frames),))]:
            if audit[key].shape != shape or not np.isfinite(audit[key]).all():
                raise ValueError(f'Invalid independent landmark array: {key}')
        if (not np.issubdtype(frames.dtype, np.integer) or not np.isin(frames, indices).all()
            or not np.isin(audit['observation_roles'], [0,1]).all()
            or not np.isin(audit['positive_depth'], [0,1]).all()):
            raise ValueError('Invalid independent observation frame or role.')
        cloud['audit'] = audit
    if views:
        environment = []
        total_points = 0
        for view in views:
            data = arrays[view['file']]
            m, f = view['point_count'], view['face_count']
            total_points += m
            if not 0 <= m <= 150000 or not 0 <= f <= 150000 or total_points > 1500000:
                raise ValueError('Dense geometry exceeds import limits.')
            for key, shape in [('points',(m,3)),('colors',(m,3)),('uv',(m,2)),('pixels',(m,2)),('depths',(m,)),('faces',(f,4))]:
                if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
                    raise ValueError('Invalid dense geometry array: '+key)
            faces, pixels = data['faces'], data['pixels']
            if (not np.issubdtype(faces.dtype,np.integer) or np.any(faces<0) or np.any(faces>=m)
                or not np.issubdtype(pixels.dtype,np.integer) or np.any(pixels<0)
                or np.any(pixels[:,0]>=view['width']) or np.any(pixels[:,1]>=view['height'])
                or np.any(data['depths']<=0) or np.any(data['uv']<0) or np.any(data['uv']>1)):
                raise ValueError('Invalid dense geometry indices or depths.')
            idx = view['frame_index']
            k = np.asarray(view['intrinsics'])
            expected_k = intrinsic[idx] * np.array([view['width']/source['width'],view['height']/source['height']]*2)
            if (not np.allclose(view['c2w'],poses[idx],atol=1e-5) or k.shape!=(4,)
                or not np.allclose(k,expected_k,rtol=1e-5,atol=1e-4)
                or view['source_frame_index'] != indices[idx]):
                raise ValueError('Dense view does not match the frozen camera.')
            environment.append(dict(view=view, arrays=data, texture=arrays[view['texture']]))
        if total_points != manifest['environment']['point_count']:
            raise ValueError('Dense point count does not match its manifest.')
        cloud['environment'] = environment
    return manifest, camera, cloud
