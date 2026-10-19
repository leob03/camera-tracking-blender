"""Optional stock post-SLAM depth, frozen-camera surface accumulation."""
import argparse
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from worker import digest, write_status


def capture_masks(pipeline, work):
    """Record exactly the masks consumed by SLAM, with one open archive."""
    import torch
    from vipe.streams.base import ProcessedVideoStream, StreamProcessor
    archive = zipfile.ZipFile(work / 'solve_masks.zip', 'w', zipfile.ZIP_DEFLATED)
    seen = set()
    class RecordMasks(StreamProcessor):
        def __call__(self, index, frame):
            if index not in seen:
                valid = frame.mask if frame.mask is not None else torch.ones(frame.size(), device=frame.device, dtype=torch.bool)
                sky = frame.sky_mask
                excluded = frame.instance != 0 if frame.instance is not None else ~valid
                payload = io.BytesIO()
                np.savez_compressed(payload, valid=valid.cpu().numpy(), sky=sky.cpu().numpy(), excluded=excluded.cpu().numpy())
                archive.writestr(f'{index:06d}.npz', payload.getvalue())
                seen.add(index)
            return frame
    original = pipeline._add_init_processors
    pipeline._add_init_processors = lambda stream: ProcessedVideoStream(original(stream), [RecordMasks()])
    return archive


def surface_arrays(depth, rgb, excluded, intrinsics, pose, stride):
    """Sample static pixels; reject depth edges and connect valid grid neighbors."""
    import cv2
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[yy, xx]
    reject = cv2.dilate(excluded.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    finite = np.isfinite(depth) & (depth > 0)
    safe = np.where(finite, depth, 0).astype(np.float32)
    hi = cv2.dilate(safe, np.ones((3, 3), np.uint8))
    lo = cv2.erode(safe, np.ones((3, 3), np.uint8))
    valid = finite[yy, xx] & ~reject[yy, xx] & ((hi-lo)[yy, xx] <= 0.08*z)
    fx, fy, cx, cy = intrinsics
    local = np.stack(((xx-cx)*z/fx, (yy-cy)*z/fy, z), axis=-1)[valid]
    points = local @ pose[:3, :3].T + pose[:3, 3]
    ids = np.full(valid.shape, -1, dtype=np.int32)
    ids[valid] = np.arange(valid.sum())
    faces = np.stack((ids[:-1,:-1], ids[:-1,1:], ids[1:,1:], ids[1:,:-1]), axis=-1).reshape(-1,4)
    depths = np.stack((z[:-1,:-1],z[:-1,1:],z[1:,1:],z[1:,:-1]), axis=-1).reshape(-1,4)
    connected = (faces >= 0).all(axis=1) & np.isfinite(depths).all(axis=1)
    connected &= np.ptp(depths, axis=1) <= 0.08*np.maximum(np.min(depths,axis=1), 1e-8)
    pixels = np.stack((xx,yy),axis=-1)[valid].astype(np.int32)
    return dict(points=points.astype(np.float32), colors=np.clip(rgb[yy,xx][valid],0,1).astype(np.float32),
                faces=faces[connected], uv=np.stack(((pixels[:,0]+.5)/w, 1-(pixels[:,1]+.5)/h),axis=-1).astype(np.float32),
                pixels=pixels, depths=z[valid].astype(np.float32))


class SampledViewDepth:
    """Per-view depth for sampled frames, following AdaptiveDepthProcessor without the video model."""
    def __init__(self, slam):
        from vipe.priors.depth.priorda import PriorDAModel
        self.slam = slam
        self.prompt_model = PriorDAModel()
        self.metric_model = None
        self.min_uv = None

    def uv_score(self, depth, patch_count=10):
        h, w = depth.size(0)//patch_count, depth.size(1)//patch_count
        crop = (depth > 0)[:h*patch_count, :w*patch_count].reshape(patch_count, h, patch_count, w)
        return crop.any(dim=(1, 3)).float().mean().item()

    def project(self, index, frame):
        return self.slam.slam_map.project_map(index, 0, frame.size(), frame.intrinsics, self.slam.get_view_trajectory(0)[index], frame.camera_type, infill=False)

    def __call__(self, index, frame):
        from vipe.priors.depth import DepthEstimationInput, make_depth_model
        frame = frame.cuda()
        if self.min_uv is None:
            # Same map-coverage rule as the stock processor: every tenth solved frame.
            self.min_uv = min(self.uv_score(self.project(i, frame)) for i in range(0, self.slam.trajectory.shape[0], 10))
            print(f'[ViPE dense] Minimum UV score {self.min_uv:.3f}; using {"metric depth" if self.min_uv < 0.3 else "SLAM-prompted PriorDA"}', flush=True)
        if self.min_uv < 0.3:
            if self.metric_model is None:
                self.metric_model = make_depth_model('unidepth-l')
            frame.metric_depth = self.metric_model.estimate(DepthEstimationInput(rgb=frame.rgb.float().cuda(), intrinsics=frame.intrinsics, camera_type=frame.camera_type)).metric_depth
        else:
            prompt = self.project(index, frame)
            if frame.mask is not None:
                prompt = prompt * frame.mask.float()
            frame.metric_depth = self.prompt_model.estimate(DepthEstimationInput(rgb=frame.rgb.float().cuda(), prompt_metric_depth=prompt)).metric_depth
        return frame


def run(request_path, config_path):
    import torch
    from PIL import Image
    from vipe.ext.lietorch import SE3
    from vipe.slam.interface import SLAMMap, SLAMOutput
    from vipe.streams.raw_mp4_stream import RawMp4Stream
    from vipe.streams.base import ProcessedVideoStream, StreamProcessor
    from vipe.utils.cameras import CameraType
    from vipe.pipeline.processors import AdaptiveDepthProcessor
    from vipe.utils.logging import configure_logging
    started = time.monotonic()
    request = json.loads(request_path.read_text())
    cfg = json.loads(config_path.read_text())
    work, output = request_path.parent, Path(request['output'])
    manifest = json.loads((output/'result.json').read_text())
    before = digest(output/'camera.npz')
    torch.set_num_threads(cfg.get('cpu_threads',2))
    torch.cuda.set_per_process_memory_fraction(cfg.get('gpu_memory_fraction',.9))
    configure_logging()
    write_status(work,'running','Computing optional dense video depth',85)
    with np.load(work/'depth_context.npz',allow_pickle=False) as data:
        vectors = torch.from_numpy(data['poses']).cuda()
        intrinsic = torch.from_numpy(data['intrinsics']).cuda()
    payload = torch.load(work/'native_slam_map.pt',map_location='cpu',weights_only=False)
    slam_map = SLAMMap(**{k:payload[k].cuda() for k in ['dense_disp_xyz','dense_disp_rgb','dense_disp_packinfo']},dense_disp_frame_inds=payload['dense_disp_frame_inds'])
    slam = SLAMOutput(trajectory=SE3(vectors), intrinsics=intrinsic[None], rig=SE3(torch.tensor([[0,0,0,0,0,0,1]],device='cuda',dtype=torch.float32)),slam_map=slam_map)
    count = manifest['selection']['count']
    with np.load(output/'camera.npz',allow_pickle=False) as data:
        exported = data['intrinsics']
    sampled = set(np.linspace(0,count-1,min(count,12),dtype=int).tolist())
    env_dir = output/'environment'
    env_dir.mkdir(exist_ok=True)
    views = []
    with zipfile.ZipFile(work/'solve_masks.zip') as masks:
        class FrozenInputs(StreamProcessor):
            def __call__(self,index,frame):
                frame.pose = slam.get_view_trajectory(0)[index]
                frame.intrinsics = intrinsic.clone()
                frame.camera_type = CameraType.PINHOLE
                with np.load(io.BytesIO(masks.read(f'{index:06d}.npz')),allow_pickle=False) as data:
                    frame.mask = torch.from_numpy(data['valid']).to(frame.device)
                    sky = torch.from_numpy(data['sky']).to(frame.device)
                    excluded = torch.from_numpy(data['excluded']).to(frame.device)
                frame.instance = excluded.to(torch.uint8)
                frame.instance[sky] = 2
                frame.instance_phrases = {1:'excluded',2:'sky'}
                h,w=frame.size()
                factor=min(1.,1280/max(h,w))
                if factor<1:frame=frame.resize((max(2,round(h*factor/2)*2),max(2,round(w*factor/2)*2)))
                return frame
        fast = manifest['settings'].get('dense_mode', 'fast') == 'fast'
        frames = RawMp4Stream(work/'plate.mp4')
        if fast:
            # Only the sampled views are kept, so depth is computed only for them: PriorDA
            # completes the projected native SLAM map (metric, in the solve's own scale).
            stream = (FrozenInputs()(index, frame) for index, frame in enumerate(frames) if index in sampled)
            depth = SampledViewDepth(slam)
        else:
            stream = ProcessedVideoStream(frames, [FrozenInputs(), AdaptiveDepthProcessor(slam)])
        done = 0
        for position, frame in enumerate(stream):
            index = sorted(sampled)[position] if fast else position
            if index not in sampled:continue
            if fast: frame = depth(index, frame)
            h,w=frame.size()
            stride=max(1,round(max(h,w)/342))
            # Unproject with the exported camera (scaled to this view) so geometry matches the import exactly.
            k=exported[index]*np.array([w/manifest['source']['width'],h/manifest['source']['height']]*2)
            pose=frame.pose.matrix().detach().cpu().numpy()
            rgb=frame.rgb.detach().cpu().numpy()
            arrays=surface_arrays(frame.metric_depth.detach().cpu().numpy(),rgb,frame.instance.detach().cpu().numpy()!=0,k,pose,stride)
            data_name=f'environment/view_{index:06d}.npz'
            image_name=f'environment/plate_{index:06d}.png'
            np.savez_compressed(output/data_name,**arrays)
            Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(output/image_name)
            for name in [data_name,image_name]:manifest['files'][name]={'sha256':digest(output/name),'bytes':(output/name).stat().st_size}
            views.append(dict(frame_index=index,source_frame_index=int(index+manifest['selection']['first_source_frame']),file=data_name,texture=image_name,width=w,height=h,stride=stride,intrinsics=k.tolist(),c2w=pose.tolist(),point_count=len(arrays['points']),face_count=len(arrays['faces'])))
            done += 1
            write_status(work,'running',f'Dense view {done}/{len(sampled)}',90)
    if digest(output/'camera.npz') != before:raise RuntimeError('Dense reconstruction changed the frozen camera')
    manifest['version']=3
    manifest['processing']['dense_depth_computed']=True
    manifest['environment']=dict(kind='unmerged_static_depth_surfaces',model='sampled_priorda_slam_prompt' if fast else 'adaptive_unidepth-l_svda',mode='fast' if fast else 'video',views=views,point_count=sum(v['point_count'] for v in views),camera_unchanged=True,camera_sha256=before,depth_long_edge_limit=1280,filtering='Actual solve TrackAnything exclusions including sky, 5x5 dilation; finite positive depth; 8% depth-edge rejection',elapsed_seconds=time.monotonic()-started)
    (output/'result.json').write_text(json.dumps(manifest,indent=2))
    (work/'DENSE_ENVIRONMENT.json').write_text(json.dumps(manifest['environment'],indent=2))
    print('DENSE_ENVIRONMENT_COMPLETE',manifest['environment']['point_count'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.request,args.config)
    except Exception as exc:
        write_status(args.request.parent,'failed',str(exc)[-1500:],0)
        raise
