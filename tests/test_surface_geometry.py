"""CPU geometry checks (NumPy and OpenCV, no models/GPU)."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'comfyui_vipe_remote'))
from dense_stage import NEAR_SKY_FACTOR, reject_mask, surface_arrays

class SurfaceGeometry(unittest.TestCase):
    def test_projection_and_mask_hole(self):
        depth=np.full((40,60),10,dtype=np.float32)
        excluded=np.zeros(depth.shape,bool);excluded[15:25,25:35]=True
        rgb=np.ones((40,60,3),np.float32)*.3
        k=np.array([50,51,29,19])
        pose=np.eye(4);pose[:3,3]=[4,5,6]
        data=surface_arrays(depth,rgb,excluded,k,pose,2)
        xyz=data['points']-pose[:3,3]
        uv=xyz[:,:2]/xyz[:,2,None]*k[:2]+k[2:]
        np.testing.assert_allclose(uv,data['pixels'],atol=1e-5)
        x,y=data['pixels'].T
        self.assertFalse(np.any((x>=23)&(x<=36)&(y>=13)&(y<=26)))
        for face in data['faces']:
            pixels=data['pixels'][face]
            self.assertLessEqual(np.ptp(pixels[:,0]),2)
            self.assertLessEqual(np.ptp(pixels[:,1]),2)

    def test_depth_discontinuities_and_nonfinite(self):
        depth=np.full((40,60),10,dtype=np.float32);depth[:,30:]=30
        depth[3,4]=np.nan;depth[8,9]=0
        data=surface_arrays(depth,np.zeros((40,60,3)),np.zeros(depth.shape,bool),np.array([50,50,30,20]),np.eye(4),1)
        self.assertTrue(np.isfinite(data['points']).all())
        self.assertTrue(np.all(data['depths']>0))
        for face in data['faces']:
            self.assertEqual(np.ptp(data['depths'][face]),0)
        self.assertFalse(np.any(np.isin(data['pixels'][:,0],[29,30])))

    def test_keep_ceilings_keeps_near_sky_only(self):
        # Static scene at 10, an indoor "sky" ceiling at 9, real sky at the depth model's far limit, one excluded object.
        instance=np.zeros((30,40),np.uint8);depth=np.full(instance.shape,10,np.float32)
        instance[:8]=2;depth[:8]=9                      # ceiling labelled sky
        instance[8:12]=2;depth[8:12]=1000               # real sky
        instance[20:25,5:10]=1;depth[20:25,5:10]=4      # excluded (moving) object
        reject,kept=reject_mask(instance,depth,True)
        self.assertFalse(reject[:8].any())
        self.assertTrue(reject[8:12].all() and reject[20:25,5:10].all())
        self.assertAlmostEqual(kept,8/12)
        reject,kept=reject_mask(instance,depth,False)
        np.testing.assert_array_equal(reject,instance!=0);self.assertEqual(kept,0.0)
        depth[:8]=10*NEAR_SKY_FACTOR*1.01               # just beyond the limit: dropped
        self.assertTrue(reject_mask(instance,depth,True)[0][:8].all())

if __name__=='__main__':unittest.main()
