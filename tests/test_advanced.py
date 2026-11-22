from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'AstroCam-Stack'))
import numpy as np
import astrocam_advanced as a

rng=np.random.default_rng(120)
# Known projective motion with independent centroid noise. A similarity-only
# model cannot describe this transformation across the entire field.
shape=(1200,900)
src_points=np.column_stack((rng.uniform(60,840,150),rng.uniform(60,1140,150)))
H=np.array([[1.0005,-.001,1.5],[.001,1.0003,-1],[2e-6,-3e-6,1.]])
target=a.project(src_points,H)+rng.normal(0,.06,(150,2))
source={'points':src_points,'flux':rng.uniform(100,1000,150),'width':np.full(150,7.),'shape':shape}
ref={'points':target,'flux':source['flux']*1.12,'width':np.full(150,7.),'shape':shape}
coarse=np.eye(3);coarse[:2,2]=[1.5,-1]
fit,stats=a.fit_registration(source,ref,coarse,15)
print('Projective registration:',stats)
assert stats['model']=='homography'
assert stats['validation_selected_rms_px']<.2
assert stats['validation_selected_rms_px']<stats['validation_previous_rms_px']*.3
assert abs(stats['photometric_gain']-1.12)<.001
assert np.median(np.linalg.norm(a.project(src_points,fit)-a.project(src_points,H),axis=1))<.1

# Weighted averaging must retain a real point-source-shaped signal while
# rejecting an isolated, very bright corrupted exposure at those same pixels.
n=18
yy,xx=np.mgrid[:20,:24]
truth=1000+800*np.exp(-((xx-12)**2+(yy-10)**2)/(2*2**2))
truth=np.repeat(truth[...,None],3,axis=2).astype(np.float32)
block=truth+rng.normal(0,10,(n,20,24,3)).astype(np.float32)
block[0,8:12,10:14]=60000
weights=np.linspace(.6,1.4,n).astype(np.float32)
for method in ['mad','winsorized']:
    result,rejected,neff=a.weighted_integrate(block,weights,np.array([5,5,5],np.float32),method)
    print(method,'median error',np.median(np.abs(result-truth)),'peak relative',result[10,12].mean()/truth[10,12].mean())
    assert np.median(np.abs(result-truth))<3
    assert abs(result[10,12].mean()/truth[10,12].mean()-1)<.01
    assert rejected[9,11].min()>=1
    assert np.all((neff>0)&(neff<=n+.001))
# Constant pixels have zero empirical MAD; the floor must still yield finite
# exactly constant results, including a single extreme outlier.
constant=np.full((n,3,4,3),.125,np.float32);constant[3,1,1]=2.
for method in ['mad','winsorized']:
    result,_,_=a.weighted_integrate(constant,np.ones(n,np.float32),np.full(3,.001,np.float32),method)
    assert np.allclose(result,.125,atol=1e-6)

# Additive plane matching removes a known gradient difference while preserving
# shared extended structure (a proxy for a galaxy halo) and stellar gain.
h,w=480,640
yy,xx=np.mgrid[:h,:w];halo=.02*np.exp(-((xx-w/2)**2+(yy-h/2)**2)/(2*35**2))
base=np.repeat((.1+halo)[...,None],3,axis=2).astype(np.float32)
gain=1.13
plane=.007+.003*(xx/(w-1)*2-1)-.004*(yy/(h-1)*2-1)
frame=(base-np.repeat(plane[...,None],3,axis=2))/gain
small=a.cv2.resize(base,(w//6,h//6),interpolation=a.cv2.INTER_AREA)
coef=a.background_plane_difference(small,frame.astype(np.float32),gain)
print('Recovered additive planes:',coef)
assert np.max(np.abs(coef-np.array([.007,.003,-.004])))<.0006
print('ADVANCED TESTS PASSED')
