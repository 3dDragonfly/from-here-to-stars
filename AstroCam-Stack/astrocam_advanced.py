"""AstroCam v0.4: validated wide-field registration and weighted integration.

MAD and Winsorized estimators are compared on identical normalized inputs.
No learned image generation or cosmetic denoising is used in the comparison.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import sep
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from astrocam_stack import analyse, decode, register, common_crop

VERSION = '0.4.0'
LUMA = np.array([.2126, .7152, .0722], np.float32)
sep.set_extract_pixstack(1000000)
cv2.setNumThreads(2)


def log(text):
    print(text, flush=True)


def mad(values, axis=None):
    center = np.median(values, axis=axis, keepdims=axis is not None)
    return 1.4826*np.median(np.abs(values-center), axis=axis)


def locate_stars(rgb):
    gray = np.ascontiguousarray(rgb.astype(np.float32) @ LUMA / 65535)
    h,w = gray.shape
    half = cv2.resize(gray, (w//2,h//2), interpolation=cv2.INTER_AREA)
    bg = sep.Background(half, bw=64,bh=64,fw=3,fh=3)
    stars = sep.extract(half-bg.back(), 6, err=bg.rms(), minarea=6)
    stars = stars[(stars['b']>.7)&(stars['a']<6)&(stars['a']/stars['b']<3)&(stars['flag']==0)]
    stars = stars[np.argsort(stars['flux'])[::-1]][:400]
    x,y = stars['x']*2+.5,stars['y']*2+.5
    residual = np.ascontiguousarray(gray-cv2.resize(bg.back(),(w,h),interpolation=cv2.INTER_CUBIC))
    xf,yf,flags = sep.winpos(residual,x,y,3.5,subpix=5)
    flux,_,ff = sep.sum_circle(residual,xf,yf,10,subpix=5)
    good = (flags==0)&(ff==0)&np.isfinite(flux)&(flux>0)&(np.hypot(xf-x,yf-y)<4)
    good &= (xf>24)&(xf<w-24)&(yf>24)&(yf<h-24)
    peaks=[]
    for xx,yy in zip(xf,yf):
        ix,iy=round(xx),round(yy)
        peaks.append(float(rgb[max(0,iy-5):iy+6,max(0,ix-5):ix+6].max()))
    good &= np.array(peaks)<64000
    return {'points':np.column_stack((xf[good],yf[good])), 'flux':flux[good],
            'width':(2*2.35482*np.sqrt(stars['a']*stars['b']))[good],
            'noise':float(bg.globalrms), 'shape':(h,w)}


def match_points(source, reference, coarse, radius=5):
    projected=cv2.perspectiveTransform(source.reshape(-1,1,2).astype(np.float64),coarse).reshape(-1,2)
    distance,index=cKDTree(reference).query(projected,distance_upper_bound=radius)
    keep=[]; used=set()
    for i in np.argsort(distance):
        if not np.isfinite(distance[i]): break
        if index[i] not in used:
            keep.append(i);used.add(index[i])
    keep=np.array(keep,int)
    return keep,index[keep]


def project(points,matrix):
    return cv2.perspectiveTransform(points.reshape(-1,1,2).astype(np.float64),matrix).reshape(-1,2)


def geometry_ok(matrix,shape):
    h,w=shape
    corners=np.array([[0,0],[w,0],[w,h],[0,h]],np.float64)
    moved=project(corners,matrix)
    return bool(np.isfinite(moved).all() and cv2.isContourConvex(moved.astype(np.float32))
                and .85*w*h < cv2.contourArea(moved.astype(np.float32)) < 1.15*w*h
                and np.max(np.linalg.norm(moved-corners,axis=1)) < .15*max(h,w))


def fit_registration(src,ref,coarse,seed):
    si,ri=match_points(src['points'],ref['points'],coarse)
    if len(si)<40: raise ValueError(f'Kun {len(si)} praecise stjernepar; mindst 40 kraeves')
    a,b=src['points'][si],ref['points'][ri]
    order=np.random.default_rng(seed).permutation(len(a))
    split=max(12,len(a)//4); test=order[:split]; train=order[split:]
    similarity,_=cv2.estimateAffinePartial2D(a[train],b[train],method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=5000,confidence=.999)
    if similarity is None: raise ValueError('Kunne ikke tilpasse similarity-model')
    similarity=np.vstack((similarity,[0,0,1]))
    homography,_=cv2.findHomography(a[train],b[train],cv2.RANSAC,1.5,maxIters=5000,confidence=.999)
    err_old=np.linalg.norm(project(a[test],coarse)-b[test],axis=1)
    err_sim=np.linalg.norm(project(a[test],similarity)-b[test],axis=1)
    err_h=np.linalg.norm(project(a[test],homography)-b[test],axis=1) if homography is not None else np.full(len(test),np.inf)
    # Coordinate clipping prevents a few pathological centroid failures from
    # dominating the score; full residual summaries are retained in the report.
    score=lambda v:float(np.sqrt(np.mean(np.minimum(v,5)**2)))
    use_h=homography is not None and geometry_ok(homography,src['shape']) and score(err_h)<.97*score(err_sim)
    h,w=src['shape']
    cells=set(zip((b[train,0]/w*3).astype(int),(b[train,1]/h*3).astype(int)))
    use_h=use_h and len(cells)>=5
    # Report held-out scores from the training-only fits. Refit the selected
    # model to all pairs only AFTER model choice; these points are not then
    # used to claim an independent validation score.
    if use_h:
        final,inliers=cv2.findHomography(a,b,cv2.RANSAC,1.5,maxIters=5000,confidence=.999)
        chosen_error=err_h;model='homography'
    else:
        final,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=5000,confidence=.999)
        final=np.vstack((final,[0,0,1]));chosen_error=err_sim;model='similarity'
    if not geometry_ok(final,src['shape']): raise ValueError('Ugyldig billedgeometri')
    if score(chosen_error)>min(1.5,score(err_old)*1.05):
        final=coarse;model='previous_similarity';chosen_error=err_old
    ratio=ref['flux'][ri]/src['flux'][si]
    photometric=np.isfinite(ratio)&(ratio>.5)&(ratio<2)
    for _ in range(3):
        center=np.median(ratio[photometric]);spread=max(float(mad(ratio[photometric])),.015)
        photometric &= np.abs(ratio-center)<3*spread
    if photometric.sum()<20: raise ValueError('For faa stabile stjerner til lysnormalisering')
    gain=float(np.median(ratio[photometric]))
    selected=inliers.ravel().astype(bool) if inliers is not None else np.ones(len(a),bool)
    outer=np.linalg.norm((b[test]-[w/2,h/2])/[w/2,h/2],axis=1)>.65
    stats={'model':model,'pairs':len(a),'inliers':int(selected.sum()),'training_cells_3x3':len(cells),
           'validation_count':len(test),'validation_previous_rms_px':score(err_old),
           'validation_refined_similarity_rms_px':score(err_sim),'validation_homography_rms_px':score(err_h),
           'validation_selected_rms_px':score(chosen_error),'validation_outer_count':int(outer.sum()),
           'validation_outer_selected_rms_px':score(chosen_error[outer]) if outer.any() else None,
           'photometric_gain':gain,'photometric_stars':int(photometric.sum()),
           'photometric_ratio_mad':float(mad(ratio[photometric])),
           'relative_psf_area':float(np.median((src['width'][si[photometric]]/ref['width'][ri[photometric]])**2))}
    return final,stats


def background_plane_difference(reference,frame,gain):
    """Fit additive reference-minus-frame gradient, not an absolute sky model.

    Shared diffuse astronomical structure is retained by differencing first.
    """
    h,w=frame.shape[:2]
    small=cv2.resize(frame,(w//6,h//6),interpolation=cv2.INTER_AREA)
    ys=np.linspace(-1,1,18);xs=np.linspace(-1,1,14)
    yy,xx=np.meshgrid(ys,xs,indexing='ij')
    design=np.column_stack((np.ones(xx.size),xx.ravel(),yy.ravel()))
    coefficients=[]
    for c in range(3):
        rb=sep.Background(np.ascontiguousarray(reference[...,c]),bw=24,bh=24,fw=3,fh=3).back()
        fb=sep.Background(np.ascontiguousarray(small[...,c]),bw=24,bh=24,fw=3,fh=3).back()
        delta=cv2.resize(rb-gain*fb,(14,18),interpolation=cv2.INTER_AREA).ravel()
        keep=np.ones(len(delta),bool)
        for _ in range(4):
            coef=np.linalg.lstsq(design[keep],delta[keep],rcond=None)[0]
            residual=delta-design@coef
            keep=np.abs(residual-np.median(residual))<max(3*float(mad(residual)),.0005)
        coefficients.append(coef)
    return np.array(coefficients)


def noise_channels(rgb):
    # MAD of adjacent-pixel differences, same statistic for all frames. This
    # ranks background noise; no gain/Poisson calibration is claimed.
    values=[]
    for c in range(3):
        a=rgb[::2,::2,c]
        d=(a[1:,1:]-a[:-1,:-1]).ravel()
        values.append(max(float(mad(d))/np.sqrt(2),1/65535))
    return np.array(values)


def weighted_integrate(block,weights,noise_floor,method):
    center=np.median(block,axis=0)
    sigma=np.maximum(mad(block,axis=0),noise_floor)
    if method=='winsorized':
        # Gaussian consistency factor for population variance after clamping
        # standard-normal values to +/-1.5 sigma. Only the scale estimate
        # uses clamped values; final averaging uses original retained samples.
        c=1.5;phi=math.exp(-c*c/2)/math.sqrt(2*math.pi);q=.5*math.erfc(c/math.sqrt(2))
        correction=1/math.sqrt(1-2*c*phi+2*(c*c-1)*q)
        for _ in range(8):
            limited=np.clip(block,center-c*sigma,center+c*sigma)
            new_center=limited.mean(axis=0)
            new_sigma=np.maximum(limited.std(axis=0,ddof=1)*correction,noise_floor)
            change=float(np.quantile(np.abs(new_sigma-sigma)/(sigma+1e-12),.95))
            center,sigma=new_center,new_sigma
            if change<.001: break
    elif method!='mad':
        raise ValueError('Ukendt clipping-metode')
    keep=np.abs(block-center)<=3*sigma
    ww=weights[:,None,None,None]*keep
    denom=ww.sum(axis=0)
    result=np.divide((block*ww).sum(axis=0),denom,out=center.copy(),where=denom>0)
    neff=np.divide(denom**2,(ww*ww).sum(axis=0),out=np.zeros_like(denom),where=denom>0)
    rejected=(~keep).sum(axis=0)
    return result,rejected,neff


def select_frames(args):
    source=args.input.resolve(strict=True)
    if args.selection_report:
        prior=json.loads(args.selection_report.read_text(encoding='utf-8'))
        if Path(prior['input']).resolve()!=source: raise ValueError('Udvalgsrapporten tilhoerer en anden mappe')
        selected=[dict(r) for r in prior['frames'] if r['accepted']]
        if len(selected)<3: raise ValueError('Mindst 3 udvalgte RAW-optagelser kraeves')
        for row in selected:
            name=row['file']
            if Path(name).name!=name or Path(name).suffix.lower()!='.dng': raise ValueError('Ugyldigt RAW-filnavn')
            if not (source/name).is_file(): raise ValueError('En udvalgt RAW-fil mangler')
        return source,selected,prior['reference'],prior
    files=sorted(p for p in source.iterdir() if p.suffix.lower()=='.dng' and p.is_file())
    rows=[]
    for i,p in enumerate(files):
        log(f'Kvalitetskontrol {i+1}/{len(files)}: {p.name}')
        try: rows.append(analyse(p,None))
        except Exception as error: log(f'Fravalgt: {p.name}: {error}')
    viable=[r for r in rows if r['stars']>=25 and r['saturated']<=.005]
    if not viable: raise ValueError('Ingen brugbare RAW-optagelser')
    ref=min(viable,key=lambda r:r['median'])
    selected=[r for r in viable if r['median']<=ref['median']*args.max_sky_ratio and r['shape']==ref['shape']]
    if len(selected)<3: raise ValueError('Mindst 3 brugbare RAW-optagelser kraeves')
    return source,selected,ref['file'],None


def background_subtracted(rgb):
    a=rgb.astype(np.float32).copy()
    for c in range(3):
        bg=sep.Background(np.ascontiguousarray(a[...,c]),bw=128,bh=128,fw=3,fh=3)
        a[...,c]-=bg.back()
    return a


def aperture_flux(gray,points,radius=10):
    values=[]
    rr=20; gy,gx=np.mgrid[-rr:rr+1,-rr:rr+1]
    for x,y in points:
        ix,iy=round(x),round(y)
        patch=gray[iy-rr:iy+rr+1,ix-rr:ix+rr+1]
        if patch.shape!=(2*rr+1,2*rr+1): values.append(np.nan);continue
        r2=(gx+ix-x)**2+(gy+iy-y)**2
        sky=np.median(patch[(r2>15**2)&(r2<20**2)])
        values.append(float((patch[r2<radius**2]-sky).sum()))
    return np.array(values)


def compare_results(output,masters,reference_points,crop,prior,registration_report):
    log('Sammenligner med faelles baggrunds- og kontrastindstillinger...')
    x0,y0,x1,y1=crop
    points=reference_points-[x0,y0]
    h,w=next(iter(masters.values())).shape[:2]
    points=points[(points[:,0]>25)&(points[:,0]<w-25)&(points[:,1]>25)&(points[:,1]<h-25)]
    residuals={name:background_subtracted(rgb) for name,rgb in masters.items()}
    gray={name:np.ascontiguousarray(rgb@LUMA) for name,rgb in residuals.items()}
    if 'previous' in masters:
        a=aperture_flux(gray['previous'],points);b=aperture_flux(gray['mad'],points)
        valid=np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)
        ratios=b[valid]/a[valid]
        keep=np.abs(ratios-np.median(ratios))<max(3*float(mad(ratios)),.02)
        baseline_gain=float(np.median(ratios[keep]))
        residuals['previous']*=baseline_gain;gray['previous']*=baseline_gain
    else: baseline_gain=None
    # Same blank-aperture locations for every output. Use the union of source
    # detections, avoiding a favourable mask selected from only one algorithm.
    mask=np.zeros((h,w),np.uint8)
    for name,g in gray.items():
        log('Finder faelles baggrundsomraader: '+name)
        bg=sep.Background(g,bw=128,bh=128,fw=3,fh=3)
        sources=sep.extract(g-bg.back(),3,err=bg.rms(),minarea=6)
        for star in sources:
            cv2.ellipse(mask,(round(star['x']),round(star['y'])),(max(18,int(star['a']*4)),max(18,int(star['b']*4))),float(np.degrees(star['theta'])),0,360,1,-1)
    yy,xx=np.mgrid[32:h-32:70,32:w-32:70]
    blanks=np.column_stack((xx.ravel(),yy.ravel()))
    expanded=cv2.dilate(mask,np.ones((41,41),np.uint8))
    blanks=blanks[expanded[blanks[:,1],blanks[:,0]]==0]
    if len(blanks)<50: raise ValueError('For faa faelles baggrundsmaalinger til sammenligning')
    metrics={}
    # Only use well-separated reference sources for width estimates.
    distances,_=cKDTree(points).query(points,k=2)
    width_points=points[distances[:,1]>36][:120]
    from scipy.optimize import least_squares
    for name,g in gray.items():
        log('Maaling af stoej og stjernebredde: '+name)
        blank_flux=aperture_flux(g,blanks)
        star_flux=aperture_flux(g,points)
        widths=[]
        for x,y in width_points:
            ix,iy=round(x),round(y);patch=g[iy-10:iy+11,ix-10:ix+11]
            gy,gx=np.mgrid[-10:11,-10:11]
            bg=float(np.median(patch));amp=float(patch.max()-bg)
            if amp<=0: continue
            def model(p):
                b,a,dx,dy,sx,sy,theta=p
                u=(gx-dx)*np.cos(theta)+(gy-dy)*np.sin(theta)
                v=-(gx-dx)*np.sin(theta)+(gy-dy)*np.cos(theta)
                return (b+a*np.exp(-.5*((u/sx)**2+(v/sy)**2))-patch).ravel()
            fit=least_squares(model,[bg,amp,x-ix,y-iy,3,3,0],bounds=([-np.inf,0,-4,-4,1,1,-np.pi],[np.inf,np.inf,4,4,9,9,np.pi]),max_nfev=120)
            if fit.success and 1.05<min(fit.x[4:6]) and max(fit.x[4:6])<8.9:
                widths.append(2.35482*np.sqrt(fit.x[4]*fit.x[5]))
        metrics[name]={'blank_apertures':len(blanks),'blank_aperture_mad':float(mad(blank_flux[np.isfinite(blank_flux)])),
                       'median_star_aperture_flux':float(np.nanmedian(star_flux)),
                       'stellar_gaussian_fwhm_median_px':float(np.median(widths)) if widths else None,
                       'stellar_width_fits':len(widths),'star_fluxes':star_flux.tolist()}
    flux_ratio=np.array(metrics['winsorized']['star_fluxes'])/np.array(metrics['mad']['star_fluxes'])
    metrics['paired_winsorized_to_mad_star_flux_ratio_quantiles']=np.nanpercentile(flux_ratio,[5,50,95]).tolist()
    for name in masters: metrics[name].pop('star_fluxes')
    # A small noise difference is not enough to establish a winner. Flux and
    # stellar-width consistency are required; default to transparent MAD.
    a,b=metrics['mad'],metrics['winsorized']
    ratio=float(np.nanmedian(flux_ratio))
    acceptable=.98<ratio<1.02 and b['stellar_gaussian_fwhm_median_px']<=1.02*a['stellar_gaussian_fwhm_median_px']
    winner='winsorized' if acceptable and b['blank_aperture_mad']<.98*a['blank_aperture_mad'] else 'mad'
    white=prior['display_transform']['white_level_linear'] if prior else max(float(np.percentile(residuals['mad'][::4,::4],99.95)),.01)
    pedestal=.0015574503690004347 if prior else .001
    display={}
    for name,data in residuals.items():
        a=np.uint8(np.clip(np.arcsinh(np.maximum(data+pedestal,0)*(5/white))/np.arcsinh(5),0,1)*255)
        display[name]=a
        Image.fromarray(a).save(output/f'{name}-display.jpg',quality=95)
        if name!='previous': Image.fromarray(a).save(output/f'{name}-display.png')
    preview=Image.fromarray(display[winner]);preview.thumbnail((1000,1300));preview.save(output/'AstroCam-v04-preview.jpg',quality=95)
    Image.fromarray(display[winner]).save(output/'AstroCam-v04-selected.png')
    names=list(masters)
    fontpath='C:/Windows/Fonts/segoeui.ttf'
    font=ImageFont.truetype(fontpath,24);small=ImageFont.truetype(fontpath,19)
    canvas=Image.new('RGB',(600*len(names),1420),'#10151e');draw=ImageDraw.Draw(canvas)
    labels={'previous':'Tidligere v0.3','mad':'Ny: vaegtet MAD','winsorized':'Ny: vaegtet Winsorized'}
    # Two fixed locations, identical pixels and transfer function. No denoising.
    centers=[('Stjernefelt',(w//2,h//2)),('Ydre felt',(min(w-281,max(280,round(.8*w))),min(h-141,max(140,round(.59*h)))))]
    for col,name in enumerate(names):
        draw.text((col*600+15,10),labels[name],font=font,fill='white')
        full=Image.fromarray(display[name]);full.thumbnail((570,740));canvas.paste(full,(col*600+(600-full.width)//2,55))
        for j,(label,(cx,cy)) in enumerate(centers):
            cut=Image.fromarray(display[name]).crop((cx-280,cy-140,cx+280,cy+140))
            canvas.paste(cut,(col*600+20,825+j*300))
            draw.text((col*600+20,795+j*300),label+' - 100%',font=small,fill='#c5d0df')
    canvas.save(output/'AstroCam-method-comparison.jpg',quality=95)
    return {'metrics':metrics,'baseline_display_gain':baseline_gain,'chosen_method':winner,
            'selection_rule':'Winsorized only if blank-aperture MAD >2% lower, paired median star flux within 2%, stellar FWHM no more than 2% worse; otherwise MAD',
            'shared_display':{'background_mesh_pixels':128,'pedestal':pedestal,'white':white,'curve':'asinh(5*x/white)/asinh(5)','denoising':False},
            'interpretation':'Image-quality diagnostics, not calibrated photon statistics or proof that the same method wins on other sessions.'}


def run_advanced(args):
    import rawpy
    source,rows,reference_name,prior=select_frames(args)
    output=args.output.resolve() if args.output else source/('Stack_v04_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    report={'version':VERSION,'input':str(source),'output':str(output),'reference':reference_name,
            'selection_report':str(args.selection_report.resolve()) if args.selection_report else None,
            'same_frame_count':len(rows),'calibration':'No separate dark/flat/bias frames',
            'normalization':'One scalar stellar-flux gain, then per-channel additive plane matching the reference background',
            'weighting':'Inverse normalized luminance-noise variance divided by relative PSF area; capped to [0.3,3] times median raw weight',
            'frames':[],'status':'running'}
    report_path=output/'comparison-report.json'
    try:
        with rawpy.imread(str(source/reference_name)) as raw: wb=list(raw.camera_whitebalance)
        rgb=decode(source/reference_name,wb)
        ref=locate_stars(rgb);h,w=rgb.shape[:2]
        needed=rgb.nbytes*2*(len(rows)+5)
        if shutil.disk_usage(output).free<needed: raise ValueError('For lidt plads til flydende cache og resultater')
        ref_small=cv2.resize(rgb.astype(np.float32)/65535,(w//6,h//6),interpolation=cv2.INTER_AREA)
        del rgb
        overlap=np.ones((h,w),bool)
        before={r['file']:(source/r['file']).stat() for r in rows}
        cache_parent=output.resolve()
        with tempfile.TemporaryDirectory(prefix='.astrocam-v04-',dir=output) as name:
            cache=Path(name).resolve()
            if cache.parent!=cache_parent: raise ValueError('Cachemappe uden for output')
            for i,row in enumerate(rows):
                log(f'Praecis justering {i+1}/{len(rows)}: {row["file"]}')
                rgb=decode(source/row['file'],wb);stars=locate_stars(rgb)
                if stars['shape']!=ref['shape']: raise ValueError('Billedstoerrelserne er forskellige')
                if row['file']==reference_name:
                    matrix=np.eye(3);stats={'model':'reference','photometric_gain':1.,'relative_psf_area':1.}
                else:
                    if 'matrix' in row: coarse=np.vstack((row['matrix'],[0,0,1]))
                    else:
                        coarse2,_,_=register((stars['points']-.5)/2,(ref['points']-.5)/2)
                        coarse=np.vstack((coarse2,[0,0,1]))
                    matrix,stats=fit_registration(stars,ref,coarse,i+983)
                aligned=cv2.warpPerspective(rgb.astype(np.float32)/65535,matrix,(w,h),flags=cv2.INTER_LINEAR)
                mask=cv2.warpPerspective(np.ones((h,w),np.float32),matrix,(w,h),flags=cv2.INTER_LINEAR)
                overlap &= cv2.erode((mask>.999).astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)==1
                gain=stats['photometric_gain']
                coef=background_plane_difference(ref_small,aligned,gain)
                aligned*=gain
                yy=np.linspace(-1,1,h,dtype=np.float32)[:,None];xx=np.linspace(-1,1,w,dtype=np.float32)[None,:]
                for c in range(3): aligned[...,c]+=np.float32(coef[c,0])+np.float32(coef[c,1])*xx+np.float32(coef[c,2])*yy
                noise=noise_channels(aligned)
                stats.update(file=row['file'],matrix=matrix.tolist(),background_plane_coefficients=coef.tolist(),noise_rgb=noise.tolist(),source_size=before[row['file']].st_size,source_mtime_ns=before[row['file']].st_mtime_ns)
                report['frames'].append(stats)
                np.save(cache/f'{i:04}.npy',aligned)
                log(f"  {stats['model']}; gain {gain:.3f}; kontrolfejl {stats.get('validation_selected_rms_px',0):.3f} px")
                del rgb,aligned,mask
            crop=list(common_crop(overlap));del overlap
            if prior:
                pc=prior['crop_xyxy'];crop=[max(crop[0],pc[0]),max(crop[1],pc[1]),min(crop[2],pc[2]),min(crop[3],pc[3])]
            x0,y0,x1,y1=crop;shape=(y1-y0,x1-x0,3)
            noises=np.array([r['noise_rgb'] for r in report['frames']])
            luma_noise=np.sqrt((noises**2*LUMA**2).sum(axis=1))
            areas=np.array([r['relative_psf_area'] for r in report['frames']])
            raw_weights=1/(luma_noise**2*np.maximum(areas,.5))
            weights=np.clip(raw_weights/np.median(raw_weights),.3,3);weights/=weights.mean()
            for r,weight in zip(report['frames'],weights): r['weight']=float(weight)
            report['effective_frame_count_before_rejection']=float(weights.sum()**2/(weights**2).sum())
            floor=np.median(noises,axis=0).astype(np.float32)*.5
            masters={method:np.empty(shape,np.float32) for method in ['mad','winsorized']}
            rejections={method:np.empty(shape[:2],np.uint8) for method in masters}
            counts={m:0 for m in masters};effective={m:[] for m in masters}
            for y in range(0,shape[0],32):
                end=min(y+32,shape[0]);tiles=[]
                for i in range(len(rows)):
                    mapped=np.load(cache/f'{i:04}.npy',mmap_mode='r')
                    tiles.append(np.array(mapped[y0+y:y0+end,x0:x1],dtype=np.float32));del mapped
                block=np.stack(tiles);del tiles
                for method in masters:
                    tile,rejected,neff=weighted_integrate(block,weights.astype(np.float32),floor,method)
                    masters[method][y:end]=tile
                    counts[method]+=int(rejected.sum())
                    rejections[method][y:end]=np.uint8(np.rint(rejected.mean(axis=2)/len(rows)*255))
                    effective[method].append(float(np.median(neff)))
                del block,tile,rejected,neff
                if y%512==0:log(f'Kombinerer begge metoder: {end}/{shape[0]} raekker')
            report['crop_xyxy']=crop;report['output_shape_hwc']=list(shape)
            for method,master in masters.items():
                if not np.isfinite(master).all():raise ValueError('Ugyldige pixelvaerdier')
                path=output/f'{method}-master-linear-float32.tif'
                tifffile.imwrite(path,master,photometric='rgb',metadata={'description':f'AstroCam {VERSION}; normalized weighted {method}; linear RGB, no display gamma or denoising. No dark/flat calibration.'})
                check=tifffile.memmap(path)
                if check.shape!=master.shape or check.dtype!=np.float32 or not np.array_equal(check[::47,::43],master[::47,::43]):raise ValueError('TIFF-verifikation fejlede')
                del check
                Image.fromarray(rejections[method]).save(output/f'{method}-rejection-fraction.png')
            report['rejection']={m:{'fraction_of_channel_samples':counts[m]/(len(rows)*np.prod(shape)),'median_tile_effective_frames':float(np.median(effective[m]))} for m in masters}
            if prior:
                baseline_path=args.selection_report.parent/'AstroCam-master-linear-16bit.tif'
                if not baseline_path.is_file():baseline_path=Path(prior['output'])/'AstroCam-master-linear-16bit.tif'
                baseline=tifffile.imread(baseline_path)
                pc=prior['crop_xyxy']
                baseline=baseline[y0-pc[1]:y1-pc[1],x0-pc[0]:x1-pc[0]].astype(np.float32)/65535
                masters={'previous':baseline,**masters}
            report['comparison']=compare_results(output,masters,ref['points'],crop,prior,report['frames'])
            log('Valgt metode: '+report['comparison']['chosen_method'])
        # Confirm that every input RAW retained its original size and timestamp.
        for name,stat in before.items():
            after=(source/name).stat()
            if after.st_size!=stat.st_size or after.st_mtime_ns!=stat.st_mtime_ns:raise ValueError('En kildefil er aendret under behandlingen')
        report['source_files_unchanged_size_and_mtime']=True
        report['status']='complete'
        log('FAERDIG: '+str(output))
        return output
    except BaseException as error:
        report['status']='failed';report['error']=str(error);raise
    finally:
        report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        if report['frames']:
            fields=['file','model','weight','photometric_gain','relative_psf_area','pairs','validation_count','validation_previous_rms_px','validation_selected_rms_px','validation_outer_count','validation_outer_selected_rms_px']
            with (output/'frame-measurements.csv').open('w',newline='',encoding='utf-8-sig') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['frames'])
