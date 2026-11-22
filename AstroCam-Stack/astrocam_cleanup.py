"""Deterministic cosmetic cleanup of an existing astronomical stack.

No synthetic content, deconvolution, replacement stars or learned models.
The original linear master is only read. Outputs use a display stretch and
must not replace raw/calibrated data for photometry.
"""
from pathlib import Path
import argparse
import json
import sys
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
import cv2
import numpy as np
import sep
import tifffile
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(2)
sep.set_extract_pixstack(1000000)


def blur(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def robust_sigma(a):
    values = a[::3] if a.ndim == 1 else a[::3, ::3].ravel()
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mesh', type=int, default=80)
    parser.add_argument('--strength', type=float, default=1.1)
    parser.add_argument('--protect-circle', type=float, nargs=3, action='append', default=[], metavar=('X','Y','RADIUS'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    v04 = (args.source / 'comparison-report.json').is_file()
    if v04:
        report = json.loads((args.source / 'comparison-report.json').read_text(encoding='utf-8'))
        method = report['comparison']['chosen_method']
        master_path = args.source / f'{method}-master-linear-float32.tif'
        comparison_image = args.source / f'{method}-display.png'
        display_settings = report['comparison']['shared_display']
        pedestal = display_settings['pedestal']
        white = display_settings['white']
        # Match the v0.4 comparison's background model, not a stronger one.
        args.mesh = display_settings['background_mesh_pixels']
    else:
        report = json.loads((args.source / 'rapport.json').read_text(encoding='utf-8'))
        master_path = args.source / 'AstroCam-master-linear-16bit.tif'
        comparison_image = args.source / 'AstroCam-stacked.png'
        pedestal = .001
        white = report['display_transform']['white_level_linear']
    source_hash = hashlib.file_digest(master_path.open('rb'), 'sha256').hexdigest()
    data = tifffile.imread(master_path)
    if data.dtype == np.uint16:
        rgb = data.astype(np.float32) / 65535
    elif np.issubdtype(data.dtype, np.floating):
        rgb = data.astype(np.float32)
    else:
        raise ValueError('Unsupported master pixel type')
    del data
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and np.isfinite(rgb).all()
    h, w = rgb.shape[:2]
    print('Estimating the sky background per colour channel...', flush=True)
    background_info = []
    for c in range(3):
        channel = np.ascontiguousarray(rgb[..., c])
        bg = sep.Background(channel, bw=args.mesh, bh=args.mesh, fw=3, fh=3)
        background_info.append({'channel': c, 'global_background': float(bg.globalback), 'global_rms': float(bg.globalrms)})
        rgb[..., c] -= bg.back()
        del channel, bg
    weights = np.array([0.2126, 0.7152, 0.0722], np.float32)
    luma = rgb @ weights
    luma = np.ascontiguousarray(luma)
    bg = sep.Background(luma, bw=128, bh=128, fw=3, fh=3)
    objects = sep.extract(luma, 4, err=bg.rms(), minarea=6)
    stars = objects[(objects['b'] > 0.9) & (objects['a'] < 14) & (objects['a'] / objects['b'] < 4) & (objects['flag'] == 0)]
    del objects, bg
    print(f'Protecting {len(stars)} compact sources during noise reduction...', flush=True)
    hard_mask = np.zeros((h, w), np.uint8)
    for s in stars:
        axes = (max(5, int(np.ceil(s['a'] * 3.5))), max(5, int(np.ceil(s['b'] * 3.5))))
        cv2.ellipse(hard_mask, (round(s['x']), round(s['y'])), axes,
                    float(s['theta'] * 180/np.pi), 0, 360, 255, -1)
    for x,y,radius in args.protect_circle:
        if not (0 <= x < w and 0 <= y < h and 0 < radius < min(w,h)/4):
            raise ValueError('Invalid protection circle')
        cv2.circle(hard_mask,(round(x),round(y)),round(radius),255,-1)
    mask = blur(cv2.dilate(hard_mask, np.ones((5, 5), np.uint8)).astype(np.float32)/255, 2)
    blank = (mask < 0.01)
    print('Reducing fine-scale luminance noise...', flush=True)
    current = luma
    details = []
    thresholds = []
    for sigma, factor in [(0.85, 1.0), (1.7, 0.65), (3.4, 0.35)]:
        low = blur(luma, sigma)
        detail = current - low
        noise = robust_sigma(detail[blank])
        threshold = noise * factor * args.strength
        # Smooth shrinkage of high-frequency residuals. The compact-source
        # mask restores the original luminance around detected stars.
        filtered = detail * np.maximum(0, 1 - threshold**2 / (detail**2 + 1e-16))
        details.append(filtered)
        thresholds.append({'sigma_pixels': sigma, 'noise': noise, 'threshold': threshold})
        current = low
    clean_luma = current.copy()
    for detail in details:
        clean_luma += detail
    del details, detail, filtered, low, current
    clean_luma = mask * luma + (1-mask) * clean_luma
    print('Reducing colour speckling...', flush=True)
    for c in range(3):
        chroma = rgb[..., c] - luma
        smooth = blur(chroma, 1.4 * args.strength)
        # Keep most original colour at star cores, smooth background colour.
        colour_protection = mask * 0.75
        rgb[..., c] = clean_luma + colour_protection * chroma + (1-colour_protection) * smooth
    del chroma, smooth, colour_protection

    original_noise = robust_sigma((luma - blur(luma, 1))[blank])
    cleaned_noise = robust_sigma((clean_luma - blur(clean_luma, 1))[blank])
    # Same empty locations before/after; this includes correlated noise on
    # approximately a stellar-aperture scale, beyond the fine-grain statistic.
    yy,xx = np.mgrid[32:h-32:70,32:w-32:70]
    blank_points = np.column_stack((xx.ravel(),yy.ravel()))
    exclusion = cv2.dilate((mask>.01).astype(np.uint8),np.ones((41,41),np.uint8))
    blank_points = blank_points[exclusion[blank_points[:,1],blank_points[:,0]]==0]
    def aperture_values(gray,points):
        values=[]
        gy,gx=np.mgrid[-20:21,-20:21]
        for x,y in points:
            ix,iy=round(x),round(y)
            patch=gray[iy-20:iy+21,ix-20:ix+21]
            if patch.shape!=(41,41): continue
            rr=(gx+ix-x)**2+(gy+iy-y)**2
            sky=np.median(patch[(rr>15**2)&(rr<20**2)])
            values.append(float((patch[rr<10**2]-sky).sum()))
        return np.array(values)
    blank_before = aperture_values(luma,blank_points)
    blank_after = aperture_values(clean_luma,blank_points)
    aperture_noise_before = 1.4826*np.median(np.abs(blank_before-np.median(blank_before)))
    aperture_noise_after = 1.4826*np.median(np.abs(blank_after-np.median(blank_after)))
    protected_before = aperture_values(luma,[(x,y) for x,y,_ in args.protect_circle])
    protected_after = aperture_values(clean_luma,[(x,y) for x,y,_ in args.protect_circle])
    checks = []
    for s in sorted(stars, key=lambda s: s['flux'], reverse=True)[:200]:
        x, y = int(round(s['x'])), int(round(s['y']))
        if x < 20 or y < 20 or x >= w-20 or y >= h-20:
            continue
        yy, xx = np.mgrid[y-16:y+17, x-16:x+17]
        rr = (xx-s['x'])**2 + (yy-s['y'])**2
        aperture, annulus = rr < 8**2, (rr > 12**2) & (rr < 16**2)
        before = luma[y-16:y+17, x-16:x+17]
        after = clean_luma[y-16:y+17, x-16:x+17]
        b = float(np.sum(before[aperture] - np.median(before[annulus])))
        a = float(np.sum(after[aperture] - np.median(after[annulus])))
        if b > 0.02:
            checks.append(a/b)
    settings = {
        'input_master': str(master_path.resolve()),
        'input_master_sha256': source_hash,
        'comparison_image': str(comparison_image.resolve()),
        'size_hwc': list(rgb.shape), 'background_mesh_pixels': args.mesh,
        'background_filter_meshes': 3, 'background_channels': background_info,
        'compact_sources_protected': len(stars), 'source_mask_fraction': float((mask > 0.5).mean()),
        'noise_shrinkage': thresholds, 'chroma_smoothing_sigma_pixels': 1.4 * args.strength,
        'star_colour_original_fraction_at_mask_core': 0.75,
        'background_high_frequency_mad_before_linear': original_noise,
        'background_high_frequency_mad_after_linear': cleaned_noise,
        'background_high_frequency_mad_ratio': cleaned_noise / original_noise,
        'blank_aperture_count': len(blank_points),
        'blank_aperture_mad_before_linear': float(aperture_noise_before),
        'blank_aperture_mad_after_linear': float(aperture_noise_after),
        'blank_aperture_mad_ratio': float(aperture_noise_after/aperture_noise_before),
        'additional_protected_circles': args.protect_circle,
        'protected_circle_aperture_flux_ratio': (protected_after/protected_before).tolist(),
        'star_aperture_checks_count': len(checks),
        'star_aperture_luminance_ratio_quantiles_5_50_95': np.percentile(checks, [5,50,95]).tolist() if checks else [],
        'limitations': 'Cosmetic display processing. Noise statistic measures smoothness, not improved photon statistics. Background removal can suppress diffuse astronomical structures. No new dark/flat calibration. Not intended for photometry.',
    }
    print(json.dumps({k:v for k,v in settings.items() if 'ratio' in k or 'count' in k}, indent=2), flush=True)
    if not checks or abs(np.median(checks)-1) > 0.05:
        raise RuntimeError('Star-preservation check failed: median aperture luminance changed over 5 percent')

    # Use the earlier image's white level so the comparison does not obtain
    # a cleaner appearance simply by reducing the overall display contrast.
    rgb = np.maximum(rgb + pedestal, 0)
    rgb = np.clip(np.arcsinh(rgb * (5/white)) / np.arcsinh(5), 0, 1)
    assert np.isfinite(rgb).all()
    display16 = np.uint16(np.rint(rgb * 65535))
    del rgb
    display8 = np.uint8(np.rint(display16.astype(np.float32) / 257))
    settings['display'] = {'pedestal_linear': pedestal, 'white_level_linear': white, 'curve': 'asinh, factor 5'}
    tifffile.imwrite(args.output / 'AstroCam-cleaned-display-16bit.tif', display16, photometric='rgb',
                     metadata={'description': 'Cosmetic background subtraction and star-protected denoising; asinh display curve. Nonlinear; not for photometry.'})
    Image.fromarray(display8).save(args.output / 'AstroCam-cleaned.png')
    Image.fromarray(display8).save(args.output / 'AstroCam-cleaned.jpg', quality=95)
    preview = Image.fromarray(display8)
    preview.thumbnail((900, 1200))
    preview.save(args.output / 'AstroCam-cleaned-preview.jpg', quality=94)
    old = np.asarray(Image.open(comparison_image).convert('RGB'))
    assert old.shape == display8.shape
    panel = Image.new('RGB', (1600, 1190), '#10151e')
    draw = ImageDraw.Draw(panel)
    try:
        title_font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 25)
        small_font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 20)
    except OSError:
        title_font = small_font = ImageFont.load_default()
    for col, (array, title) in enumerate([(old, 'v0.4 stack - before noise reduction' if v04 else 'Previous processing'), (display8, 'Noise reduction - same display contrast')]):
        x = 20 + col*800
        draw.text((x, 12), title, fill='white', font=title_font)
        view = Image.fromarray(array)
        view.thumbnail((760, 720))
        panel.paste(view, (x+(760-view.width)//2, 55))
        cy, cx = h//2, w//2
        crop = Image.fromarray(array[cy-180:cy+180, cx-360:cx+360])
        panel.paste(crop, (x+20, 825))
        draw.text((x+20, 787), 'Central detail at 100% pixel scale', fill='#bfc8d8', font=small_font)
    panel.save(args.output / 'AstroCam-before-after.jpg', quality=95)
    assert hashlib.file_digest(master_path.open('rb'), 'sha256').hexdigest() == source_hash
    settings['linear_master_sha256_unchanged'] = True
    (args.output / 'processing.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    # Verify saved outputs, including all channels of the 16-bit display TIFF.
    assert np.array_equal(tifffile.imread(args.output / 'AstroCam-cleaned-display-16bit.tif'), display16)
    with Image.open(args.output / 'AstroCam-cleaned.png') as image:
        image.verify()
    print('Saved and verified:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
