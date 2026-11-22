"""AstroCam RAW stacking. Original files are opened read-only.

Dependencies live in ./lib; numpy, scipy, OpenCV and Pillow may also be
provided by the selected Python installation. See requirements.txt.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import astroalign
import cv2
import numpy as np
import rawpy
import sep
import tifffile
from PIL import Image, ImageDraw

VERSION = "0.3.0"
cv2.setNumThreads(2)


def log(message):
    print(message, flush=True)


def decode(path, wb, half=False):
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(
            half_size=half, user_wb=wb, no_auto_bright=True,
            adjust_maximum_thr=0, gamma=(1, 1), output_bps=16,
        )


def analyse(path, wb):
    rgb = decode(path, wb, half=True).astype(np.float32) / 65535
    gray = np.ascontiguousarray(rgb.mean(axis=2))
    bg = sep.Background(gray)
    stars = sep.extract(gray - bg.back(), 8, err=bg.rms(), minarea=6)
    stars = stars[(stars["b"] > 0.7) & (stars["a"] < 6)
                  & (stars["a"] / stars["b"] < 3) & (stars["flag"] == 0)]
    count = len(stars)
    stars = stars[np.argsort(stars["flux"])[::-1]][:100]
    return {
        "file": path.name, "stars": count,
        "median": float(np.median(gray)),
        "saturated": float((rgb.max(axis=2) > 0.98).mean()),
        "noise": float(np.median(bg.rms())),
        "shape": list(rgb.shape[:2]),
        "points": np.column_stack((stars["x"], stars["y"])).tolist(),
        "accepted": False, "reason": "",
    }


def register(points, reference):
    transform, pairs = astroalign.find_transform(
        np.asarray(points), np.asarray(reference), max_control_points=70
    )
    src, dst = pairs
    residual = np.linalg.norm(transform(src) - dst, axis=1)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    if len(src) < 15 or rms > 1.2:
        raise ValueError(f"For faa stjernematches eller for stor fejl: {len(src)}, {rms:.2f} px")
    if not 0.98 <= transform.scale <= 1.02 or abs(transform.rotation) > math.radians(15):
        raise ValueError("Usandsynlig skalering eller rotation")
    # Half-size pixel centres correspond to full-size positions 2*x + 0.5.
    matrix = transform.params[:2].copy()
    matrix[:, 2] = 2 * matrix[:, 2] + 0.5 - matrix[:, :2] @ np.array([0.5, 0.5])
    return matrix, len(src), rms * 2


def common_crop(mask):
    h, w = mask.shape
    x0, y0, x1, y1 = 0, 0, w, h
    while x1 - x0 > w // 2 and y1 - y0 > h // 2:
        edges = [mask[y0, x0:x1].all(), mask[y1-1, x0:x1].all(),
                 mask[y0:y1, x0].all(), mask[y0:y1, x1-1].all()]
        if all(edges):
            if not mask[y0:y1, x0:x1].all():
                raise ValueError("Huller i billedernes faelles overlap")
            return x0, y0, x1, y1
        if not edges[0]: y0 += 1
        if not edges[1]: y1 -= 1
        if not edges[2]: x0 += 1
        if not edges[3]: x1 -= 1
    raise ValueError("For lidt faelles billedareal")


def clipped_mean(block, noise_floor):
    # A robust centre prevents a bright satellite/hot pixel from setting its
    # own rejection threshold. Keep at least a physical noise floor.
    median = np.median(block, axis=0)
    sigma = 1.4826 * np.median(np.abs(block - median), axis=0)
    limit = 3 * np.maximum(sigma, noise_floor)
    keep = np.abs(block - median) <= limit
    count = keep.sum(axis=0)
    total = np.sum(block, axis=0, where=keep, dtype=np.float32)
    result = np.divide(total, count, out=median.copy(), where=count > 0)
    return result, int(keep.size - keep.sum())


def display_image(linear, settings=None):
    """Cosmetic sky-background removal + asinh stretch; master stays linear.

    Large diffuse astronomical structures can be attenuated by this preview
    transform. Use the master for scientific/photometric work.
    """
    h, w = linear.shape[:2]
    grid = np.zeros((24, 18, 3), np.float32)
    for gy in range(24):
        for gx in range(18):
            tile = linear[gy*h//24:(gy+1)*h//24, gx*w//18:(gx+1)*w//18]
            grid[gy, gx] = np.percentile(tile, 35, axis=(0, 1))
    background = cv2.resize(cv2.GaussianBlur(grid, (0, 0), 1.2), (w, h), interpolation=cv2.INTER_CUBIC)
    corrected = linear - background
    # A small neutral pedestal avoids crushing ordinary sky noise to black.
    pedestal = settings["pedestal_linear"] if settings else max(float(np.std(corrected[::8, ::8])) * 0.15, 0.0005)
    corrected = np.maximum(corrected + pedestal, 0)
    white = settings["white_level_linear"] if settings else max(float(np.percentile(corrected[::3, ::3], 99.95)), 0.003)
    stretched = np.arcsinh(corrected * (5 / white)) / np.arcsinh(5)
    return np.uint8(np.clip(stretched, 0, 1) * 255), {
        "background": "24x18 tile 35th percentile, smoothed, per-channel subtraction",
        "stretch": "asinh", "white_level_linear": white, "pedestal_linear": pedestal,
    }


def write_report(output, report):
    (output / "rapport.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["file", "accepted", "reason", "stars", "median", "saturated", "matches", "rms_full_px"]
    with (output / "billedkvalitet.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["frames"])


def run(args):
    source = args.input.resolve(strict=True)
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".dng")
    if len(files) < 3:
        raise ValueError("Vaelg en optageserie med mindst 3 DNG-filer. JPEG-par stackes ikke oveni RAW.")
    with rawpy.imread(str(files[0])) as raw:
        wb = list(raw.camera_whitebalance)
        if not all(np.isfinite(wb)) or min(wb[:3]) <= 0:
            raise ValueError("Ugyldig hvidbalance i foerste RAW-billede")
    output = args.output.resolve() if args.output else source / ("Stack_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
    if output == source:
        raise ValueError("Output skal ligge i sin egen mappe")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "version": VERSION, "input": str(source), "output": str(output),
        "created": datetime.now().astimezone().isoformat(),
        "method": "Linear RGB, fixed first-frame camera WB, stellar similarity registration, tiled median/MAD 3-sigma clipped mean",
        "calibration": "LibRaw DNG decoding; no separate dark, bias or flat frames supplied",
        "max_sky_ratio": args.max_sky_ratio, "frames": [],
    }
    try:
        for i, path in enumerate(files):
            log(f"Analyse {i+1}/{len(files)}: {path.name}")
            try:
                row = analyse(path, wb)
            except Exception as error:
                row = {"file": path.name, "accepted": False, "reason": "Laesefejl: " + str(error)}
            report["frames"].append(row)
        candidates = [r for r in report["frames"] if r.get("stars", 0) >= 25 and r.get("saturated", 1) <= 0.005]
        if not candidates:
            raise ValueError("Ingen billeder med tilstraekkelige stjerner og uden kraftig overeksponering")
        reference = min(candidates, key=lambda r: r["median"])
        report["reference"] = reference["file"]
        threshold = reference["median"] * args.max_sky_ratio
        accepted = []
        for row in report["frames"]:
            if row["reason"]:
                continue
            if row["shape"] != reference["shape"]:
                row["reason"] = "Anden billedstoerrelse"
            elif row["stars"] < 25:
                row["reason"] = "Faerre end 25 stjerner"
            elif row["median"] > threshold or row["saturated"] > 0.005:
                row["reason"] = "Lys himmel/skyer eller overeksponering"
            else:
                try:
                    if row is reference:
                        matrix, matches, rms = np.eye(2, 3), row["stars"], 0.0
                    else:
                        matrix, matches, rms = register(row["points"], reference["points"])
                    row.update(accepted=True, reason="OK", matrix=matrix.tolist(), matches=matches, rms_full_px=rms)
                    accepted.append(row)
                    log(f"Godkendt: {row['file']}, matches={matches}, fejl={rms:.2f} px")
                except Exception as error:
                    row["reason"] = "Justering mislykkedes: " + str(error)
        if len(accepted) < 3:
            raise ValueError("Faerre end 3 billeder bestod kvalitetskontrol og stjernejustering")
        report["accepted_count"] = len(accepted)
        report["rejected_count"] = len(files) - len(accepted)
        log(f"Stacker {len(accepted)} af {len(files)} RAW-billeder i fuld oploesning.")
        write_report(output, report)

        first = decode(source / reference["file"], wb)
        h, w = first.shape[:2]
        # Each aligned frame is stored as uint16 on the output drive. Only
        # small row tiles are combined in RAM, even for long sessions.
        needed = first.nbytes * (len(accepted) + 2) + 64 * 1024**2
        if shutil.disk_usage(output).free < needed:
            raise ValueError(f"For lidt ledig diskplads; kraever cirka {needed/1024**3:.1f} GiB under stacking")
        del first
        overlap = np.ones((h, w), bool)
        cache_parent = output.resolve()
        with tempfile.TemporaryDirectory(prefix=".astrocam-cache-", dir=output) as cache_name:
            cache = Path(cache_name).resolve()
            if cache.parent != cache_parent:
                raise ValueError("Ugyldig cachemappe")
            for i, row in enumerate(accepted):
                log(f"Justerer {i+1}/{len(accepted)}: {row['file']}")
                rgb = decode(source / row["file"], wb)
                if rgb.shape != (h, w, 3):
                    raise ValueError("RAW-billederne har forskellige fulde dimensioner")
                matrix = np.asarray(row["matrix"])
                aligned = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR)
                mask = cv2.warpAffine(np.ones((h, w), np.uint8), matrix, (w, h), flags=cv2.INTER_LINEAR)
                # Erode to exclude all interpolation with the zero-filled border.
                overlap &= cv2.erode(mask, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) == 1
                np.save(cache / f"{i:04}.npy", aligned)
                del rgb, aligned, mask
            crop = common_crop(overlap)
            del overlap
            x0, y0, x1, y1 = crop
            shape = (y1-y0, x1-x0, 3)
            master = np.empty(shape, np.uint16)
            noise_floor = max(float(reference["noise"]) * 0.5 * 65535, 2)
            rejected_samples = 0
            for y in range(0, shape[0], 64):
                stop = min(y + 64, shape[0])
                tiles = []
                for i in range(len(accepted)):
                    mapped = np.load(cache / f"{i:04}.npy", mmap_mode="r")
                    tiles.append(np.array(mapped[y0+y:y0+stop, x0:x1], dtype=np.float32))
                    del mapped
                block = np.stack(tiles)
                del tiles
                tile, removed = clipped_mean(block, noise_floor)
                master[y:stop] = np.uint16(np.clip(np.rint(tile), 0, 65535))
                rejected_samples += removed
                del block, tile
                if y % 512 == 0:
                    log(f"Kombinerer raekker {stop}/{shape[0]}")
        report["crop_xyxy"] = list(crop)
        report["output_shape_hwc"] = list(shape)
        report["rejected_channel_samples"] = rejected_samples
        report["sigma_noise_floor_uint16"] = noise_floor
        tifffile.imwrite(output / "AstroCam-master-linear-16bit.tif", master, photometric="rgb",
                         metadata={"axes": "YXS", "description": "Linear RGB; no display gamma; " + report["method"]})
        display, transform = display_image(master.astype(np.float32) / 65535)
        report["display_transform"] = transform
        report["display_warning"] = "Background removal is cosmetic and can suppress diffuse astronomical structures; master has no background subtraction."
        Image.fromarray(display).save(output / "AstroCam-stacked.png")
        Image.fromarray(display).save(output / "AstroCam-stacked.jpg", quality=95)
        preview = Image.fromarray(display)
        preview.thumbnail((1200, 1500))
        preview.save(output / "AstroCam-preview.jpg", quality=94)
        # Identical display transform on the reference allows a fair comparison.
        original = decode(source / reference["file"], wb)[y0:y1, x0:x1].astype(np.float32) / 65535
        single_display, _ = display_image(original, transform)
        cx, cy = shape[1]//2, shape[0]//2
        panel = Image.new("RGB", (1280, 680), (18, 20, 28))
        draw = ImageDraw.Draw(panel)
        for i, (data, title) in enumerate([(single_display, "Enkelt RAW - samme kontrastindstillinger"), (display, f"Stack: {len(accepted)} RAW-billeder")]):
            cut = data[max(cy-320,0):cy+320, max(cx-320,0):cx+320]
            panel.paste(Image.fromarray(cut), (i*640, 40))
            draw.text((i*640+12, 12), title, fill="white")
        panel.save(output / "AstroCam-sammenligning.jpg", quality=95)
        # Pixel values stay linear in the TIFF. Display products are separate.
        report["status"] = "complete"
        write_report(output, report)
        log(f"FAERDIG: {output}")
        return output
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        write_report(output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description="Stack AstroCam DNG-optagelser uden at aendre originalerne")
    parser.add_argument("--input", required=True, type=Path, help="Mappe med en enkelt optageserie")
    parser.add_argument("--output", type=Path, help="Ny outputmappe; skal endnu ikke eksistere")
    parser.add_argument("--max-sky-ratio", type=float, default=1.15, help="Maksimal medianbaggrund relativt til det moerkeste billede (standard 1.15)")
    parser.add_argument("--selection-report", type=Path, help="Genbrug praecis samme RAW-udvalg fra en tidligere rapport.json")
    parser.add_argument("--legacy", action="store_true", help="Brug den tidligere v0.3-metode")
    args = parser.parse_args()
    if not math.isfinite(args.max_sky_ratio) or args.max_sky_ratio < 1:
        parser.error("--max-sky-ratio skal vaere et endeligt tal >= 1")
    try:
        if args.legacy:
            run(args)
        else:
            from astrocam_advanced import run_advanced
            run_advanced(args)
    except KeyboardInterrupt:
        log("Afbrudt. Originalerne er bevaret.")
        return 130
    except Exception as error:
        log(f"FEJL: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
