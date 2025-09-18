import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

import cv2
import numpy as np
from tqdm import tqdm
try:
    from rembg import remove as rembg_remove, new_session as rembg_session
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

# -------------------- Configuration Defaults -------------------- #
FALLBACK_FG_MAX_RATIO = 0.85  # if mask covers more than this fraction, attempt stricter threshold
FALLBACK_FG_MIN_RATIO = 0.001 # if mask too tiny, relax
DEBUG_CANVAS_SCALE = 0.4

# -------------------- Mask / Crop Utilities -------------------- #

def load_image(path: Path, bayer_pattern: Optional[str] = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 2 and bayer_pattern:
        code_map = {
            'rg': cv2.COLOR_BayerRG2RGB,
            'gr': cv2.COLOR_BayerGR2RGB,
            'bg': cv2.COLOR_BayerBG2RGB,
            'gb': cv2.COLOR_BayerGB2RGB,
        }
        key = bayer_pattern.lower()
        if key in code_map:
            try:
                img = cv2.cvtColor(img, code_map[key])
            except Exception:
                pass  # Fall back to original if conversion fails
    # Convert to RGB if multi-channel BGR
    if img.ndim == 3 and img.shape[2] == 3:
        # Heuristic: assume OpenCV returned BGR when source had 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 2:
        # Keep grayscale as-is
        pass
    return img


def compute_foreground_mask(img: np.ndarray, channel: str = "auto", method: str = "otsu", invert: bool = True,
                             blur: int = 3, percentile: float = 0.995) -> np.ndarray:
    """Return uint8 mask in {0,255}. Assumes dark background by default.

    channel: 'r','g','b','luma','auto'
    method: 'otsu','percentile'
    invert: invert threshold assignment (useful for dark backgrounds)
    percentile: used when method == 'percentile'
    """
    if img.ndim == 3:
        if channel == 'auto':
            # pick channel with highest contrast (std dev)
            stds = [img[..., i].std() for i in range(3)]
            ch_idx = int(np.argmax(stds))
        elif channel in ['r','g','b']:
            ch_idx = {'r':0,'g':1,'b':2}[channel]
        elif channel == 'luma':
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ch = gray
            ch_idx = None
        else:
            raise ValueError("Invalid channel")
        if channel != 'luma':
            ch = img[..., ch_idx]
    else:
        ch = img

    if blur and blur > 1:
        k = max(1, blur)
        if k % 2 == 0:
            k += 1
        ch_blur = cv2.GaussianBlur(ch, (k, k), 0)
    else:
        ch_blur = ch

    if method == 'otsu':
        _thr, mask = cv2.threshold(ch_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'percentile':
        thresh_val = np.quantile(ch_blur.reshape(-1), percentile)
        _, mask = cv2.threshold(ch_blur, int(thresh_val), 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("Unsupported method")

    if invert:
        mask = 255 - mask

    return mask.astype(np.uint8)


def fallback_adjust_mask(img: np.ndarray, mask: np.ndarray, invert: bool, orig_method: str, percentile: float) -> np.ndarray:
    h, w = mask.shape
    fg_ratio = mask.sum() / (255.0 * h * w)
    # Too large foreground: tighten
    if fg_ratio > FALLBACK_FG_MAX_RATIO:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Use higher percentile or Canny edges intersection
        p_hi = min(0.9995, max(percentile, 0.995))
        thr_val = np.quantile(gray.reshape(-1), p_hi)
        _, m2 = cv2.threshold(gray, int(thr_val), 255, cv2.THRESH_BINARY)
        if invert:
            m2 = 255 - m2
        # Edge refinement: emphasize internal structure
        edges = cv2.Canny(gray, 50, 150)
        m2 = cv2.bitwise_and(m2, cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=1))
        # Fallback to original mask intersection
        combined = cv2.bitwise_and(mask, m2)
        if combined.sum() > 0.05 * mask.sum():
            mask = combined
    # Too small foreground: relax by lowering threshold if percentile method
    elif fg_ratio < FALLBACK_FG_MIN_RATIO:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        p_lo = min(0.99, percentile) if orig_method == 'percentile' else 0.99
        thr_val = np.quantile(gray.reshape(-1), p_lo)
        _, m2 = cv2.threshold(gray, int(thr_val), 255, cv2.THRESH_BINARY)
        if invert:
            m2 = 255 - m2
        mask = cv2.bitwise_or(mask, m2)
    return mask


def refine_mask(mask: np.ndarray, morph_kernel: int = 5, dilate: int = 2, erode: int = 0, keep_largest: bool = True) -> np.ndarray:
    m = mask.copy()
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    if erode > 0:
        m = cv2.erode(m, kernel, iterations=erode)
    if dilate > 0:
        m = cv2.dilate(m, kernel, iterations=dilate)

    if keep_largest:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            areas = [cv2.contourArea(c) for c in cnts]
            c = cnts[int(np.argmax(areas))]
            m2 = np.zeros_like(m)
            cv2.drawContours(m2, [c], -1, 255, thickness=cv2.FILLED)
            m = m2
    return m


def mask_to_bbox(mask: np.ndarray, pad: int = 8, square: bool = False, clamp: Optional[Tuple[int,int]] = None,
                 pad_rel: float = 0.0) -> Tuple[int,int,int,int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    if square:
        w_box = x1 - x0 + 1
        h_box = y1 - y0 + 1
        side = max(w_box, h_box)
        cx = (x0 + x1)//2
        cy = (y0 + y1)//2
        x0 = cx - side//2
        x1 = x0 + side - 1
        y0 = cy - side//2
        y1 = y0 + side - 1
    # Absolute padding
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    # Relative padding (percentage of size)
    if pad_rel > 0:
        w_box = x1 - x0 + 1
        h_box = y1 - y0 + 1
        add_w = int(w_box * pad_rel)
        add_h = int(h_box * pad_rel)
        x0 -= add_w; x1 += add_w
        y0 -= add_h; y1 += add_h
    h_full, w_full = mask.shape
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w_full-1, x1); y1 = min(h_full-1, y1)
    if clamp:
        max_w, max_h = clamp
        x1 = min(x1, max_w-1)
        y1 = min(y1, max_h-1)
    return int(x0), int(y0), int(x1), int(y1)


def apply_bbox(img: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    x0,y0,x1,y1 = bbox
    return img[y0:y1+1, x0:x1+1]

# -------------------- Processing Pipeline -------------------- #

def ai_segment(img: np.ndarray, model_name: str = 'u2net') -> np.ndarray:
    if not REMBG_AVAILABLE:
        raise RuntimeError("rembg not installed. Install with pip install rembg")
    session = rembg_session(model_name)
    # rembg expects RGB uint8
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = img
    result = rembg_remove(rgb, session=session, only_mask=True)
    # result is single-channel alpha 0..255
    return result


def process_image(path: Path, out_dir: Path, args, global_bbox: Optional[Tuple[int,int,int,int]] = None) -> Dict:
    img = load_image(path, getattr(args, 'bayer_pattern', None))
    if getattr(args, 'bbox_only', False):
        # Fast path: compute mask only to get bbox, optionally coarse threshold if ai_seg disabled
        if args.ai_seg:
            mask = ai_segment(img, args.ai_seg_model)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            mask = compute_foreground_mask(img, channel=args.channel, method=args.method, invert=args.invert,
                                           blur=args.blur, percentile=args.percentile)
            if args.enable_fallback:
                mask = fallback_adjust_mask(img, mask, args.invert, args.method, args.percentile)
        mask = refine_mask(mask, morph_kernel=args.morph_kernel, dilate=args.dilate, erode=args.erode, keep_largest=not args.no_keep_largest)
        bbox = mask_to_bbox(mask, pad=args.pad, square=args.square, pad_rel=args.pad_rel) if global_bbox is None else global_bbox
        cropped_img = apply_bbox(img, bbox)
        cropped_mask = apply_bbox(mask, bbox)
    else:
        if args.ai_seg:
            mask = ai_segment(img, args.ai_seg_model)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            mask = compute_foreground_mask(img, channel=args.channel, method=args.method, invert=args.invert,
                                           blur=args.blur, percentile=args.percentile)
            if args.enable_fallback:
                mask = fallback_adjust_mask(img, mask, args.invert, args.method, args.percentile)
        mask = refine_mask(mask, morph_kernel=args.morph_kernel, dilate=args.dilate, erode=args.erode, keep_largest=not args.no_keep_largest)
        bbox = mask_to_bbox(mask, pad=args.pad, square=args.square, pad_rel=args.pad_rel) if global_bbox is None else global_bbox
        cropped_img = apply_bbox(img, bbox)
        cropped_mask = apply_bbox(mask, bbox)

    stem = path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mask and not getattr(args, 'bbox_only', False):
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), cropped_mask)
    if args.save_rgba and not getattr(args, 'bbox_only', False):
        if cropped_img.ndim == 2:
            rgb = np.repeat(cropped_img[...,None], 3, axis=2)
        else:
            rgb = cropped_img
        if getattr(args, 'rgba_opaque', False):
            alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        else:
            alpha = cropped_mask
        rgba = np.dstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), alpha])
        cv2.imwrite(str(out_dir / f"{stem}_rgba.png"), rgba)
    if args.save_cropped or getattr(args, 'bbox_only', False):
        to_save = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR) if cropped_img.ndim == 3 else cropped_img
        cv2.imwrite(str(out_dir / f"{stem}_crop.png"), to_save)

    if args.debug_visual:
        debug_img = build_debug_visual(img, mask, bbox)
        cv2.imwrite(str(out_dir / f"{stem}_debug.png"), debug_img)

    meta = {
        'file': str(path.name),
        'width': int(img.shape[1]),
        'height': int(img.shape[0]),
        'bbox': {'x0':bbox[0],'y0':bbox[1],'x1':bbox[2],'y1':bbox[3]},
        'crop_width': int(bbox[2]-bbox[0]+1),
        'crop_height': int(bbox[3]-bbox[1]+1),
        'bbox_only': bool(getattr(args, 'bbox_only', False))
    }
    return meta


def build_debug_visual(img: np.ndarray, mask: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = rgb.copy()
    colored_mask = np.zeros_like(rgb)
    colored_mask[:,:,1] = mask  # green mask
    alpha = 0.4
    overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)
    x0,y0,x1,y1 = bbox
    cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,0,255), 2)
    # Side-by-side: original | overlay | cropped
    crop = apply_bbox(rgb, bbox)
    h = max(rgb.shape[0], overlay.shape[0], crop.shape[0])
    def pad(im):
        if im.shape[0] == h:
            return im
        pad_h = h - im.shape[0]
        return cv2.copyMakeBorder(im, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    concat = np.hstack([pad(rgb), pad(overlay), pad(crop)])
    return concat


def gather_images(scene_dir: Path, pattern: str = 'cam_*.tiff', patterns_raw: str = '', recursive: bool = False) -> List[Path]:
    """Flexible image discovery.

    patterns_raw: semicolon separated explicit glob patterns. If provided, only those globs are used.
    patterns_raw empty: gather ANY supported image extension. For backward compatibility we still try the
      legacy pattern inside capture_* folders first when not recursive.
    recursive: search subdirectories (rglob) vs only top-level / capture_*.
    Returns deduplicated, sorted list of file Paths.
    """
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    results: Set[Path] = set()

    def add_if_match(p: Path):
        if p.is_file() and p.suffix.lower() in exts:
            results.add(p)

    if patterns_raw:
        patterns = [p.strip() for p in patterns_raw.split(';') if p.strip()]
        for pat in patterns:
            if recursive:
                for found in scene_dir.rglob(pat):
                    add_if_match(found)
            else:
                for found in scene_dir.glob(pat):
                    add_if_match(found)
    else:
        # No explicit patterns: accept any supported extension.
        if recursive:
            for found in scene_dir.rglob('*'):
                add_if_match(found)
        else:
            # Try legacy capture_* subfolder structure first (any image file within).
            capture_dirs = sorted(d for d in scene_dir.glob('capture_*') if d.is_dir())
            if capture_dirs:
                for cdir in capture_dirs:
                    for found in cdir.iterdir():
                        add_if_match(found)
            # Also include images directly under scene_dir (any extension)
            for found in scene_dir.iterdir():
                add_if_match(found)
            # If legacy pattern produced nothing and legacy pattern argument supplied, we still allow
            # pattern-based matching as an additional fallback (covers user specifying custom pattern param).
            if not results and pattern:
                for found in scene_dir.glob(pattern):
                    add_if_match(found)
                for cdir in capture_dirs:
                    for found in cdir.glob(pattern):
                        add_if_match(found)

    return sorted(results)


def compute_union_bbox(img_paths: List[Path], args) -> Tuple[int,int,int,int]:
    x0_u, y0_u, x1_u, y1_u = 1e9, 1e9, -1, -1
    for p in tqdm(img_paths, desc='Union bbox pass'):
        img = load_image(p, getattr(args, 'bayer_pattern', None))
        mask = compute_foreground_mask(img, channel=args.channel, method=args.method, invert=args.invert,
                                       blur=args.blur, percentile=args.percentile)
        mask = refine_mask(mask, morph_kernel=args.morph_kernel, dilate=args.dilate, erode=args.erode, keep_largest=not args.no_keep_largest)
        x0,y0,x1,y1 = mask_to_bbox(mask, pad=args.pad, square=args.square, pad_rel=args.pad_rel)
        x0_u = min(x0_u, x0); y0_u = min(y0_u, y0)
        x1_u = max(x1_u, x1); y1_u = max(y1_u, y1)
    return int(x0_u), int(y0_u), int(x1_u), int(y1_u)


def process_scene(scene_dir: Path, out_root: Path, args):
    img_paths = gather_images(scene_dir, pattern=args.pattern)
    if not img_paths:
        print(f"No images found in {scene_dir}")
        return
    if args.max_images > 0:
        img_paths = img_paths[:args.max_images]
    scene_name = scene_dir.name
    out_dir = out_root / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)

    global_bbox = None
    if args.union_bbox:
        global_bbox = compute_union_bbox(img_paths, args)
        print(f"Union bbox for {scene_name}: {global_bbox}")

    metadata = []
    for p in tqdm(img_paths, desc=f"Processing {scene_name}"):
        meta = process_image(p, out_dir, args, global_bbox=global_bbox)
        metadata.append(meta)

    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump({'scene': scene_name, 'images': metadata, 'union_bbox': global_bbox}, f, indent=2)


# -------------------- CLI -------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description='Foreground crop & mask preprocessing for NeRF/Gaussian Splatting datasets.')
    ap.add_argument('--input-root', type=Path, required=True, help='Root directory containing scene folders.')
    ap.add_argument('--output-root', type=Path, required=True, help='Directory to write processed outputs.')
    ap.add_argument('--scenes', type=str, nargs='*', default=None, help='Subset of scene folder names to process; default all.')
    ap.add_argument('--pattern', type=str, default='cam_*.tiff', help='Glob pattern for images inside capture_* folders.')
    ap.add_argument('--channel', type=str, default='auto', choices=['auto','r','g','b','luma'])
    ap.add_argument('--method', type=str, default='otsu', choices=['otsu','percentile'])
    ap.add_argument('--percentile', type=float, default=0.995, help='Percentile threshold when method=percentile.')
    ap.add_argument('--invert', action='store_true', help='Invert mask (use if background darker than object).')
    ap.add_argument('--blur', type=int, default=3, help='Gaussian blur kernel size (odd).')
    ap.add_argument('--morph-kernel', type=int, default=5)
    ap.add_argument('--dilate', type=int, default=2)
    ap.add_argument('--erode', type=int, default=0)
    ap.add_argument('--no-keep-largest', action='store_true', help='Disable keeping only largest component.')
    ap.add_argument('--pad', type=int, default=8, help='Padding around bbox.')
    ap.add_argument('--square', action='store_true', help='Force square crop.')
    ap.add_argument('--pad-rel', type=float, default=0.0, help='Relative padding fraction (e.g. 0.1 adds 10% of bbox size each side).')
    ap.add_argument('--union-bbox', action='store_true', help='Compute and use union bbox across all images in scene.')
    ap.add_argument('--save-mask', action='store_true')
    ap.add_argument('--save-rgba', action='store_true')
    ap.add_argument('--save-cropped', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='List images & planned actions without writing files.')
    ap.add_argument('--max-images', type=int, default=0, help='Limit number of images per scene (0 = all).')
    ap.add_argument('--enable-fallback', action='store_true', help='Enable adaptive fallback if mask covers too much or too little.')
    ap.add_argument('--debug-visual', action='store_true', help='Save side-by-side debug visualization with bbox overlay.')
    ap.add_argument('--ai-seg', action='store_true', help='Use AI segmentation (rembg) instead of thresholding.')
    ap.add_argument('--ai-seg-model', type=str, default='u2net', help='rembg model name (u2net, u2netp, u2net_human_seg, etc).')
    ap.add_argument('--bayer-pattern', type=str, choices=['rg','gr','bg','gb'], help='Apply Bayer demosaic using given pattern if source TIFF is single-channel.')
    return ap.parse_args()


def main():
    args = parse_args()
    input_root: Path = args.input_root
    scenes = args.scenes if args.scenes else [d.name for d in input_root.iterdir() if d.is_dir()]

    if args.dry_run:
        print('Dry run: scenes to process ->', scenes)

    for scene_name in scenes:
        scene_dir = input_root / scene_name
        if not scene_dir.is_dir():
            print(f"Skip missing scene {scene_name}")
            continue
        if args.dry_run:
            imgs = gather_images(scene_dir, pattern=args.pattern)
            if args.max_images > 0:
                imgs = imgs[:args.max_images]
            print(f"Scene {scene_name}: {len(imgs)} images (showing first 5):")
            for p in imgs[:5]:
                print('  ', p)
            continue
        process_scene(scene_dir, args.output_root, args)

if __name__ == '__main__':
    main()

