import argparse
import csv
import os
from glob import glob

import cv2 as cv
import numpy as np


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        img = np.load(path)
    elif ext == '.npz':
        data = np.load(path)
        key = 'image' if 'image' in data else ('isar' if 'isar' in data else data.files[0])
        img = data[key]
    else:
        img = cv.imread(path, cv.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f'Failed to load image: {path}')
        if img.ndim == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.integer):
        img = img / np.iinfo(img.dtype).max
    elif img.size > 0 and np.nanmax(img) > 1.5:
        img = img / 255.0
    img = img.astype(np.float32, copy=False)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(img, 0.0, None)


def find_images(image_dir):
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.npy', '*.npz']
    paths = []
    for pattern in patterns:
        paths.extend(glob(os.path.join(image_dir, pattern)))
    return sorted(paths)


def border_mask(shape, border_frac):
    h, w = shape
    border = max(1, int(round(min(h, w) * border_frac)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return mask, border


def classify_image(img, border_frac=0.1, bg_quantile=0.90, near_quantile=0.995, strong_quantile=0.999):
    mask, border = border_mask(img.shape, border_frac)
    border_pixels = img[mask]
    if border_pixels.size == 0:
        raise ValueError('Empty border region')

    noise_floor = float(np.median(border_pixels))
    noise_sigma = float(1.4826 * np.median(np.abs(border_pixels - noise_floor)))
    low_thr = float(np.quantile(border_pixels, bg_quantile))
    high_thr = float(np.quantile(border_pixels, near_quantile))
    strong_thr = float(np.quantile(border_pixels, strong_quantile))

    background = img <= low_thr
    near = (img > low_thr) & (img <= high_thr)
    strong = img > strong_thr

    total = img.size
    bg_count = int(background.sum())
    near_count = int(near.sum())
    strong_count = int(strong.sum())

    labels = np.zeros(img.shape, dtype=np.uint8)
    labels[near] = 1
    labels[strong] = 2

    return {
        'noise_floor': noise_floor,
        'noise_sigma': noise_sigma,
        'low_thr': float(low_thr),
        'high_thr': float(high_thr),
        'strong_thr': float(strong_thr),
        'border_width': int(border),
        'background_count': bg_count,
        'near_count': near_count,
        'strong_count': strong_count,
        'background_ratio': bg_count / total,
        'near_ratio': near_count / total,
        'strong_ratio': strong_count / total,
        'max': float(img.max()),
        'mean': float(img.mean()),
        'labels': labels,
    }


def make_label_vis(img, labels):
    base = np.clip(img, 0.0, None)
    vmax = float(base.max())
    if vmax > 1e-8:
        base = base / vmax
    gray = (base * 255).astype(np.uint8)
    vis = np.stack([gray, gray, gray], axis=-1)

    vis[labels == 1] = np.array([0, 215, 255], dtype=np.uint8)   # near noise
    vis[labels == 2] = np.array([0, 0, 255], dtype=np.uint8)     # strong
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default=os.path.join('public_data', '1999JV6', 'image'))
    parser.add_argument('--out_csv', type=str, default=os.path.join('exp', '1999JV6', 'logs', 'noise_debug.csv'))
    parser.add_argument('--vis_dir', type=str, default=os.path.join('exp', '1999JV6', 'logs', 'noise_debug_vis'))
    parser.add_argument('--border_frac', type=float, default=0.1)
    parser.add_argument('--bg_quantile', type=float, default=0.90)
    parser.add_argument('--near_quantile', type=float, default=0.995)
    parser.add_argument('--strong_quantile', type=float, default=0.999)
    parser.add_argument('--max_images', type=int, default=None)
    args = parser.parse_args()

    paths = find_images(args.image_dir)
    if args.max_images is not None:
        paths = paths[:args.max_images]
    if len(paths) == 0:
        raise FileNotFoundError(f'No images found under {args.image_dir}')

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    overview_tiles = []
    overview_rows = 2
    overview_cols = 9

    fieldnames = [
        'file', 'noise_floor', 'noise_sigma', 'low_thr', 'high_thr', 'strong_thr', 'border_width',
        'background_count', 'near_count', 'strong_count',
        'background_ratio', 'near_ratio', 'strong_ratio', 'mean', 'max'
    ]

    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for path in paths:
            img = load_image(path)
            result = classify_image(
                img,
                border_frac=args.border_frac,
                bg_quantile=args.bg_quantile,
                near_quantile=args.near_quantile,
                strong_quantile=args.strong_quantile,
            )

            row = {k: result[k] for k in fieldnames if k != 'file'}
            row['file'] = os.path.basename(path)
            writer.writerow(row)

            vis = make_label_vis(img, result['labels'])
            out_path = os.path.join(args.vis_dir, os.path.basename(path))
            cv.imwrite(out_path, vis)
            overview_tiles.append(vis)

            print(
                f"{os.path.basename(path)} "
                f"floor={result['noise_floor']:.6g} sigma={result['noise_sigma']:.6g} "
                f"bg={result['background_ratio']:.3f} near={result['near_ratio']:.3f} strong={result['strong_ratio']:.3f}"
            )

    overview_path = os.path.join(os.path.dirname(args.out_csv) or '.', 'noise_debug_2x9.png')
    if len(overview_tiles) > 0:
        tile_h, tile_w = overview_tiles[0].shape[:2]
        blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tiles = overview_tiles[:overview_rows * overview_cols]
        tiles += [blank] * (overview_rows * overview_cols - len(tiles))
        overview = np.vstack([
            np.hstack(tiles[r * overview_cols:(r + 1) * overview_cols])
            for r in range(overview_rows)
        ])
        cv.imwrite(overview_path, overview)

    print(f'CSV saved to {args.out_csv}')
    print(f'Visuals saved to {args.vis_dir}')
    print(f'Overview saved to {overview_path}')


if __name__ == '__main__':
    main()
