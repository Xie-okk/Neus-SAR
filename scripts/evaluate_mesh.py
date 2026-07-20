import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_POINTS = 100000
PROJECTION_SIZE = 900
RANDOM_SEED = 20260720

# Edit only this block when different experiments use different mesh iterations.
DEFAULT_GT_MESH = './public_data/1999JV6/1999 JV6 Radar_60.ply'
EVAL_CASES = [
    {
        'case': '1999JV6',
        'pred_mesh': './exp/1999JV6/meshes/mesh_001000.ply',
        'gt_mesh': DEFAULT_GT_MESH,
    },
    {
        'case': '1999JV6_snr00',
        'pred_mesh': './exp/1999JV6_snr00/meshes/mesh_001000.ply',
        'gt_mesh': DEFAULT_GT_MESH,
    },
    {
        'case': '1999JV6_snr05',
        'pred_mesh': './exp/1999JV6_snr05/meshes/mesh_001000.ply',
        'gt_mesh': DEFAULT_GT_MESH,
    },
    {
        'case': '1999JV6_snr10',
        'pred_mesh': './exp/1999JV6_snr10/meshes/mesh_001000.ply',
        'gt_mesh': DEFAULT_GT_MESH,
    },
]

MASK_VIEWS = (
    ('xy', 0, 1, 'X', 'Y'),
    ('xz', 0, 2, 'X', 'Z'),
    ('yz', 1, 2, 'Y', 'Z'),
)

SUMMARY_FIELDS = [
    'case',
    'pred_mesh',
    'gt_mesh',
    'rank_score',
    'chamfer_l1',
    'rmse_symmetric',
    'hausdorff_p95',
    'projection_iou_mean',
    'projection_missing_mean',
    'projection_excess_mean',
    'bbox_mean_relative_length_error',
    'bbox_center_distance',
    'x_length_ratio',
    'y_length_ratio',
    'z_length_ratio',
]


METRIC_INFO = {
    'rank_score': ('score', 'lower', 'Min-max normalized combined score across evaluated cases'),
    'chamfer_l1': ('mesh unit', 'lower', 'Symmetric mean nearest-surface distance'),
    'rmse_symmetric': ('mesh unit', 'lower', 'Symmetric RMS nearest-surface distance'),
    'hausdorff_p95': ('mesh unit', 'lower', 'Worst directional 95th percentile surface distance'),
    'projection_iou_mean': ('ratio', 'higher', 'Mean silhouette IoU over XY/XZ/YZ projections'),
    'projection_missing_mean': ('gt area ratio', 'lower', 'Mean GT-only silhouette area, measures missing reconstruction'),
    'projection_excess_mean': ('gt area ratio', 'lower', 'Mean pred-only silhouette area, measures excess reconstruction'),
    'bbox_mean_relative_length_error': ('ratio', 'lower', 'Mean relative X/Y/Z bounding-box length error'),
    'bbox_center_distance': ('mesh unit', 'lower', 'Euclidean distance between predicted and GT bbox centers'),
    'x_length_ratio': ('ratio', 'near 1', 'Predicted X bbox length divided by GT X bbox length'),
    'y_length_ratio': ('ratio', 'near 1', 'Predicted Y bbox length divided by GT Y bbox length'),
    'z_length_ratio': ('ratio', 'near 1', 'Predicted Z bbox length divided by GT Z bbox length'),
}


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_mesh(path):
    mesh = trimesh.load(str(path), force='mesh')
    if mesh.is_empty:
        raise ValueError(f'Empty mesh: {path}')
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f'Mesh has no triangles: {path}')
    return mesh


def shared_half_extent(meshes, padding=1.05):
    coordinates = np.concatenate([
        np.asarray(mesh.vertices, dtype=np.float64)
        for mesh in meshes
    ], axis=0)
    half_extent = float(np.max(np.abs(coordinates))) * float(padding)
    return max(half_extent, 1e-6)


def mesh_axis_stats(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    min_xyz = vertices.min(axis=0)
    max_xyz = vertices.max(axis=0)
    length_xyz = max_xyz - min_xyz
    center_xyz = 0.5 * (min_xyz + max_xyz)
    return min_xyz, max_xyz, length_xyz, center_xyz


def compute_bbox_metrics(pred_mesh, gt_mesh):
    pred_min, pred_max, pred_len, pred_center = mesh_axis_stats(pred_mesh)
    gt_min, gt_max, gt_len, gt_center = mesh_axis_stats(gt_mesh)
    metrics = {}
    rel_errors = []
    for axis_index, axis_name in enumerate(('x', 'y', 'z')):
        gt_length = gt_len[axis_index]
        pred_length = pred_len[axis_index]
        length_ratio = pred_length / gt_length if gt_length != 0.0 else np.nan
        rel_error = abs(length_ratio - 1.0) if np.isfinite(length_ratio) else np.nan
        rel_errors.append(rel_error)
        metrics[f'{axis_name}_gt_min'] = gt_min[axis_index]
        metrics[f'{axis_name}_gt_max'] = gt_max[axis_index]
        metrics[f'{axis_name}_gt_length'] = gt_length
        metrics[f'{axis_name}_pred_min'] = pred_min[axis_index]
        metrics[f'{axis_name}_pred_max'] = pred_max[axis_index]
        metrics[f'{axis_name}_pred_length'] = pred_length
        metrics[f'{axis_name}_length_ratio'] = length_ratio
    metrics['bbox_center_distance'] = float(np.linalg.norm(pred_center - gt_center))
    metrics['bbox_mean_relative_length_error'] = float(np.nanmean(rel_errors))
    return metrics


def sample_surface_points(mesh, sample_points, seed_offset):
    np.random.seed(RANDOM_SEED + seed_offset)
    samples, _ = trimesh.sample.sample_surface(mesh, sample_points)
    return np.asarray(samples, dtype=np.float64)


def compute_surface_distance_metrics(pred_mesh, gt_mesh, sample_points):
    pred_samples = sample_surface_points(pred_mesh, sample_points, 1)
    gt_samples = sample_surface_points(gt_mesh, sample_points, 2)
    pred_tree = cKDTree(pred_samples)
    gt_tree = cKDTree(gt_samples)
    pred_to_gt, _ = gt_tree.query(pred_samples)
    gt_to_pred, _ = pred_tree.query(gt_samples)
    return {
        'chamfer_l1': float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean())),
        'rmse_symmetric': float(np.sqrt(0.5 * (np.mean(pred_to_gt ** 2) + np.mean(gt_to_pred ** 2)))),
        'hausdorff_p95': float(max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95))),
    }


def rasterize_projection(mesh, horizontal_axis, vertical_axis, half_extent,
                         image_size=PROJECTION_SIZE, padding=24):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    projected = vertices[:, [horizontal_axis, vertical_axis]]
    usable_size = image_size - 1 - 2 * padding
    scale = usable_size / (2.0 * half_extent)

    pixels = np.empty_like(projected)
    pixels[:, 0] = (projected[:, 0] + half_extent) * scale + padding
    pixels[:, 1] = (half_extent - projected[:, 1]) * scale + padding
    triangles = np.rint(pixels[np.asarray(mesh.faces, dtype=np.int64)]).astype(np.int32)

    mask_image = Image.new('L', (image_size, image_size), color=0)
    draw = ImageDraw.Draw(mask_image)
    for triangle in triangles:
        draw.polygon([tuple(point) for point in triangle], fill=255)
    return np.asarray(mask_image, dtype=np.uint8)


def mask_boundary(mask):
    binary = mask > 0
    padded = np.pad(binary, 1, mode='constant', constant_values=False)
    eroded = np.ones_like(binary)
    for row_offset in range(3):
        for col_offset in range(3):
            eroded &= padded[
                row_offset:row_offset + binary.shape[0],
                col_offset:col_offset + binary.shape[1],
            ]
    boundary = (binary & ~eroded).astype(np.uint8) * 255
    boundary_image = Image.fromarray(boundary, mode='L')
    return np.asarray(boundary_image.filter(ImageFilter.MaxFilter(3))) > 0


def projection_overlay(gt_mask, pred_mask):
    gt = gt_mask > 0
    pred = pred_mask > 0
    overlap = gt & pred
    gt_only = gt & ~pred
    pred_only = pred & ~gt

    image = np.full((*gt_mask.shape, 3), 255, dtype=np.uint8)
    image[overlap] = np.array([190, 190, 190], dtype=np.uint8)
    image[gt_only] = np.array([55, 126, 184], dtype=np.uint8)
    image[pred_only] = np.array([228, 26, 28], dtype=np.uint8)

    image[mask_boundary(gt_mask)] = np.array([0, 70, 170], dtype=np.uint8)
    image[mask_boundary(pred_mask)] = np.array([190, 0, 0], dtype=np.uint8)
    return image


def projection_masks(pred_mesh, gt_mesh, half_extent, image_size=PROJECTION_SIZE):
    masks = []
    for view_name, horizontal_axis, vertical_axis, _, _ in MASK_VIEWS:
        gt_mask = rasterize_projection(gt_mesh, horizontal_axis, vertical_axis, half_extent, image_size=image_size)
        pred_mask = rasterize_projection(pred_mesh, horizontal_axis, vertical_axis, half_extent, image_size=image_size)
        masks.append((view_name, gt_mask, pred_mask))
    return masks


def compute_projection_metrics(masks):
    metrics = {}
    ious = []
    missings = []
    excesses = []
    for view_name, gt_mask, pred_mask in masks:
        gt = gt_mask > 0
        pred = pred_mask > 0
        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        gt_area = max(int(gt.sum()), 1)
        iou = intersection / union if union > 0 else 0.0
        missing = np.logical_and(gt, ~pred).sum() / gt_area
        excess = np.logical_and(pred, ~gt).sum() / gt_area
        metrics[f'projection_iou_{view_name}'] = float(iou)
        metrics[f'projection_missing_{view_name}'] = float(missing)
        metrics[f'projection_excess_{view_name}'] = float(excess)
        ious.append(iou)
        missings.append(missing)
        excesses.append(excess)
    metrics['projection_iou_mean'] = float(np.mean(ious))
    metrics['projection_missing_mean'] = float(np.mean(missings))
    metrics['projection_excess_mean'] = float(np.mean(excesses))
    return metrics


def add_overlay_axis(axis, overlay, view_name, case_name, half_extent):
    axis.imshow(
        overlay,
        extent=(-half_extent, half_extent, -half_extent, half_extent),
        origin='upper',
    )
    axis.axhline(0.0, color='black', linewidth=0.5, alpha=0.35)
    axis.axvline(0.0, color='black', linewidth=0.5, alpha=0.35)
    axis.set_xlim(-half_extent, half_extent)
    axis.set_ylim(-half_extent, half_extent)
    axis.set_aspect('equal')
    axis.set_title(f'{case_name} | {view_name.upper()}', fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def save_overlay_views(masks, output_path, half_extent, case_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for axis, (view_name, gt_mask, pred_mask) in zip(axes, masks):
        add_overlay_axis(axis, projection_overlay(gt_mask, pred_mask), view_name, case_name, half_extent)
    add_overlay_legend(fig)
    fig.suptitle('Mesh orthographic silhouette overlay', fontsize=15)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def add_overlay_legend(fig):
    legend = (
        Patch(facecolor='#bebebe', label='Overlap'),
        Patch(facecolor='#377eb8', label='GT only / missing'),
        Patch(facecolor='#e41a1c', label='Prediction only / excess'),
    )
    fig.legend(handles=legend, loc='lower center', ncol=3, frameon=False)


def metric_rows(metrics):
    rows = []
    for key in SUMMARY_FIELDS[3:]:
        unit, better, description = METRIC_INFO.get(key, ('', '', ''))
        rows.append({
            'metric': key,
            'value': metrics.get(key, np.nan),
            'unit': unit,
            'better': better,
            'description': description,
        })
    for axis in ('x', 'y', 'z'):
        for part in ('gt_min', 'gt_max', 'gt_length', 'pred_min', 'pred_max', 'pred_length'):
            key = f'{axis}_{part}'
            rows.append({
                'metric': key,
                'value': metrics.get(key, np.nan),
                'unit': 'mesh unit',
                'better': 'reference',
                'description': f'{axis.upper()} axis {part.replace("_", " ")}',
            })
    return rows


def write_case_metrics_csv(metrics, output_path):
    fieldnames = ['metric', 'value', 'unit', 'better', 'description']
    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows(metrics))


def write_summary_csv(results, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in sorted(results, key=lambda x: x['metrics'].get('rank_score', np.inf)):
            row = {field: result['metrics'].get(field, '') for field in SUMMARY_FIELDS}
            row['case'] = result['case']
            row['pred_mesh'] = result['pred_path']
            row['gt_mesh'] = result['gt_path']
            writer.writerow(row)


def normalized(values, higher_is_better=False):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or np.allclose(values.max(), values.min()):
        return np.zeros_like(values)
    scores = (values - values.min()) / (values.max() - values.min())
    if higher_is_better:
        scores = 1.0 - scores
    return scores


def add_rank_scores(results):
    cd = normalized([r['metrics']['chamfer_l1'] for r in results])
    rmse = normalized([r['metrics']['rmse_symmetric'] for r in results])
    iou = normalized([r['metrics']['projection_iou_mean'] for r in results], higher_is_better=True)
    bbox = normalized([r['metrics']['bbox_mean_relative_length_error'] for r in results])
    scores = 0.30 * cd + 0.25 * rmse + 0.25 * iou + 0.20 * bbox
    for result, score in zip(results, scores):
        result['metrics']['rank_score'] = float(score)


def save_metrics_summary_plot(results, output_path):
    ordered = sorted(results, key=lambda x: x['metrics'].get('rank_score', np.inf))
    cases = [r['case'] for r in ordered]
    plot_specs = [
        ('chamfer_l1', 'Chamfer L1 (lower is better)'),
        ('rmse_symmetric', 'RMSE (lower is better)'),
        ('projection_iou_mean', 'Projection IoU (higher is better)'),
        ('bbox_mean_relative_length_error', 'BBox length error (lower is better)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, (metric, title) in zip(axes.ravel(), plot_specs):
        values = [r['metrics'][metric] for r in ordered]
        axis.bar(cases, values, color='#4c78a8')
        axis.set_title(title)
        axis.tick_params(axis='x', rotation=25)
        axis.grid(axis='y', alpha=0.25)
    fig.suptitle('Mesh metric summary', fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_overlay_summary_plot(results, output_path, half_extent):
    ordered = sorted(results, key=lambda x: x['metrics'].get('rank_score', np.inf))
    fig, axes = plt.subplots(len(ordered), 3, figsize=(15, max(3.2, 3.0 * len(ordered))))
    if len(ordered) == 1:
        axes = np.asarray([axes])
    for row_index, result in enumerate(ordered):
        for col_index, (view_name, gt_mask, pred_mask) in enumerate(result['masks']):
            overlay = projection_overlay(gt_mask, pred_mask)
            add_overlay_axis(axes[row_index, col_index], overlay, view_name, result['case'], half_extent)
    add_overlay_legend(fig)
    fig.suptitle('Mesh overlay summary', fontsize=15)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def evaluate_case(case_config, half_extent):
    case_name = case_config['case']
    pred_path = resolve_path(case_config['pred_mesh'])
    gt_path = resolve_path(case_config['gt_mesh'])
    pred_mesh = load_mesh(pred_path)
    gt_mesh = load_mesh(gt_path)

    metrics = {}
    metrics.update(compute_surface_distance_metrics(pred_mesh, gt_mesh, SAMPLE_POINTS))
    metrics.update(compute_bbox_metrics(pred_mesh, gt_mesh))
    masks = projection_masks(pred_mesh, gt_mesh, half_extent, image_size=PROJECTION_SIZE)
    metrics.update(compute_projection_metrics(masks))

    log_dir = PROJECT_ROOT / 'exp' / case_name / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    overlay_path = log_dir / 'three_view_overlay.png'
    save_overlay_views(masks, overlay_path, half_extent, case_name)

    return {
        'case': case_name,
        'pred_path': str(pred_path),
        'gt_path': str(gt_path),
        'log_dir': log_dir,
        'overlay_path': overlay_path,
        'metrics': metrics,
        'masks': masks,
    }


def main():
    loadable_cases = []
    meshes_for_extent = []
    for case_config in EVAL_CASES:
        pred_path = resolve_path(case_config['pred_mesh'])
        gt_path = resolve_path(case_config['gt_mesh'])
        if not pred_path.exists():
            print(f'Skip {case_config["case"]}: missing pred mesh {pred_path}')
            continue
        if not gt_path.exists():
            print(f'Skip {case_config["case"]}: missing GT mesh {gt_path}')
            continue
        pred_mesh = load_mesh(pred_path)
        gt_mesh = load_mesh(gt_path)
        loadable_cases.append(case_config)
        meshes_for_extent.extend([pred_mesh, gt_mesh])

    if not loadable_cases:
        raise FileNotFoundError('No configured mesh pairs were found. Edit EVAL_CASES in scripts/evaluate_mesh.py.')

    half_extent = shared_half_extent(meshes_for_extent)
    results = [evaluate_case(case_config, half_extent) for case_config in loadable_cases]
    add_rank_scores(results)

    for result in results:
        metrics_path = result['log_dir'] / 'mesh_metrics.csv'
        write_case_metrics_csv(result['metrics'], metrics_path)
        print(f'{result["case"]}: metrics saved to {metrics_path}')
        print(f'{result["case"]}: overlay saved to {result["overlay_path"]}')

    summary_csv = PROJECT_ROOT / 'exp' / 'mesh_metrics_summary.csv'
    summary_png = PROJECT_ROOT / 'exp' / 'mesh_metrics_summary.png'
    overlay_png = PROJECT_ROOT / 'exp' / 'mesh_overlay_summary.png'
    write_summary_csv(results, summary_csv)
    save_metrics_summary_plot(results, summary_png)
    save_overlay_summary_plot(results, overlay_png, half_extent)
    print(f'Summary CSV saved to {summary_csv}')
    print(f'Summary plot saved to {summary_png}')
    print(f'Overlay summary saved to {overlay_png}')


if __name__ == '__main__':
    main()
