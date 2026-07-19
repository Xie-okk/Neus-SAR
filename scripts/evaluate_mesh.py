import argparse
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff


def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    if mesh.is_empty:
        raise ValueError(f'Empty mesh: {path}')
    return mesh


def mesh_summary(mesh):
    verts = np.asarray(mesh.vertices)
    bounds = mesh.bounds
    return {
        'n_verts': int(len(verts)),
        'n_faces': int(len(mesh.faces)),
        'min': bounds[0].tolist(),
        'max': bounds[1].tolist(),
        'extent': (bounds[1] - bounds[0]).tolist(),
        'center': np.mean(verts, axis=0).tolist(),
        'volume': float(mesh.volume) if mesh.is_volume else None,
        'area': float(mesh.area),
    }


def sample_surface_points(mesh, sample_points):
    samples, _ = trimesh.sample.sample_surface_even(mesh, sample_points)
    return np.asarray(samples, dtype=np.float64)


def chamfer_and_hausdorff(samples_a, samples_b):
    tree_a = cKDTree(samples_a)
    tree_b = cKDTree(samples_b)

    dist_a_to_b, _ = tree_b.query(samples_a)
    dist_b_to_a, _ = tree_a.query(samples_b)

    chamfer_l1 = 0.5 * (np.mean(dist_a_to_b) + np.mean(dist_b_to_a))
    chamfer_l2 = 0.5 * (np.mean(dist_a_to_b ** 2) + np.mean(dist_b_to_a ** 2))
    rms = 0.5 * (np.sqrt(np.mean(dist_a_to_b ** 2)) + np.sqrt(np.mean(dist_b_to_a ** 2)))

    h_ab = directed_hausdorff(samples_a, samples_b)[0]
    h_ba = directed_hausdorff(samples_b, samples_a)[0]

    return {
        'chamfer_l1': float(chamfer_l1),
        'chamfer_l2': float(chamfer_l2),
        'chamfer_rms': float(rms),
        'hausdorff_ab': float(h_ab),
        'hausdorff_ba': float(h_ba),
        'hausdorff': float(max(h_ab, h_ba)),
    }


def box_metrics(mesh_a, mesh_b):
    min_a, max_a = mesh_a.bounds
    min_b, max_b = mesh_b.bounds
    extent_a = max_a - min_a
    extent_b = max_b - min_b

    center_a = mesh_a.vertices.mean(axis=0)
    center_b = mesh_b.vertices.mean(axis=0)

    volume_a = float(mesh_a.volume) if mesh_a.is_volume else None
    volume_b = float(mesh_b.volume) if mesh_b.is_volume else None

    metrics = {
        'center_distance': float(np.linalg.norm(center_a - center_b)),
        'extent_l1': float(np.mean(np.abs(extent_a - extent_b))),
        'extent_l2': float(np.sqrt(np.mean((extent_a - extent_b) ** 2))),
        'extent_rel_error': float(np.mean(np.abs(extent_a - extent_b) / np.maximum(extent_b, 1e-12))),
        'area_ratio': float(min(mesh_a.area, mesh_b.area) / max(mesh_a.area, mesh_b.area)),
        'area_difference': float(abs(mesh_a.area - mesh_b.area)),
    }

    if volume_a is not None and volume_b is not None and volume_a != 0.0 and volume_b != 0.0:
        metrics['volume_ratio'] = float(min(volume_a, volume_b) / max(volume_a, volume_b))
        metrics['volume_difference'] = float(abs(volume_a - volume_b))
    else:
        metrics['volume_ratio'] = None
        metrics['volume_difference'] = None

    return metrics


def _rows_to_void(arr):
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))).ravel()


def iou_voxel(mesh_a, mesh_b, pitch):
    if pitch <= 0:
        return {'iou_voxel': None, 'iou_error': 'pitch must be positive'}

    try:
        vox_a = mesh_a.voxelized(pitch).fill()
        vox_b = mesh_b.voxelized(pitch).fill()
    except Exception as e:
        return {'iou_voxel': None, 'iou_error': str(e)}

    pts_a = np.asarray(vox_a.points, dtype=np.float64)
    pts_b = np.asarray(vox_b.points, dtype=np.float64)
    if len(pts_a) == 0 or len(pts_b) == 0:
        return {'iou_voxel': None, 'iou_error': 'empty voxel set'}

    lo = np.minimum(pts_a.min(axis=0), pts_b.min(axis=0))
    idx_a = np.unique(np.round((pts_a - lo) / pitch).astype(np.int32), axis=0)
    idx_b = np.unique(np.round((pts_b - lo) / pitch).astype(np.int32), axis=0)

    a_view = _rows_to_void(idx_a)
    b_view = _rows_to_void(idx_b)
    inter = np.intersect1d(a_view, b_view).size
    union = np.union1d(a_view, b_view).size
    iou = float(inter / union) if union > 0 else None
    return {'iou_voxel': iou, 'iou_error': None}


def compare_meshes(pred_path, gt_path, sample_points=10000, iou_pitch=0.5):
    pred_mesh = load_mesh(pred_path)
    gt_mesh = load_mesh(gt_path)

    pred_summary = mesh_summary(pred_mesh)
    gt_summary = mesh_summary(gt_mesh)

    pred_samples = sample_surface_points(pred_mesh, sample_points)
    gt_samples = sample_surface_points(gt_mesh, sample_points)

    metrics = {}
    metrics.update(chamfer_and_hausdorff(pred_samples, gt_samples))
    metrics.update(box_metrics(pred_mesh, gt_mesh))

    metrics.update(iou_voxel(pred_mesh, gt_mesh, iou_pitch))

    return {
        'pred_path': pred_path,
        'gt_path': gt_path,
        'pred_summary': pred_summary,
        'gt_summary': gt_summary,
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, default=os.path.join('exp', '1999JV6', 'meshes', 'mesh_001000.ply'))
    parser.add_argument('--gt', type=str, default=os.path.join('public_data', '1999JV6', '1999 JV6 Radar_60.ply'))
    parser.add_argument('--sample_points', type=int, default=10000)
    parser.add_argument('--iou_pitch', type=float, default=None, help='Voxel pitch in physical units for optional IoU')
    parser.add_argument('--json_out', type=str, default=None)
    args = parser.parse_args()

    result = compare_meshes(args.pred, args.gt, sample_points=args.sample_points)

    print('pred:', result['pred_path'])
    print('gt:', result['gt_path'])
    print('pred_summary:', json.dumps(result['pred_summary'], ensure_ascii=False))
    print('gt_summary:', json.dumps(result['gt_summary'], ensure_ascii=False))
    print('metrics:')
    for key, value in result['metrics'].items():
        print(f'  {key}: {value}')

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
