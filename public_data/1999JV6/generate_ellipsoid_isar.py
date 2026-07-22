import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np
import trimesh
from tqdm import tqdm


DEFAULT_DATA_DIR = Path(__file__).resolve().parent


# ========================== User settings ==========================
# All lengths are in meters. Edit these values directly before running.
DATA_DIR = DEFAULT_DATA_DIR
META_NAME = "radar_meta.json"
OUTPUT_IMAGE_DIR = "image_ellipsoid"
ELLIPSOID_META_NAME = "ellipsoid_meta.json"
OVERVIEW_NAME = "ellipsoid_overview.png"
MESH_NAME = "ellipsoid_model.ply"

ELLIPSOID_A = 24.0
ELLIPSOID_B = 12.0
ELLIPSOID_C = 12.0
ELLIPSOID_CENTER = [0.0, 0.0, 0.0]
ROTATION_AXIS = [0.0, 0.0, 1.0]

SUBDIVISIONS = 5
N_SIDELOBES = 1
NORMALIZATION_PERCENTILE = 99.9
SKIP_MESH_EXPORT = False
# ==================================================================


def normalize_vector(value, name):
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise ValueError("{} must be a 3-vector, got shape {}".format(name, vector.shape))
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("{} must be non-zero".format(name))
    return vector / norm


def axis_spacing(frame, axis_key, spacing_key):
    if spacing_key in frame:
        return float(frame[spacing_key])
    axis = np.asarray(frame[axis_key], dtype=np.float64).reshape(-1)
    if axis.size < 2:
        raise ValueError("{} needs at least two samples to infer spacing".format(axis_key))
    return float(np.mean(np.diff(axis)))


def scalar_frame_value(frame, key):
    value = np.asarray(frame[key], dtype=np.float64).reshape(-1)
    if value.size != 1:
        raise ValueError("{} must be scalar".format(key))
    return float(value[0])


def create_ellipsoid_mesh(a, b, c, subdivisions, center):
    if min(a, b, c) <= 0.0:
        raise ValueError("Ellipsoid axes a, b, c must be positive")
    mesh = trimesh.creation.icosphere(subdivisions=int(subdivisions), radius=1.0)
    mesh.vertices[:, 0] *= float(a)
    mesh.vertices[:, 1] *= float(b)
    mesh.vertices[:, 2] *= float(c)
    mesh.apply_translation(np.asarray(center, dtype=np.float64) - mesh.centroid)
    mesh.fix_normals()
    return mesh


def projected_triangle_area(points):
    edge1 = points[1] - points[0]
    edge2 = points[2] - points[0]
    return 0.5 * abs(edge1[0] * edge2[1] - edge1[1] * edge2[0])

def write_png(path, image):
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(str(path))


def polygon_mask(points_xy, shape):
    height, width = shape
    image = Image.new("L", (width, height), 0)
    polygon = [(int(x), int(y)) for x, y in points_xy]
    ImageDraw.Draw(image).polygon(polygon, outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8)


def convolve2d_same(image, kernel):
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    output = np.zeros_like(image, dtype=np.float32)
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            patch = padded[y:y + kh, x:x + kw]
            output[y, x] = float(np.sum(patch * kernel))
    return output


def apply_sinc_psf(image, range_resolution, range_grid_spacing,
                   doppler_resolution, doppler_grid_spacing, n_sidelobes):
    range_res_pix = range_resolution / range_grid_spacing
    doppler_res_pix = doppler_resolution / doppler_grid_spacing
    if range_res_pix <= 0.0 or doppler_res_pix <= 0.0:
        raise ValueError("Resolution/grid spacing ratio must be positive")

    r_half = int(np.ceil(float(n_sidelobes) * range_res_pix))
    d_half = int(np.ceil(float(n_sidelobes) * doppler_res_pix))
    r_coords = np.arange(-r_half, r_half + 1, dtype=np.float64) / range_res_pix
    d_coords = np.arange(-d_half, d_half + 1, dtype=np.float64) / (doppler_res_pix + 1e-8)

    kernel = np.outer(np.sinc(r_coords), np.sinc(d_coords)) ** 2
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        return image
    kernel = (kernel / kernel_sum).astype(np.float32)
    return convolve2d_same(image, kernel)


def render_ellipsoid_frame(mesh, frame, rotation_axis, n_sidelobes):
    height, width = [int(v) for v in frame["image_size"]]
    radar_los = normalize_vector(frame["radar_los"], "radar_los")
    rotation_axis = normalize_vector(rotation_axis, "rotation_axis")

    rotation_period = scalar_frame_value(frame, "rotation_period")
    radar_frequency = scalar_frame_value(frame, "radar_frequency")
    range_resolution = scalar_frame_value(frame, "range_resolution")
    doppler_resolution = scalar_frame_value(frame, "doppler_resolution")
    range_grid_spacing = axis_spacing(frame, "range_axis", "range_grid_spacing")
    doppler_grid_spacing = axis_spacing(frame, "doppler_axis", "doppler_grid_spacing")

    c0 = 299792458.0
    wavelength = c0 / radar_frequency
    omega_vec = rotation_axis * (2.0 * math.pi / rotation_period)

    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)

    incidence_cos = np.sum(normals * radar_los[None, :], axis=-1)
    visible_mask = incidence_cos > 0.0

    visible_cos = incidence_cos[visible_mask]
    visible_areas = face_areas[visible_mask]
    visible_triangles = triangles[visible_mask]
    scatter_intensity = visible_areas * (visible_cos ** 2)

    image = np.zeros((height, width), dtype=np.float32)
    if visible_triangles.shape[0] == 0:
        return image

    range_coord_v = -np.sum(visible_triangles * radar_los[None, None, :], axis=-1)
    range_bin_v = range_coord_v / range_grid_spacing + 0.5 * (height - 1)

    velocity_v = np.cross(omega_vec[None, None, :], visible_triangles)
    doppler_coord_v = (2.0 / wavelength) * np.sum(
        velocity_v * radar_los[None, None, :], axis=-1
    )
    doppler_bin_v = doppler_coord_v / doppler_grid_spacing + 0.5 * (width - 1)

    points_2d = np.stack([doppler_bin_v, range_bin_v], axis=-1)
    for points, intensity in zip(points_2d, scatter_intensity):
        area_2d = projected_triangle_area(points)
        if area_2d < 0.5:
            r_idx = int(round(float(np.mean(points[:, 1]))))
            d_idx = int(round(float(np.mean(points[:, 0]))))
            if 0 <= r_idx < height and 0 <= d_idx < width:
                image[r_idx, d_idx] += float(intensity)
            continue

        points_int = np.round(points).astype(np.int32)
        x_min, y_min = np.min(points_int, axis=0)
        x_max, y_max = np.max(points_int, axis=0)
        if x_max < 0 or x_min >= width or y_max < 0 or y_min >= height:
            continue

        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(width - 1, x_max), min(height - 1, y_max)
        if x_max < x_min or y_max < y_min:
            continue

        local_points = points_int - np.array([x_min, y_min], dtype=np.int32)
        mask = polygon_mask(local_points, (y_max - y_min + 1, x_max - x_min + 1))
        pixels_covered = int(mask.sum())
        if pixels_covered > 0:
            image[y_min:y_max + 1, x_min:x_max + 1] += mask * (float(intensity) / pixels_covered)

    image = apply_sinc_psf(
        image,
        range_resolution=range_resolution,
        range_grid_spacing=range_grid_spacing,
        doppler_resolution=doppler_resolution,
        doppler_grid_spacing=doppler_grid_spacing,
        n_sidelobes=n_sidelobes,
    )
    return np.clip(image, 0.0, None).astype(np.float32)


def save_overview(images, output_path, cols=9):
    if len(images) == 0:
        return
    rows = int(math.ceil(len(images) / float(cols)))
    tile_shape = images[0].shape
    blank = np.zeros(tile_shape, dtype=np.uint8)
    tiles = list(images) + [blank] * (rows * cols - len(images))
    overview = np.vstack([
        np.hstack(tiles[row * cols:(row + 1) * cols])
        for row in range(rows)
    ])
    write_png(output_path, overview)


def load_frames(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta.get("frames")
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("{} must contain a non-empty frames list".format(meta_path))
    return frames


def generate_dataset():
    data_dir = Path(DATA_DIR).resolve()
    meta_path = data_dir / META_NAME
    output_image_dir = data_dir / OUTPUT_IMAGE_DIR
    ellipsoid_meta_path = data_dir / ELLIPSOID_META_NAME
    overview_path = data_dir / OVERVIEW_NAME
    mesh_path = data_dir / MESH_NAME

    frames = load_frames(meta_path)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    center = [float(v) for v in ELLIPSOID_CENTER]
    mesh = create_ellipsoid_mesh(
        ELLIPSOID_A, ELLIPSOID_B, ELLIPSOID_C,
        subdivisions=SUBDIVISIONS,
        center=center,
    )
    if not SKIP_MESH_EXPORT:
        mesh.export(str(mesh_path))

    frame_images = []
    frame_stats = []
    print("Generating ellipsoid ISAR images from {}".format(meta_path))
    print("Axes: a={:.6g} m, b={:.6g} m, c={:.6g} m".format(
        ELLIPSOID_A, ELLIPSOID_B, ELLIPSOID_C
    ))
    for frame in tqdm(frames):
        file_name = os.path.basename(frame["file"])
        image = render_ellipsoid_frame(
            mesh,
            frame,
            rotation_axis=ROTATION_AXIS,
            n_sidelobes=N_SIDELOBES,
        )
        frame_images.append((file_name, image))
        frame_stats.append({
            "file": file_name,
            "min": float(np.min(image)),
            "max": float(np.max(image)),
            "sum": float(np.sum(image)),
        })

    stack = np.stack([image for _, image in frame_images], axis=0).astype(np.float32)
    global_scale = float(np.percentile(stack, NORMALIZATION_PERCENTILE))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = float(np.max(stack))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        raise ValueError("Generated ellipsoid image stack is empty")

    expected_files = {file_name for file_name, _ in frame_images}
    overview_tiles = []
    for file_name, image in frame_images:
        image_u8 = np.clip(image / global_scale * 255.0, 0.0, 255.0).astype(np.uint8)
        output_path = output_image_dir / file_name
        write_png(output_path, image_u8)
        overview_tiles.append(image_u8)

    save_overview(overview_tiles, overview_path)

    extra_pngs = sorted(
        path.name for path in output_image_dir.glob("*.png")
        if path.name not in expected_files
    )
    if extra_pngs:
        print("Warning: extra PNG files in {} may be read by the dataset: {}".format(
            output_image_dir, ", ".join(extra_pngs)
        ))

    ellipsoid_meta = {
        "source_meta": META_NAME,
        "image_dir": OUTPUT_IMAGE_DIR,
        "axes_m": {
            "a": float(ELLIPSOID_A),
            "b": float(ELLIPSOID_B),
            "c": float(ELLIPSOID_C),
        },
        "center_m": center,
        "rotation_axis": [float(v) for v in ROTATION_AXIS],
        "subdivisions": int(SUBDIVISIONS),
        "mesh_file": None if SKIP_MESH_EXPORT else MESH_NAME,
        "noise": False,
        "normalization": {
            "type": "global_percentile",
            "percentile": float(NORMALIZATION_PERCENTILE),
            "scale": float(global_scale),
        },
        "n_sidelobes": int(N_SIDELOBES),
        "frame_count": len(frame_images),
        "frames": frame_stats,
    }
    with open(ellipsoid_meta_path, "w", encoding="utf-8") as f:
        json.dump(ellipsoid_meta, f, indent=4)

    print("Saved images : {}".format(output_image_dir))
    print("Saved meta   : {}".format(ellipsoid_meta_path))
    print("Saved overview: {}".format(overview_path))
    if not SKIP_MESH_EXPORT:
        print("Saved mesh   : {}".format(mesh_path))
    print("Global scale : {:.6g}".format(global_scale))


if __name__ == "__main__":
    generate_dataset()



