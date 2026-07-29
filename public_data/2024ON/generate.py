import json
import os
from datetime import datetime, timezone

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import convolve
from tqdm import tqdm


if __name__ == "__main__":
    np.random.seed(42)

    # ===================== User parameters =====================
    OUTPUT_DIR = r"D:\A_master\AAA组会\AAAA研究方向\网络重建\NeuS-main\public_data\2024ON"
    TRUE_IMAGE_DIR = os.path.join(OUTPUT_DIR, "image")
    OVERVIEW_FILE = "compare.png"

    FRAME_COUNT = 54
    FALLBACK_IMAGE_SIZE = (78, 108)  # (range, azimuth), only used if true images are missing

    # Radar physical resolutions.
    RANGE_RESOLUTION = 3.75       # m
    DOPPLER_RESOLUTION = 0.1     # Hz

    # Image grid spacing. Set these manually; they do not need to equal resolution.
    RANGE_GRID_SPACING = RANGE_RESOLUTION/1     # m/pixel
    DOPPLER_GRID_SPACING = DOPPLER_RESOLUTION/1   # Hz/pixel

    # Radar and spin parameters.
    RADAR_FREQUENCY = 8.57e9       # Hz; change this if needed
    ROTATION_PERIOD = 6.0 * 3600  # s
    VIEW_SEQUENCE_DIRECTION = 1.0  # +1/-1; frame-to-frame view angle order
    ELEVATION_DEG = 0.0           # deg
    AZIMUTH_START_DEG = 45.0        # deg; tune this value manually

    # Time span read from the labelled 2024 ON collage: 2024 September 16, 00:27-05:50 UT.
    OBS_START_UTC = "2024-09-16T00:27:00Z"
    OBS_END_UTC = "2024-09-16T05:50:00Z"

    # Ellipsoid semi-axes in meters. These are radii, not diameters.
    # Full diameters are 2*a, 2*b, 2*c.
    ELLIPSOID_SEMI_AXES = (150.0, 75.0, 75.0)
    ELLIPSOID_SUBDIVISIONS = 3

    # Use None for clean ellipsoid preview; set a number such as 20.0 to add noise.
    NOISE_SNR_DB = None
    NOISE_SEED = 42

    # ===================== True image size =====================
    true_image_ext = ".png"
    first_true_image = None
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = os.path.join(TRUE_IMAGE_DIR, f"000{ext}")
        if os.path.exists(candidate):
            first_true_image = candidate
            true_image_ext = ext
            break

    if first_true_image is not None:
        with Image.open(first_true_image) as img:
            width, height = img.size
        print(f"Read true image size from {first_true_image}: height={height}, width={width}")
    else:
        height, width = FALLBACK_IMAGE_SIZE
        print(f"True image/000.* was not found; fallback image size: height={height}, width={width}")

    existing_true_images = [
        os.path.join(TRUE_IMAGE_DIR, f"{i:03d}{true_image_ext}")
        for i in range(FRAME_COUNT)
        if os.path.exists(os.path.join(TRUE_IMAGE_DIR, f"{i:03d}{true_image_ext}"))
    ]
    if len(existing_true_images) != FRAME_COUNT:
        print(
            f"Warning: found {len(existing_true_images)} / {FRAME_COUNT} true images with extension "
            f"{true_image_ext}; missing frames will be black in {OVERVIEW_FILE}."
        )

    # ===================== Derived radar parameters =====================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    c = 299792458.0
    wavelength = c / RADAR_FREQUENCY
    omega = 2.0 * np.pi / ROTATION_PERIOD
    omega_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64) * omega

    elevation = np.radians(ELEVATION_DEG)
    omega_eff_ref = abs(omega * np.cos(elevation))
    if omega_eff_ref <= 1e-8:
        raise ValueError("ELEVATION_DEG is too close to +/-90 deg; azimuth scale is singular.")

    scale_hz_to_m_ref = wavelength / (2.0 * omega_eff_ref)
    AZIMUTH_RESOLUTION = DOPPLER_RESOLUTION * scale_hz_to_m_ref
    AZIMUTH_GRID_SPACING = DOPPLER_GRID_SPACING * scale_hz_to_m_ref

    range_axis = (np.arange(height) - 0.5 * (height - 1)) * RANGE_GRID_SPACING
    doppler_axis = (np.arange(width) - 0.5 * (width - 1)) * DOPPLER_GRID_SPACING
    azimuth_axis = doppler_axis * scale_hz_to_m_ref

    obs_start = datetime.fromisoformat(OBS_START_UTC.replace("Z", "+00:00"))
    obs_end = datetime.fromisoformat(OBS_END_UTC.replace("Z", "+00:00"))
    if obs_start.tzinfo is None:
        obs_start = obs_start.replace(tzinfo=timezone.utc)
    if obs_end.tzinfo is None:
        obs_end = obs_end.replace(tzinfo=timezone.utc)

    total_observation_sec = (obs_end - obs_start).total_seconds()
    if total_observation_sec <= 0:
        raise ValueError("OBS_END_UTC must be later than OBS_START_UTC.")
    frame_interval_sec = total_observation_sec / (FRAME_COUNT - 1)
    frame_azimuth_step_deg = VIEW_SEQUENCE_DIRECTION * 360.0 * frame_interval_sec / ROTATION_PERIOD

    overview_cols = 8
    overview_panel_rows = int(np.ceil(FRAME_COUNT / overview_cols))

    print(f"Generating {FRAME_COUNT}-frame ellipsoid preview, true/ellipsoid comparison, and radar_meta.json")
    print(f"- Output directory          : {OUTPUT_DIR}")
    print(
        f"- Comparison layout         : each frame block: true row / ellipsoid row / overlay row, "
        f"{overview_cols} columns x {overview_panel_rows} rows each"
    )
    print(f"- Observation start/end     : {OBS_START_UTC} -> {OBS_END_UTC}")
    print(f"- Frame interval            : {frame_interval_sec / 60.0:.6g} min")
    print(f"- Rotation period           : {ROTATION_PERIOD / 3600.0:.6g} h")
    print(f"- View sequence direction   : {VIEW_SEQUENCE_DIRECTION:+.0f}")
    print(f"- Initial azimuth           : {AZIMUTH_START_DEG:.6g} deg")
    print(f"- Frame azimuth step        : {frame_azimuth_step_deg:.6g} deg")
    print(f"- Last frame azimuth        : {AZIMUTH_START_DEG + (FRAME_COUNT - 1) * frame_azimuth_step_deg:.6g} deg")
    print(f"- Range resolution          : {RANGE_RESOLUTION:.6g} m")
    print(f"- Doppler resolution        : {DOPPLER_RESOLUTION:.6g} Hz")
    print(f"- Range grid spacing        : {RANGE_GRID_SPACING:.6g} m/pixel")
    print(f"- Doppler grid spacing      : {DOPPLER_GRID_SPACING:.6g} Hz/pixel")
    print(f"- Azimuth resolution        : {AZIMUTH_RESOLUTION:.6g} m")
    print(f"- Azimuth grid spacing      : {AZIMUTH_GRID_SPACING:.6g} m/pixel")
    print(f"- Hz-to-meter scale         : {scale_hz_to_m_ref:.6g} m/Hz")
    print(f"- Elevation                 : {ELEVATION_DEG:.6g} deg")
    print(f"- Ellipsoid semi-axes       : {ELLIPSOID_SEMI_AXES}")
    print(f"- Ellipsoid diameters       : {tuple(2.0 * x for x in ELLIPSOID_SEMI_AXES)}")

    # ===================== Ellipsoid mesh =====================
    mesh = trimesh.creation.icosphere(subdivisions=ELLIPSOID_SUBDIVISIONS, radius=1.0)
    mesh.vertices *= np.asarray(ELLIPSOID_SEMI_AXES, dtype=np.float64)
    mesh.apply_translation(-mesh.centroid)
    mesh.fix_normals()

    print(f"- Mesh vertices / faces     : {len(mesh.vertices):,} / {len(mesh.faces):,}")

    face_areas = mesh.area_faces.astype(np.float64)
    normals = mesh.face_normals.astype(np.float64)

    frames_meta = []
    generated_images = []
    rng = np.random.default_rng(NOISE_SEED)

    # ===================== Facet projection for preview =====================
    for i in tqdm(range(FRAME_COUNT), desc="Rendering ellipsoid preview"):
        frame_time_sec = i * frame_interval_sec
        frame_time_utc = obs_start.timestamp() + frame_time_sec
        frame_time_iso = datetime.fromtimestamp(frame_time_utc, tz=timezone.utc).isoformat().replace("+00:00", "Z")

        azimuth_deg = AZIMUTH_START_DEG + i * frame_azimuth_step_deg
        azimuth = np.radians(azimuth_deg)

        radar_los = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float64,
        )

        eff_omega_vec = np.cross(radar_los, omega_vec)
        eff_omega = np.linalg.norm(eff_omega_vec)
        if eff_omega <= 1e-8:
            raise ValueError(f"Frame {i:03d} has near-zero effective spin rate.")

        scale_hz_to_m = wavelength / (2.0 * eff_omega)
        azimuth_resolution = DOPPLER_RESOLUTION * scale_hz_to_m
        azimuth_grid_spacing = DOPPLER_GRID_SPACING * scale_hz_to_m
        frame_azimuth_axis = doppler_axis * scale_hz_to_m

        incidence_cos = np.sum(normals * radar_los, axis=-1)
        visible_mask = incidence_cos > 0

        v_cos = incidence_cos[visible_mask]
        v_areas = face_areas[visible_mask]
        scatter_intensity = v_areas * (v_cos ** 2)

        preview_image = np.zeros((height, width), dtype=np.float32)
        faces_vertices = mesh.triangles[visible_mask].astype(np.float64)

        range_coord = -np.sum(faces_vertices * radar_los, axis=-1)
        range_bin = range_coord / RANGE_GRID_SPACING + 0.5 * (height - 1)

        velocity = np.cross(omega_vec, faces_vertices)
        doppler_coord = (2.0 / wavelength) * np.sum(velocity * radar_los, axis=-1)
        azimuth_coord = doppler_coord * scale_hz_to_m
        azimuth_bin = azimuth_coord / azimuth_grid_spacing + 0.5 * (width - 1)

        pts_2d = np.stack([azimuth_bin, range_bin], axis=-1)

        for idx in range(len(pts_2d)):
            pts = pts_2d[idx]
            intensity = scatter_intensity[idx]

            edge1 = pts[1] - pts[0]
            edge2 = pts[2] - pts[0]
            area_2d = 0.5 * abs(edge1[0] * edge2[1] - edge1[1] * edge2[0])

            if area_2d < 0.5:
                r_idx = int(round(np.mean(pts[:, 1])))
                a_idx = int(round(np.mean(pts[:, 0])))
                if 0 <= r_idx < height and 0 <= a_idx < width:
                    preview_image[r_idx, a_idx] += intensity
            else:
                pts_int = np.round(pts).astype(np.int32)
                x_min, y_min = np.min(pts_int, axis=0)
                x_max, y_max = np.max(pts_int, axis=0)

                if x_max < 0 or x_min >= width or y_max < 0 or y_min >= height:
                    continue

                x_min = max(0, int(x_min))
                y_min = max(0, int(y_min))
                x_max = min(width - 1, int(x_max))
                y_max = min(height - 1, int(y_max))

                if x_max >= x_min and y_max >= y_min:
                    mask_img = Image.new("L", (x_max - x_min + 1, y_max - y_min + 1), 0)
                    local_pts = pts_int - np.array([x_min, y_min])
                    polygon = [tuple(map(int, p)) for p in local_pts]
                    ImageDraw.Draw(mask_img).polygon(polygon, fill=1)
                    mask = np.asarray(mask_img, dtype=np.float32)

                    pixels_covered = np.sum(mask)
                    if pixels_covered > 0:
                        density = intensity / pixels_covered
                        preview_image[y_min:y_max + 1, x_min:x_max + 1] += mask * density

        range_res_pix = RANGE_RESOLUTION / RANGE_GRID_SPACING
        azimuth_res_pix = azimuth_resolution / azimuth_grid_spacing
        n_sidelobes = 1

        r_half = int(np.ceil(n_sidelobes * range_res_pix))
        a_half = int(np.ceil(n_sidelobes * azimuth_res_pix))
        r_coords = np.arange(-r_half, r_half + 1) / max(range_res_pix, 1e-8)
        a_coords = np.arange(-a_half, a_half + 1) / max(azimuth_res_pix, 1e-8)

        kernel_2d = np.outer(np.sinc(r_coords), np.sinc(a_coords)) ** 2
        kernel_2d /= kernel_2d.sum()
        preview_image = convolve(preview_image, kernel_2d.astype(np.float32), mode="constant", cval=0.0)
        preview_image = np.clip(preview_image, 0.0, None)
        generated_images.append(preview_image.astype(np.float32))

        frames_meta.append({
            "file": f"{i:03d}{true_image_ext}",
            "radar_los": radar_los.tolist(),
            "rotation_period": float(ROTATION_PERIOD),
            "radar_frequency": float(RADAR_FREQUENCY),
            "range_resolution": float(RANGE_RESOLUTION),
            "doppler_resolution": float(DOPPLER_RESOLUTION),
            "azimuth_resolution": float(azimuth_resolution),
            "range_axis": range_axis.tolist(),
            "doppler_axis": doppler_axis.tolist(),
            "azimuth_axis": frame_azimuth_axis.tolist(),
            "range_grid_spacing": float(RANGE_GRID_SPACING),
            "doppler_grid_spacing": float(DOPPLER_GRID_SPACING),
            "azimuth_grid_spacing": float(azimuth_grid_spacing),
            "image_size": [height, width]
        })

    clean_stack = np.stack(generated_images, axis=0).astype(np.float32)

    if NOISE_SNR_DB is not None:
        snr = 10 ** (NOISE_SNR_DB / 10.0)
        clean_power = np.maximum(clean_stack, 0.0)
        clean_max = float(np.max(clean_power))
        if clean_max > 0.0:
            signal_mask = clean_power > clean_max * 1e-2
            signal_power_ref = float(np.mean(clean_power[signal_mask])) if np.any(signal_mask) else clean_max
            noise_power = signal_power_ref / snr
            noise_sigma = np.sqrt(noise_power / 2.0)
            noise_i = noise_sigma * rng.standard_normal(clean_power.shape).astype(np.float32)
            noise_q = noise_sigma * rng.standard_normal(clean_power.shape).astype(np.float32)
            final_stack = ((np.sqrt(clean_power) + noise_i) ** 2 + noise_q ** 2).astype(np.float32)
            print(
                f"Added detected-power thermal noise: snr_db={NOISE_SNR_DB}, "
                f"noise_power={noise_power:.6g}, component_sigma={noise_sigma:.6g}"
            )
        else:
            final_stack = clean_stack
            print("Skip thermal noise: clean image stack has non-positive maximum")
    else:
        final_stack = clean_stack

    global_scale = float(np.percentile(clean_stack, 100))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = float(np.max(final_stack))
    print(f"Normalize ellipsoid preview by global 100 percentile: {global_scale:.6g}")

    ellipsoid_tiles = []
    for image_power in final_stack:
        if global_scale > 0.0:
            img_norm = np.clip(image_power / global_scale * 255.0, 0, 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(image_power, dtype=np.uint8)
        ellipsoid_tiles.append(Image.fromarray(img_norm, mode="L").convert("RGB"))

    true_tiles = []
    for i in range(FRAME_COUNT):
        true_path = os.path.join(TRUE_IMAGE_DIR, f"{i:03d}{true_image_ext}")
        if os.path.exists(true_path):
            tile = Image.open(true_path).convert("RGB")
            if tile.size != (width, height):
                tile = tile.resize((width, height), Image.Resampling.BILINEAR)
        else:
            tile = Image.new("RGB", (width, height), (0, 0, 0))
        true_tiles.append(tile)

    overlay_tiles = []
    for true_tile, ellipsoid_tile in zip(true_tiles, ellipsoid_tiles):
        true_arr = np.asarray(true_tile.convert("L"), dtype=np.float32)
        ell_arr = np.asarray(ellipsoid_tile.convert("L"), dtype=np.float32)

        if np.max(true_arr) > 0:
            true_arr = true_arr / np.max(true_arr) * 180.0
        if np.max(ell_arr) > 0:
            ell_arr = ell_arr / np.max(ell_arr) * 255.0

        overlay_arr = np.zeros((height, width, 3), dtype=np.uint8)
        overlay_arr[..., 0] = np.clip(true_arr + 0.85 * ell_arr, 0, 255).astype(np.uint8)
        overlay_arr[..., 1] = np.clip(true_arr, 0, 255).astype(np.uint8)
        overlay_arr[..., 2] = np.clip(true_arr, 0, 255).astype(np.uint8)
        overlay_tiles.append(Image.fromarray(overlay_arr, mode="RGB"))

    overview_rows = overview_panel_rows * 3
    overview = Image.new("RGB", (overview_cols * width, overview_rows * height), (0, 0, 0))

    for tile_idx in range(FRAME_COUNT):
        row_block = tile_idx // overview_cols
        col_idx = tile_idx % overview_cols
        y_base = row_block * 3 * height
        x_pos = col_idx * width
        overview.paste(true_tiles[tile_idx], (x_pos, y_base))
        overview.paste(ellipsoid_tiles[tile_idx], (x_pos, y_base + height))
        overview.paste(overlay_tiles[tile_idx], (x_pos, y_base + 2 * height))

    overview_path = os.path.join(OUTPUT_DIR, OVERVIEW_FILE)
    overview.save(overview_path)

    meta_dict = {
        "frames": frames_meta
    }

    meta_path = os.path.join(OUTPUT_DIR, "radar_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=4)

    print(f"Dataset metadata generated successfully at: {OUTPUT_DIR}")
    print(f"- True/ellipsoid comparison saved to {overview_path}")
    print(f"- Metadata saved to {meta_path}")



