import os
import json
import numpy as np
import trimesh
import cv2 
from tqdm import tqdm

def fun_add_noise_real(snr_db, y3):
    """
    按照 MATLAB 版本 funAddNoiseReal(SNR, y3) 的逻辑添加实值高斯噪声。
    """
    snr = 10 ** (snr_db / 10.0)

    pic = np.abs(y3)
    pic_max = np.max(pic)
    if pic_max <= 0:
        return y3.copy()

    pic_normal = pic / pic_max
    bw = pic_normal > 1e-2
    if not np.any(bw):
        return y3.copy()

    amp_mean = np.mean(np.abs(y3[bw]))
    pn = (amp_mean ** 2) / snr

    noise = np.sqrt(pn) * np.random.randn(*y3.shape)
    return y3 + noise


def generate_isar_dataset(
    stl_path,
    output_dir,
    scale_factor=60.0,
    rotation_period=5.0 * 3600,  # 5小时转换为秒
    radar_frequency=9.7e9,       # 默认 9.7 GHz
    range_resolution=1.0,        # 物理距离分辨率 (米) 
    doppler_resolution=1.0,      # 多普勒分辨率 (Hz) 
    range_grid_spacing=0.5,      # 实际距离向采样网格间距 (米/像素) - 投影用
    doppler_grid_spacing=0.5,    # 实际多普勒向采样网格间距 (Hz/像素) - 投影用
    image_size=(64, 64),         # (距离向高度, 多普勒向宽度)
    noise_snr_db=20.0,
    noise_seed=None
):
    """
    从 STL 模型生成 ISAR 图像序列与配套的 radar_meta.json
    （完全基于 RD 域投影，通过缩放系数推导并记录方位向参数）
    """
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, 'image')
    os.makedirs(image_dir, exist_ok=True)

    print(f"Loading and scaling mesh from {stl_path}...")
    mesh = trimesh.load(stl_path)
    
    # 1. 放大与居中
    mesh.apply_scale(scale_factor)
    mesh.apply_translation(-mesh.centroid) 

    # 保存缩放后的模型为 PLY 文件，名称形如 “1999 JV6 Radar_60.ply”
    ply_basename = os.path.splitext(os.path.basename(stl_path))[0] + f"_{scale_factor:g}.ply"
    ply_path = os.path.join(output_dir, ply_basename)
    mesh.export(ply_path)
    print(f"已保存缩放后的网格至 {ply_path}")
    
    extents = mesh.extents
    print(f"小行星尺寸(X, Y, Z): {extents[0]:.2f}m x {extents[1]:.2f}m x {extents[2]:.2f}m")
    
    # 2. 提取面元法(Facet Method)所需的核心几何数据
    print(f"Extracting exact mesh faces for Facet Projection...")
    face_areas = mesh.area_faces     # 每个三角面片的真实面积 (N,)
    normals = mesh.face_normals      # 每个三角面片的法线 (N, 3)
    print(f"Total scattering facets (faces): {len(face_areas):,}")

    # 自转向量
    c = 299792458.0
    wavelength = c / radar_frequency
    omega_vec = np.array([0.0, 0.0, 1.0]) * (2.0 * np.pi / rotation_period) # 默认绕 Z 轴
    
    # 3. 定义 18 个视角 (仰角 45 度 9 张，仰角 -45 度 9 张)
    views = []
    el_up = np.radians(45.0)
    for az_deg in range(0, 360, 40):
        views.append((np.radians(az_deg), el_up, 'up'))
        
    el_down = np.radians(-45.0)
    for az_deg in range(20, 360, 40):
        views.append((np.radians(az_deg), el_down, 'down'))

    frames_meta = []
    generated_images = []
    image_file_names = []
    rng = np.random.default_rng(noise_seed)
    height, width = image_size
    
    # --- 提前计算并保存距离轴和多普勒轴坐标 (单位: 米 和 Hz) ---
    range_axis = (np.arange(height) - 0.5 * (height - 1)) * range_grid_spacing
    doppler_axis = (np.arange(width) - 0.5 * (width - 1)) * doppler_grid_spacing
    

    print("Simulating ISAR images using True Facet Projection...")
    for i, (az, el, name) in enumerate(tqdm(views)):
        # 计算 Target-to-Radar 的视线向量 (LOS)
        los_x = np.cos(el) * np.cos(az)
        los_y = np.cos(el) * np.sin(az)
        los_z = np.sin(el)
        radar_los = np.array([los_x, los_y, los_z], dtype=np.float32)
        
        # 计算有效旋转平面上的“方位向单位向量” (Cross-range direction)
        # 根据雷达物理，多普勒等效于在 (LOS x Omega) 方向上的投影
        eff_omega_vec = np.cross(radar_los, omega_vec)
        eff_omega = np.linalg.norm(eff_omega_vec)

        # --- 核心思想：图不变，仅仅是轴和分辨率乘一个缩放系数！ ---
        # scale = lambda / (2 * omega_eff)
        scale_hz_to_m = wavelength / (2.0 * eff_omega) if eff_omega > 1e-8 else 0.0
        
        # 动态算出这一帧专属的方位向参数
        azimuth_resolution = doppler_resolution * scale_hz_to_m
        azimuth_grid_spacing = doppler_grid_spacing * scale_hz_to_m
        azimuth_axis = doppler_axis * scale_hz_to_m

        # --- 物理散射计算 ---
        # 1. 剔除背暗面 (入射角余弦值 < 0 的点)
        incidence_cos = np.sum(normals * radar_los, axis=-1)
        visible_mask = incidence_cos > 0
        
        v_cos = incidence_cos[visible_mask]
        v_areas = face_areas[visible_mask]
        
        # 根据统一的朗伯体面元近似公式：scatter = 真实面积 * cos^2(theta)
        scatter_intensity = v_areas * (v_cos ** 2)
        
        # --- 真正的面元投影 (Facet Rasterization) ---
        isar_image = np.zeros((height, width), dtype=np.float32)
        
        # 提取可见面片的3个顶点 (M, 3, 3)
        faces_vertices = mesh.triangles[visible_mask]
        # 距离向投影 r -> (M, 3) 单位: 米
        range_coord_v = -np.sum(faces_vertices * radar_los, axis=-1)
        range_bin_v = range_coord_v / range_grid_spacing + 0.5 * (height - 1)
        
        # 严格在 多普勒向 (Hz) 进行投影 -> (M, 3) 
        velocity_v = np.cross(omega_vec, faces_vertices)
        doppler_coord_v = (2.0 / wavelength) * np.sum(velocity_v * radar_los, axis=-1)
        doppler_bin_v = doppler_coord_v / doppler_grid_spacing + 0.5 * (width - 1)
        
        # OpenCV 要求的顶点格式 (M, 3, 2)
        pts_2d = np.stack([doppler_bin_v, range_bin_v], axis=-1)

        
        # 遍历可见面片进行面元像素填充
        for idx in range(len(pts_2d)):
            pts = pts_2d[idx]
            intensity = scatter_intensity[idx]
            
            # 计算面片在2D网格上的占据面积
            area_2d = 0.5 * np.abs(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
            
            if area_2d < 0.5:
                # 亚像素级小面元：直接退化为中心点投影，避免 OpenCV 栅格化时能量丢失
                r_idx = int(round(np.mean(pts[:, 1])))
                d_idx = int(round(np.mean(pts[:, 0])))
                if 0 <= r_idx < height and 0 <= d_idx < width:
                    isar_image[r_idx, d_idx] += intensity
            else:
                # 大面元：进行真正的多边形填充，将能量密度均匀铺在此面元覆盖的所有像素上
                pts_int = np.round(pts).astype(np.int32)
                x_min, y_min = np.min(pts_int, axis=0)
                x_max, y_max = np.max(pts_int, axis=0)
                
                # 画布外剔除
                if x_max < 0 or x_min >= width or y_max < 0 or y_min >= height:
                    continue
                    
                # 限制边界
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(width - 1, x_max), min(height - 1, y_max)
                
                if x_max >= x_min and y_max >= y_min:
                    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
                    local_pts = pts_int - np.array([x_min, y_min])
                    cv2.fillConvexPoly(mask, local_pts, 1)
                    
                    pixels_covered = np.sum(mask)
                    if pixels_covered > 0:
                        density = intensity / pixels_covered
                        isar_image[y_min:y_max+1, x_min:x_max+1] += mask * density
        # === 新增：sinc 点扩展卷积，模拟真实 PSF ===
        # 计算主瓣宽度对应的像素数
        range_res_pix = range_resolution / range_grid_spacing   # 距离向主瓣宽度（像素）
        doppler_res_pix = doppler_resolution / doppler_grid_spacing  # 多普勒主瓣宽度（像素）
        n_sidelobes = 1   # 保留 0 个旁瓣，可根据需要调整
        
        # 距离向一维 sinc 核
        r_half = int(np.ceil(n_sidelobes * range_res_pix))
        r_coords = np.arange(-r_half, r_half + 1) / range_res_pix
        k_r = np.sinc(r_coords)  # sinc(x) = sin(pi*x)/(pi*x), np.sinc 定义为 sin(pi*x)/(pi*x)
        # 多普勒向一维 sinc 核
        d_half = int(np.ceil(n_sidelobes * doppler_res_pix))
        d_coords = np.arange(-d_half, d_half + 1) / (doppler_res_pix + 1e-8)
        k_d = np.sinc(d_coords)
      
        
        # 二维可分离核（外积）
        kernel_2d = np.outer(k_r, k_d)
        kernel_2d = kernel_2d ** 2
        # 归一化保证能量守恒（核的积分为1）
        kernel_2d /= kernel_2d.sum()
        
        # 卷积（same 模式）
        isar_image = cv2.filter2D(isar_image, -1, kernel_2d,
                                  borderType=cv2.BORDER_CONSTANT)
        # Legacy compressed-amplitude output:
        # isar_image = np.sqrt(isar_image)
        isar_image = np.clip(isar_image, 0.0, None)
        # =============================================
        # 使用分位数截断 (Percentile Clipping)

        file_name = f"{i:03d}.png"
        generated_images.append(isar_image.astype(np.float32))
        image_file_names.append(file_name)
        
        # 记录 Metadata (此时横轴全变成了 azimuth，单位为米)
        frames_meta.append({
            "file": file_name,
            "radar_los": radar_los.tolist(),
            "rotation_period": float(rotation_period),
            "radar_frequency": float(radar_frequency),
            # 成像分辨率（物理分辨率）
            "range_resolution": float(range_resolution),
            "doppler_resolution": float(doppler_resolution),
            "azimuth_resolution": float(azimuth_resolution),

            # 图像坐标轴（单位：m）
            "range_axis": range_axis.tolist(),
            "doppler_axis": doppler_axis.tolist(),
            "azimuth_axis": azimuth_axis.tolist(),

            # 网格采样参数
            "range_grid_spacing": float(range_grid_spacing),
            "doppler_grid_spacing": float(doppler_grid_spacing),
            "azimuth_grid_spacing": float(azimuth_grid_spacing),

            # 图像尺寸
            "image_size": [height, width] #距离向，方位向
        })

    clean_stack = np.stack(generated_images, axis=0).astype(np.float32)
    if noise_snr_db is not None:
        snr = 10 ** (noise_snr_db / 10.0)
        clean_max = float(np.max(clean_stack))
        if clean_max > 0.0:
            signal_mask = clean_stack > clean_max * 1e-2
            signal_ref = float(np.mean(np.abs(clean_stack[signal_mask]))) if np.any(signal_mask) else clean_max
            noise_std = signal_ref / np.sqrt(snr)
            noise = noise_std * rng.standard_normal(clean_stack.shape).astype(np.float32)
            final_stack = np.clip(clean_stack + noise, 0.0, None)
            print(f"Added uniform thermal noise: snr_db={noise_snr_db}, noise_std={noise_std:.6g}")
        else:
            final_stack = clean_stack
            print('Skip thermal noise: clean image stack has non-positive maximum')
    else:
        final_stack = clean_stack

    global_scale = float(np.percentile(final_stack, 99.9))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = float(np.max(final_stack))
    print(f"Normalize saved images by global 99.9 percentile: {global_scale:.6g}")

    overview_tiles = []
    for file_name, isar_image in zip(image_file_names, final_stack):
        if global_scale > 0.0:
            img_norm = np.clip(isar_image / global_scale * 255.0, 0, 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(isar_image, dtype=np.uint8)
        overview_tiles.append(img_norm)
        img_path = os.path.join(image_dir, file_name)
        cv2.imwrite(img_path, img_norm)

    overview_path = os.path.join(output_dir, 'isar_overview_2x9.png')
    if overview_tiles:
        overview_rows = 2
        overview_cols = 9
        tile_shape = overview_tiles[0].shape
        blank_tile = np.zeros(tile_shape, dtype=np.uint8)
        padded_tiles = overview_tiles[:overview_rows * overview_cols]
        padded_tiles += [blank_tile] * (overview_rows * overview_cols - len(padded_tiles))
        overview = np.vstack([
            np.hstack(padded_tiles[row * overview_cols:(row + 1) * overview_cols])
            for row in range(overview_rows)
        ])
        cv2.imwrite(overview_path, overview)
    # 4. Write radar_meta.json
    meta_dict = {
        "frames": frames_meta
    }
    
    meta_path = os.path.join(output_dir, 'radar_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta_dict, f, indent=4)
        
    print(f"Dataset generated successfully at: {output_dir}")
    print(f"- 18 Images saved to {image_dir}")
    print(f"- Overview image saved to {overview_path}")
    print(f"- Metadata saved to {meta_path}")

if __name__ == "__main__":
    np.random.seed(42)
    INPUT_STL = "public_data/1999JV6/1999 JV6 Radar.stl" 
    OUTPUT_DIR = "public_data/1999JV6"

    # === 核心：用物理公式动态计算雷达参数 ===
    desired_azimuth_res = 1.0  # 你真正想要的：1.0 米的方位向物理分辨率
    
    # 1. 定义已知物理常量
    c = 299792458.0
    radar_freq = 9.7e9
    wavelength = c / radar_freq
    rot_period = 5.0 * 3600
    omega = 2.0 * np.pi / rot_period
    
    # 2. 针对 45° 仰角下的有效自转角速度进行补偿
    # (因为你的视角列表中主要使用 45° 仰角，以此作为参考基准计算分辨率)
    omega_eff = omega * np.cos(np.radians(45.0)) 
    
    # 3. 逆推真实需要的雷达多普勒分辨率 (Hz)
    # 公式: f_d = (2 * omega_eff * x) / lambda
    calc_doppler_res = (2.0 * omega_eff * desired_azimuth_res) / wavelength
    
    # 4. 设定采样网格 (2倍过采样，使图像平滑)
    # calc_doppler_spacing = calc_doppler_res * 0.5 
    calc_doppler_spacing = calc_doppler_res * 1

    print(f"\n[*] 目标方位向分辨率: {desired_azimuth_res} 米")
    print(f"[*] 自动推导的雷达多普勒分辨率: {calc_doppler_res:.5f} Hz")
    print(f"[*] 自动推导的多普勒网格间距: {calc_doppler_spacing:.5f} Hz/pixel\n")
        
    if os.path.exists(INPUT_STL):
        generate_isar_dataset(
            stl_path=INPUT_STL,
            output_dir=OUTPUT_DIR,
            scale_factor=60.0, 
            rotation_period=rot_period,      # 传入上面定义的变量
            radar_frequency=radar_freq,      # 传入上面定义的变量
            range_resolution=1.0,           
            doppler_resolution=calc_doppler_res,         # <--- 直接传入计算好的公式变量
            range_grid_spacing=1,         
            doppler_grid_spacing=calc_doppler_spacing,   # <--- 直接传入计算好的公式变量
            image_size=(64, 64),
            noise_snr_db=None,                           # None代表不加噪声；数值表示统一热噪声目标 SNR(dB)
            noise_seed=20260708
        )
    else:
        print(f"❌ 找不到文件: '{INPUT_STL}'")
        print(f"💡 当前运行目录是: {os.getcwd()}")
        print("请确保你是在 NeuS-main 根目录下点击的运行。")
