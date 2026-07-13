import math
import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# 辅助函数：从帧元数据中安全提取所需字段
# ------------------------------------------------------------
def _meta_scalar(frame_meta, key, default=None):
    """提取标量值，若缺失且未提供默认值则报错"""
    value = frame_meta.get(key, default)
    if value is None:
        raise KeyError('Missing frame metadata: {}'.format(key))
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float32)
    return value.reshape(-1)[0]


def _meta_vector(frame_meta, key):
    """提取 3 维向量，必须存在且长度为 3"""
    value = frame_meta.get(key)
    if value is None:
        raise KeyError('Missing frame metadata: {}'.format(key))
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float32)
    value = value.reshape(-1)
    if value.shape[0] != 3:
        raise ValueError('{} must be a 3-vector'.format(key))
    return value


class ISARRenderer:
    """
    极简可微非相干 ISAR 渲染器
    · 自转轴固定为 Z 轴
    · 按最大值归一化，便于和归一化 ISAR 图监督
    """

    def __init__(self,
                 bound=1.0,               # 3D 采样空间范围：[-bound, bound]³ 米
                 splat_mode=2,             # 1=最近邻, 2=双线性, 3=双线性+PSF
                 psf_lobes=1,              # 1=主瓣, 2=主瓣+第一旁瓣, 3=主瓣+第一/第二旁瓣
                 coord_scale=25.0,         # 投影时归一化坐标 -> 物理米制坐标的比例
                 n_azimuth=None,           # None 时取图像宽度 W
                 n_range=None,             # coarse 距离向采样数，None 时取图像高度 H
                 n_importance=0,            # NeuS up-sampling 追加的 fine 距离向采样数
                 up_sample_steps=1,         # fine 采样分几轮加入
                 n_height=None,            # None 时取 W，表示 ISAR 投影丢掉的第三维采样数
                 ray_chunk=1024):          # 按 ray 分块，降低一次性 SDF/梯度显存峰值
        self.bound = float(bound)
        self.splat_mode = int(splat_mode)
        self.psf_lobes = int(psf_lobes)
        self.coord_scale = float(coord_scale)
        self.n_azimuth = None if n_azimuth is None else int(n_azimuth)
        self.n_range = None if n_range is None else int(n_range)
        self.n_importance = int(n_importance)
        self.up_sample_steps = int(up_sample_steps)
        self.n_height = None if n_height is None else int(n_height)
        self.ray_chunk = int(ray_chunk)

        # 自转轴固定为 Z 轴（单位向量），不需要角速度大小
        self.rot_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        self.rot_axis = F.normalize(self.rot_axis, dim=0)

        if self.bound <= 0.0:
            raise ValueError('bound must be positive')
        if self.splat_mode not in (1, 2, 3):
            raise ValueError('splat_mode must be 1, 2, or 3')
        if self.psf_lobes < 1:
            raise ValueError('psf_lobes must be at least 1')
        if self.coord_scale <= 0.0:
            raise ValueError('coord_scale must be positive')
        for name, value in [('n_azimuth', self.n_azimuth), ('n_range', self.n_range), ('n_height', self.n_height)]:
            if value is not None and value < 2:
                raise ValueError('{} must be at least 2'.format(name))
        if self.n_importance < 0:
            raise ValueError('n_importance must be non-negative')
        if self.up_sample_steps < 1:
            raise ValueError('up_sample_steps must be positive')
        if self.ray_chunk < 1:
            raise ValueError('ray_chunk must be positive')

    def project_points(self, points, frame_meta):
        """
        将 3D 采样点投影到距离-方位向图像平面（物理坐标→像素索引）

        参数:
            points: (N, 3) 采样点坐标
            frame_meta: 元数据字典，必须包含：
                'radar_los'    : 3 维向量，视线方向（目标→雷达）
                'range_axis'   : 1D 张量，距离轴各像素中心的物理坐标（米）
                'azimuth_axis' : 1D 张量，方位轴各像素中心的物理坐标（米）

        返回:
            range_bin    : (N,) 距离向浮点像素索引
            azimuth_bin  : (N,) 方位向浮点像素索引
            range_coord  : (N,) 物理距离坐标（米）
            azimuth_coord: (N,) 物理方位坐标（米）
        """
        device = points.device
        points_phys = points * self.coord_scale

        # ---- 视线方向（目标→雷达），归一化 ----
        los = _meta_vector(frame_meta, 'radar_los').to(device)
        los = F.normalize(los, dim=0)

        # ---- 自转轴（Z 轴）----
        z_axis = self.rot_axis.to(device)

        # ===== 距离向坐标 =====
        # 距离 = 点在视线方向上的投影（米）
        range_coord = -torch.sum(points_phys * los, dim=-1)

        # ===== 方位向坐标 =====
        # 标准 ISAR 方位向物理公式（已消去角速度和波长）：
        #   x = ((z × p) · los) / ||z × los||
        # 单位：米
        z_cross_los = torch.cross(los,z_axis,  dim=0)    # 视线与自转轴张成平面的法向量
        norm_cross = torch.norm(z_cross_los) + 1e-8       # 防止平行时除零
        z_cross_los = z_cross_los / norm_cross
        azimuth_coord = torch.sum(points_phys *z_cross_los, dim=-1)

        # ===== 从坐标轴数组提取网格信息 =====
        range_axis = frame_meta['range_axis'].to(device)
        azimuth_axis = frame_meta['azimuth_axis'].to(device)
        height, width = range_axis.shape[0], azimuth_axis.shape[0]

        # 假设坐标轴均匀，网格间距 = 相邻像素中心的差值
        range_spacing = range_axis[1] - range_axis[0]        # 米/像素
        azimuth_spacing = azimuth_axis[1] - azimuth_axis[0]  # 米/像素

        # 第 0 个像素中心对应的物理坐标
        range_origin = range_axis[0]
        azimuth_origin = azimuth_axis[0]

        # 物理坐标 → 浮点像素索引（0 基准）
        range_bin = (range_coord - range_origin) / range_spacing
        azimuth_bin = (azimuth_coord - azimuth_origin) / azimuth_spacing

        return range_bin, azimuth_bin, range_coord, azimuth_coord

    def render_frame(self, frame_meta, sdf_network, variance_network,
                     image_shape=None, n_samples=None, cos_anneal_ratio=1.0):
        """
        渲染一帧 ISAR 图像

        参数:
            frame_meta        : 当前帧的元数据
            sdf_network       : SDF 网络，输出 SDF 值 + 特征
            variance_network : 全局方差网络 (输出标量 s)
            image_shape       : 输出图像 (H, W)，默认从 frame_meta 读取
            n_samples         : 总采样点数，默认使用构造函数中的值

        返回:
            字典，包含渲染图像及中间变量（用于损失计算和可视化）
        """
        return self.render_frame_neus(
            frame_meta,
            sdf_network,
            variance_network,
            image_shape=image_shape,
            n_samples=n_samples,
            cos_anneal_ratio=cos_anneal_ratio
        )

    def render_frame_neus(self, frame_meta, sdf_network, variance_network,
                          image_shape=None, n_samples=None, cos_anneal_ratio=1.0):
        device = next(sdf_network.parameters()).device

        # ---- 确定图像尺寸 ----
        if image_shape is None:
            range_axis = frame_meta['range_axis']
            azimuth_axis = frame_meta['azimuth_axis']
            height, width = range_axis.shape[0], azimuth_axis.shape[0]
        else:
            height, width = image_shape

        ray_bases, ray_dir, range_vals, ray_area = self.generate_parallel_rays(frame_meta, (height, width), device, n_samples)
        n_rays = ray_bases.shape[0]

        if torch.any(range_vals[1:] <= range_vals[:-1]):
            raise ValueError('range samples must be strictly ascending')

        sdf_min = None
        sdf_max = None
        image = torch.zeros(height, width, dtype=torch.float32, device=device)
        gradient_error_sum = torch.zeros([], dtype=torch.float32, device=device)
        gradient_error_count = torch.zeros([], dtype=torch.float32, device=device)
        debug_ret = {}

        for start in range(0, n_rays, self.ray_chunk):
            end = min(start + self.ray_chunk, n_rays)
            chunk_ret = self.render_ray_chunk(
                ray_bases[start:end],
                ray_dir,
                range_vals,
                ray_area,
                frame_meta,
                sdf_network,
                variance_network,
                (height, width),
                cos_anneal_ratio=cos_anneal_ratio
            )
            image = image + chunk_ret['image']
            chunk_sdf = chunk_ret.get('sdf')
            if chunk_sdf is not None:
                chunk_min = torch.min(chunk_sdf)
                chunk_max = torch.max(chunk_sdf)
                sdf_min = chunk_min if sdf_min is None else torch.minimum(sdf_min, chunk_min)
                sdf_max = chunk_max if sdf_max is None else torch.maximum(sdf_max, chunk_max)
            gradient_error_sum = gradient_error_sum + chunk_ret['gradient_error_sum']
            gradient_error_count = gradient_error_count + chunk_ret['gradient_error_count']
            debug_ret = chunk_ret

        if self.splat_mode == 3:
            image = self._apply_sinc_psf(image, frame_meta)

        
        eps = 1e-6
        image = torch.sqrt(torch.clamp(image, min=0.0) + eps) - np.sqrt(eps)

        # image = image / (image.max().detach() + 1e-8)   # normalize to [0, 1] for visualization/output

        gradient_error = gradient_error_sum / (gradient_error_count + 1e-5)

        return {
            'isar': image,                 # 渲染的 ISAR 图像 (H, W)
            'sdf': debug_ret.get('sdf'),
            'sdf_min': sdf_min,
            'sdf_max': sdf_max,
            'points': debug_ret.get('points'),
            'gradients': debug_ret.get('gradients'),
            'normals': debug_ret.get('normals'),
            'scatter': debug_ret.get('scatter'),
            'point_weight': debug_ret.get('point_weight'),
            'range_bin': debug_ret.get('range_bin'),
            'azimuth_bin': debug_ret.get('azimuth_bin'),
            'range_coord': debug_ret.get('range_coord'),
            'azimuth_coord': debug_ret.get('azimuth_coord'),
            'alpha': debug_ret.get('alpha'),
            'weights': debug_ret.get('weights'),
            's_val': debug_ret.get('s_val'),
            'gradient_error': gradient_error,
        }

    def render_ray_chunk(self, ray_bases, ray_dir, coarse_range_vals, ray_area, frame_meta,
                         sdf_network, variance_network, image_shape, cos_anneal_ratio=1.0):
        height, width = image_shape
        device = ray_bases.device
        n_rays = ray_bases.shape[0]

        range_vals = self.prepare_range_samples(
            ray_bases,
            ray_dir,
            coarse_range_vals,
            sdf_network
        )
        dists = self.compute_range_dists(range_vals)
        n_samples = range_vals.shape[1]

        points = ray_bases[:, None, :] + range_vals[..., None] * ray_dir[None, None, :]
        points_flat = points.reshape(-1, 3)
        points_flat.requires_grad_(True)

        sdf_output = sdf_network(points_flat)
        sdf = sdf_output[:, :1]
        d_output = torch.ones_like(sdf, requires_grad=False, device=device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points_flat,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        points = points_flat.reshape(n_rays, n_samples, 3)
        sdf = sdf.reshape(n_rays, n_samples, 1)
        gradients = gradients.reshape(n_rays, n_samples, 3)
        dirs = ray_dir[None, None, :].expand(n_rays, n_samples, 3)

        alpha, weights, s_val = self.compute_neus_weights(
            sdf,
            gradients,
            dirs,
            dists,
            variance_network,
            cos_anneal_ratio=cos_anneal_ratio
        )

        normals = F.normalize(gradients, dim=-1)
        los = _meta_vector(frame_meta, 'radar_los').to(device)
        los = F.normalize(los, dim=0)
        incidence_cos = torch.clamp(torch.sum(normals * (los)[None, None, :], dim=-1, keepdim=True), min=0.0)
        scatter = incidence_cos ** 2
        point_weight = weights * scatter * ray_area

        points_flat = points.reshape(-1, 3)
        point_weight_flat = point_weight.reshape(-1)
        range_bin, azimuth_bin, range_coord, azimuth_coord = self.project_points(points_flat, frame_meta)

        if self.splat_mode == 1:
            image = self._nearest_splat(range_bin, azimuth_bin, point_weight_flat, height, width)
        elif self.splat_mode in (2, 3):
            image = self._bilinear_splat(range_bin, azimuth_bin, point_weight_flat, height, width)
        else:
            raise ValueError('Unsupported splat_mode: {}'.format(self.splat_mode))

        gradient_error = (torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2
        # 原版 NeuS 只在 relaxed unit sphere 内施加 Eikonal 约束，避免远离物体的空区主导正则项。
        relax_inside_sphere = (torch.linalg.norm(points, ord=2, dim=-1) < 1.2).float().detach()

        return {
            'image': image,
            'gradient_error_sum': (gradient_error * relax_inside_sphere).sum(),
            'gradient_error_count': relax_inside_sphere.sum(),
            'sdf': sdf.reshape(-1, 1),
            'points': points_flat,
            'gradients': gradients.reshape(-1, 3),
            'normals': normals.reshape(-1, 3),
            'scatter': scatter.reshape(-1, 1),
            'point_weight': point_weight.reshape(-1, 1),
            'range_bin': range_bin,
            'azimuth_bin': azimuth_bin,
            'range_coord': range_coord,
            'azimuth_coord': azimuth_coord,
            'alpha': alpha,
            'weights': weights,
            's_val': s_val,
        }

    def prepare_range_samples(self, ray_bases, ray_dir, coarse_range_vals, sdf_network):
        n_rays = ray_bases.shape[0]
        range_vals = coarse_range_vals[None, :].expand(n_rays, -1)
        if self.n_importance == 0:
            return range_vals

        with torch.no_grad():
            points = ray_bases[:, None, :] + range_vals[..., None] * ray_dir[None, None, :]
            sdf = sdf_network.sdf(points.reshape(-1, 3)).reshape(n_rays, range_vals.shape[1])

            effective_steps = min(self.up_sample_steps, self.n_importance)
            remaining = self.n_importance
            for step in range(effective_steps):
                n_new = remaining // (effective_steps - step)
                remaining -= n_new
                if n_new <= 0:
                    continue

                new_range_vals = self.up_sample_range(
                    ray_bases,
                    ray_dir,
                    range_vals,
                    sdf,
                    n_new,
                    inv_s=64.0 * (2 ** step)
                )
                range_vals, sdf = self.cat_range_vals(
                    ray_bases,
                    ray_dir,
                    range_vals,
                    new_range_vals,
                    sdf,
                    sdf_network,
                    last=(step + 1 == effective_steps)
                )

        return range_vals.detach()

    def compute_range_dists(self, range_vals):
        dists = range_vals[:, 1:] - range_vals[:, :-1]
        dists = torch.cat([dists, dists[:, -1:]], dim=-1)
        return dists.clamp_min(1e-6)

    def up_sample_range(self, ray_bases, ray_dir, range_vals, sdf, n_importance, inv_s):
        n_rays, n_samples = range_vals.shape
        points = ray_bases[:, None, :] + range_vals[..., None] * ray_dir[None, None, :]
        radius = torch.linalg.norm(points, ord=2, dim=-1)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)

        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_range_vals, next_range_vals = range_vals[:, :-1], range_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_range_vals - prev_range_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([n_rays, 1], device=range_vals.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = next_range_vals - prev_range_vals
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([n_rays, 1], device=range_vals.device), 1.0 - alpha + 1e-7], dim=-1),
            dim=-1
        )[:, :-1]

        return self.sample_pdf(range_vals, weights, n_importance, det=True).detach()

    def cat_range_vals(self, ray_bases, ray_dir, range_vals, new_range_vals, sdf, sdf_network, last=False):
        n_rays, _ = range_vals.shape
        n_new = new_range_vals.shape[1]
        range_vals = torch.cat([range_vals, new_range_vals], dim=-1)
        range_vals, sort_index = torch.sort(range_vals, dim=-1)

        if last:
            return range_vals, sdf

        points = ray_bases[:, None, :] + new_range_vals[..., None] * ray_dir[None, None, :]
        new_sdf = sdf_network.sdf(points.reshape(-1, 3)).reshape(n_rays, n_new)
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        sdf = torch.gather(sdf, dim=1, index=sort_index)
        return range_vals, sdf

    def sample_pdf(self, bins, weights, n_samples, det=False):
        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if det:
            u = torch.linspace(
                0.0 + 0.5 / n_samples,
                1.0 - 0.5 / n_samples,
                steps=n_samples,
                device=bins.device,
                dtype=bins.dtype
            )
            u = u.expand(list(cdf.shape[:-1]) + [n_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=bins.device, dtype=bins.dtype)

        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], dim=-1)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), dim=2, index=inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), dim=2, index=inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        return bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    def _resolve_ray_counts(self, image_shape, n_samples=None):
        height, width = image_shape
        n_azimuth = width if self.n_azimuth is None else self.n_azimuth
        n_range = height if self.n_range is None else self.n_range
        n_height = max(2, width // 2) if self.n_height is None else self.n_height
        if n_samples is not None:
            n_range = int(n_samples)
        if n_azimuth < 2 or n_range < 2 or n_height < 2:
            raise ValueError('n_azimuth, n_range, and n_height must all be at least 2')
        return n_azimuth, n_range, n_height

    def _resample_axis(self, axis, n_samples, device, dtype):
        axis = axis.to(device=device, dtype=dtype)
        if axis.shape[0] == n_samples:
            return axis
        return torch.linspace(axis[0], axis[-1], n_samples, dtype=dtype, device=device)

    def generate_parallel_rays(self, frame_meta, image_shape, device, n_samples=None):
        n_azimuth, n_range, n_height = self._resolve_ray_counts(image_shape, n_samples=n_samples)
        dtype = torch.float32

        los = _meta_vector(frame_meta, 'radar_los').to(device=device, dtype=dtype)
        los = F.normalize(los, dim=0)
        ray_dir = -los

        z_axis = self.rot_axis.to(device=device, dtype=dtype)
        azimuth_basis = torch.cross(los, z_axis, dim=0)
        if torch.norm(azimuth_basis) < 1e-6:
            fallback_axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
            azimuth_basis = torch.cross(los, fallback_axis, dim=0)
        azimuth_basis = F.normalize(azimuth_basis, dim=0)
        height_basis = F.normalize(torch.cross(ray_dir, azimuth_basis, dim=0), dim=0)

        range_axis = frame_meta['range_axis']
        azimuth_axis = frame_meta['azimuth_axis']
        range_vals = self._resample_axis(range_axis, n_range, device, dtype) / self.coord_scale
        azimuth_vals = self._resample_axis(azimuth_axis, n_azimuth, device, dtype) / self.coord_scale
        height_vals = torch.linspace(-self.bound, self.bound, n_height, dtype=dtype, device=device)

        azimuth_spacing = torch.abs(azimuth_vals[1] - azimuth_vals[0])
        height_spacing = torch.abs(height_vals[1] - height_vals[0])
        ray_area = azimuth_spacing * height_spacing

        azimuth_grid, height_grid = torch.meshgrid(azimuth_vals, height_vals, indexing='ij')
        ray_bases = azimuth_grid.reshape(-1, 1) * azimuth_basis[None, :] + \
            height_grid.reshape(-1, 1) * height_basis[None, :]

        return ray_bases, ray_dir, range_vals, ray_area

    def compute_neus_weights(self, sdf, gradients, dirs, dists, variance_network, cos_anneal_ratio=1.0):
        n_rays, n_samples, _ = sdf.shape
        device = sdf.device
        if dists.dim() == 1:
            dists = dists.reshape(1, n_samples, 1)
        else:
            dists = dists.reshape(n_rays, n_samples, 1)

        inv_s = variance_network(torch.zeros([1, 1], device=device))[:, :1].clip(1e-6, 1e6)
        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
            F.relu(-true_cos) * cos_anneal_ratio
        )

        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        alpha = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clip(0.0, 1.0)

        alpha_flat = alpha.reshape(n_rays, n_samples)
        trans = torch.cumprod(
            torch.cat([torch.ones([n_rays, 1], device=device), 1.0 - alpha_flat + 1e-7], dim=-1),
            dim=-1
        )[:, :-1]
        weights = (alpha_flat * trans).reshape(n_rays, n_samples, 1)

        return alpha, weights, 1.0 / inv_s

    def render_bins(self, frame_meta, bins, sdf_network, variance_network,
                    image_shape=None, n_samples=None, cos_anneal_ratio=1.0):
        """
        渲染整幅图像，并提取指定像素位置的渲染值

        参数:
            bins: (B, 2) 采样像素坐标 [range_idx, azimuth_idx]
        返回:
            原 render_frame 的输出字典，附加 'bin_values' 键 (B, 1)
        """
        render_out = self.render_frame(
            frame_meta, sdf_network, variance_network,
            image_shape=image_shape, n_samples=n_samples,
            cos_anneal_ratio=cos_anneal_ratio
        )
        H, W = render_out['isar'].shape
        range_idx = bins[:, 0].long().clamp(0, H - 1)
        azimuth_idx = bins[:, 1].long().clamp(0, W - 1)
        render_out['bin_values'] = render_out['isar'][range_idx, azimuth_idx][:, None]
        return render_out

    def _sinc_sq(self, dx, sigma):
        """
        计算归一化 sinc² 函数值：sinc²( dx / sigma )
        其中 sinc(x) = sin(πx) / (πx)
        sigma 控制主瓣宽度（约 2*sigma 像素）
        """
        pi_x = math.pi * (dx / sigma)
        # 零点附近用 1 避免除零
        sinc = torch.where(pi_x.abs() < 1e-5,
                           torch.ones_like(pi_x),
                           torch.sin(pi_x) / pi_x)
        return sinc ** 2

    def _nearest_splat(self, range_bin, azimuth_bin, values, height, width):
        device = range_bin.device
        image = torch.zeros(height * width, dtype=values.dtype, device=device)

        range_idx = torch.round(range_bin).long()
        azimuth_idx = torch.round(azimuth_bin).long()
        valid = (range_idx >= 0) & (range_idx < height) & (azimuth_idx >= 0) & (azimuth_idx < width)

        flat_index = range_idx * width + azimuth_idx
        image.scatter_add_(0, flat_index[valid], values[valid])
        return image.reshape(height, width)

    def _bilinear_splat(self, range_bin, azimuth_bin, values, height, width):
        device = range_bin.device
        image = torch.zeros(height * width, dtype=values.dtype, device=device)

        r0 = torch.floor(range_bin).long()
        a0 = torch.floor(azimuth_bin).long()
        r1 = r0 + 1
        a1 = a0 + 1

        dr = range_bin - r0.float()
        da = azimuth_bin - a0.float()

        self._scatter_to_image(image, r0, a0, (1.0 - dr) * (1.0 - da) * values, height, width)
        self._scatter_to_image(image, r0, a1, (1.0 - dr) * da * values, height, width)
        self._scatter_to_image(image, r1, a0, dr * (1.0 - da) * values, height, width)
        self._scatter_to_image(image, r1, a1, dr * da * values, height, width)

        return image.reshape(height, width)

    def _scatter_to_image(self, image, range_idx, azimuth_idx, values, height, width):
        valid = (range_idx >= 0) & (range_idx < height) & (azimuth_idx >= 0) & (azimuth_idx < width)
        flat_index = range_idx * width + azimuth_idx
        image.scatter_add_(0, flat_index[valid], values[valid])

    def _apply_sinc_psf(self, image, frame_meta):
        device = image.device
        dtype = image.dtype

        range_axis = frame_meta['range_axis'].to(device=device, dtype=dtype)
        azimuth_axis = frame_meta['azimuth_axis'].to(device=device, dtype=dtype)
        range_spacing = torch.abs(range_axis[1] - range_axis[0]).clamp_min(1e-8)
        azimuth_spacing = torch.abs(azimuth_axis[1] - azimuth_axis[0]).clamp_min(1e-8)

        range_resolution = _meta_scalar(frame_meta, 'range_resolution').to(device=device, dtype=dtype)
        azimuth_resolution = _meta_scalar(frame_meta, 'azimuth_resolution').to(device=device, dtype=dtype)
        range_res_pix = (range_resolution / range_spacing).clamp_min(1e-6)
        azimuth_res_pix = (azimuth_resolution / azimuth_spacing).clamp_min(1e-6)

        # psf_lobes controls sinc² truncation:
        # 1 keeps the main lobe; 2 keeps main + first sidelobe; 3 keeps main + first/second sidelobes.
        radius_r = max(1, int(math.ceil(self.psf_lobes * range_res_pix.detach().cpu().item())))
        radius_a = max(1, int(math.ceil(self.psf_lobes * azimuth_res_pix.detach().cpu().item())))

        r = torch.arange(-radius_r, radius_r + 1, dtype=dtype, device=device)
        a = torch.arange(-radius_a, radius_a + 1, dtype=dtype, device=device)
        kr = self._sinc_sq(r, range_res_pix)
        ka = self._sinc_sq(a, azimuth_res_pix)
        kernel = kr[:, None] * ka[None, :]
        kernel = kernel / (kernel.sum() + 1e-8)

        image_4d = image[None, None, :, :]
        kernel_4d = kernel[None, None, :, :]
        return F.conv2d(image_4d, kernel_4d, padding=(radius_r, radius_a))[0, 0]

    def extract_geometry(self, sdf_network, bound_min, bound_max, resolution, threshold=0.0):
        """
        从 SDF 网络中提取等值面网格（用于最终可视化）

        参数:
            bound_min/max : 查询空间范围
            resolution    : 网格分辨率
            threshold     : 等值面阈值(0 对应表面)

        返回:
            mesh (顶点和面)
        """
        from models.renderer import extract_geometry
        device = next(sdf_network.parameters()).device

        return extract_geometry(
            bound_min, bound_max,
            resolution=resolution, threshold=threshold,
            query_func=lambda pts: -sdf_network.sdf(pts.to(device))  # 外部为正 SDF，需取反
        )
