import os
import csv
import random
import logging
import warnings
import argparse
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from shutil import copyfile
from pyhocon import ConfigFactory
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.isar_dataset import ISARDataset
from models.fields import SDFNetwork, SingleVarianceNetwork
from models.isar_renderer import ISARRenderer

warnings.filterwarnings(
    'ignore',
    message=r'`torch\.nn\.utils\.weight_norm` is deprecated.*',
    category=FutureWarning
)
warnings.filterwarnings(
    'ignore',
    message=r'torch\.meshgrid: in an upcoming release.*',
    category=UserWarning
)
def plot_sdf_z_plane(sdf_network, device, resolution=100, output_path='sdf_z0.png'):
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.flatten(), Y.flatten(), np.zeros_like(X).flatten()], axis=-1)
    pts_tensor = torch.tensor(pts, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        out = sdf_network(pts_tensor)
        sdf_vals = (out[0] if isinstance(out, tuple) else out)[:, :1].cpu().numpy().reshape(resolution, resolution)
    
    # 颜色映射的核心：计算一个合理的范围
    vmax = np.max(np.abs(sdf_vals))
    vmin = -vmax
    
    plt.figure(figsize=(7, 6))
    
    # 1. 填充颜色：使用 RdBu_r 映射，红色代表负值（内部），蓝色代表正值（外部）
    # 如果数据单一，我们通过 vmin/vmax 强行固定映射区间
    cf = plt.contourf(X, Y, sdf_vals, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(cf, label='SDF Value')
    
    # 2. 绘制零水平集 (SDF=0 的轮廓线)
    cs = plt.contour(X, Y, sdf_vals, levels=[0], colors='black', linewidths=2)
    # plt.clabel(cs, inline=True, fontsize=10, fmt='Surface')
    
    plt.title('SDF Cross-section at Z=0 (Color Mapped)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.3)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    print(f"SDF z=0 plane saved to {output_path}")
    plt.close()

def parse_view_ids(view_ids, n_images):
    if view_ids.lower() == 'all':
        return list(range(n_images))

    ids = []
    for item in view_ids.split(','):
        item = item.strip()
        if item == '':
            continue
        idx = int(item)
        if idx < 0 or idx >= n_images:
            raise ValueError(f'view id {idx} is out of range [0, {n_images - 1}]')
        ids.append(idx)
    if len(ids) == 0:
        raise ValueError('view_ids must contain at least one index')
    return ids


class ISARRunner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, checkpoint_name=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 读取配置文件
        self.conf_path = conf_path
        with open(self.conf_path, 'r', encoding='utf-8-sig') as f:
            conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.seed = self.conf.get_int('train.seed', default=20240708)
        self.set_random_seed(self.seed)

        # 实验目录
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)

        # 数据集
        self.dataset = ISARDataset(self.conf['dataset'])
        self.iter_step = 0

        # 训练参数
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.val_sdf_freq = self.conf.get_int('train.val_sdf_freq', default=self.val_mesh_freq)
        self.metrics_plot_freq = self.conf.get_int('train.metrics_plot_freq', default=self.val_sdf_freq)
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.image_weight = self.conf.get_float('train.image_weight', default=1.0)
        self.image_loss_type = self.conf.get_string('train.image_loss_type', default='l1').lower()
        self.huber_beta = self.conf.get_float('train.huber_beta', default=0.1)
        self.charbonnier_eps = self.conf.get_float('train.charbonnier_eps', default=1e-3)
        self.igr_weight = self.conf.get_float('train.igr_weight', default=0.1)
        self.export_init_mesh = self.conf.get_bool('validate.export_init_mesh', default=True)
        self.export_init_image = self.conf.get_bool('validate.export_init_image', default=False)
        self.export_init_sdf = self.conf.get_bool('validate.export_init_sdf', default=True)
        self.export_final_checkpoint = self.conf.get_bool('validate.export_final_checkpoint', default=True)
        self.export_final_image = self.conf.get_bool('validate.export_final_image', default=True)
        self.export_final_sdf = self.conf.get_bool('validate.export_final_sdf', default=True)
        self.export_final_mesh = self.conf.get_bool('validate.export_final_mesh', default=False)
        self.mesh_resolution = self.conf.get_int('validate.mesh_resolution', default=64)
        self.final_results_saved = False
        self.checkpoint_name = checkpoint_name
        self.is_continue = is_continue or checkpoint_name is not None
        self.mode = mode
        self.writer = None
        self.metrics_path = os.path.join(self.base_exp_dir, 'logs', 'train_metrics.csv')
        self.n_height_schedule = self.parse_n_height_schedule()
        self.validate_n_height = self.get_optional_int('validate.n_height', default=None)

        # 初始化网络
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)

        self.variance_network = SingleVarianceNetwork(
            init_val=self.conf.get_float('model.variance_network.init_val', 50.0)
        ).to(self.device)

        # 优化器
        params_to_train = list(self.sdf_network.parameters()) + \
                          list(self.variance_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        # 渲染器（使用我们最终极简版本）
        self.renderer = ISARRenderer(**self.conf['model.isar_renderer'])

        # 断点续训/验证
        latest_model_name = None
        if checkpoint_name is not None:
            latest_model_name = checkpoint_name
        elif is_continue:
            checkpoint_dir = os.path.join(self.base_exp_dir, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                model_list = []
                for fname in os.listdir(checkpoint_dir):
                    if fname.endswith('.pth'):
                        try:
                            iter_num = int(fname.split('_')[1].split('.')[0])
                            if iter_num <= self.end_iter:
                                model_list.append((iter_num, fname))
                        except:
                            pass
                if model_list:
                    model_list.sort(key=lambda x: x[0])
                    latest_model_name = model_list[-1][1]

        if latest_model_name is not None:
            logging.info(f'Loading checkpoint: {latest_model_name}')
            self.load_checkpoint(latest_model_name)

        self.apply_mode_n_height()

        # 备份代码
        if self.mode.startswith('train'):
            self.file_backup()

    def set_random_seed(self, seed):
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f'Random seed set to {seed}')
    # ===================== 训练 =====================
    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.init_metrics_log()
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        self.update_train_n_height()

        if self.iter_step == 0 and self.export_init_mesh:
            self.validate_mesh(resolution=self.mesh_resolution)

        if self.iter_step == 0 and self.export_init_image:
            validate_view_ids = self.get_validate_view_ids()
            self.validate_views(
                parse_view_ids(validate_view_ids, self.dataset.n_images),
                view_token=validate_view_ids
            )

        if self.iter_step == 0 and self.export_init_sdf:
            self.validate_sdf_z_plane()

        for _ in tqdm(range(res_step)):
            self.update_train_n_height(report_change=True)
            frame_idx = image_perm[self.iter_step % len(image_perm)].item()
            target_image, frame_meta = self.dataset.get_frame(frame_idx)
            cos_anneal_ratio = self.get_cos_anneal_ratio()

            render_out = self.renderer.render_frame(
                frame_meta,
                self.sdf_network,
                self.variance_network,
                image_shape=target_image.shape,
                cos_anneal_ratio=cos_anneal_ratio
            )

            pred_image = render_out['isar']
            pred_image_norm = pred_image / (pred_image.mean().detach() + 1e-6)
            target_image_norm = target_image / (target_image.mean().detach() + 1e-6)
            image_loss_raw = self.compute_image_loss(pred_image_norm, target_image_norm)
            
            eikonal_loss_raw = render_out['gradient_error']
            image_loss = self.image_weight * image_loss_raw
            eikonal_loss = self.igr_weight * eikonal_loss_raw
            loss = image_loss + eikonal_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1
            with torch.no_grad():
                # 既然你目前用的是官方原版（带有 * 10.0 的魔法），使用以下这行：
                current_inv_s = torch.exp(self.variance_network.variance * 10.0).item()
                # 如果你以后换回了你的纯净重构版（去掉了 * 10.0），请改为：
                # current_inv_s = torch.exp(self.variance_network.log_s).item()
            # -----------------------------
            # TensorBoard 记录
            self.writer.add_scalar('Loss/total', loss.item(), self.iter_step)
            self.writer.add_scalar('Loss/image', image_loss.item(), self.iter_step)
            self.writer.add_scalar('Loss/eikonal', eikonal_loss.item(), self.iter_step)
            self.writer.add_scalar('LossRaw/image', image_loss_raw.item(), self.iter_step)
            self.writer.add_scalar('LossRaw/eikonal', eikonal_loss_raw.item(), self.iter_step)
            self.writer.add_scalar('Statistics/cos_anneal_ratio', cos_anneal_ratio, self.iter_step)
            self.writer.add_scalar('Statistics/inv_s', current_inv_s, self.iter_step)
            self.writer.add_scalar('Statistics/n_height', self.renderer.n_height, self.iter_step)

            train_metrics = self.collect_train_metrics(
                frame_idx,
                loss,
                image_loss,
                eikonal_loss,
                image_loss_raw,
                eikonal_loss_raw,
                cos_anneal_ratio,
                current_inv_s,
                target_image,
                render_out
            )
            self.append_train_metrics(train_metrics)

            if self.iter_step % self.report_freq == 0:
                print(f"[iter {self.iter_step}] loss={loss.item():.6f} "
                      f"img_w={image_loss.item():.6f} eik_w={eikonal_loss.item():.6f} "
                      f"img_raw={image_loss_raw.item():.6f} eik_raw={eikonal_loss_raw.item():.6f} "
                      f"cos={cos_anneal_ratio:.3f} lr={self.optimizer.param_groups[0]['lr']:.2e} "
                      f"inv_s={current_inv_s:.2f}") 

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                validate_view_ids = self.get_validate_view_ids()
                self.validate_views(
                    parse_view_ids(validate_view_ids, self.dataset.n_images),
                    view_token=validate_view_ids
                )

            if self.val_mesh_freq > 0 and self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(resolution=self.mesh_resolution)

            if self.val_sdf_freq > 0 and self.iter_step % self.val_sdf_freq == 0:
                self.validate_sdf_z_plane()

            if self.metrics_plot_freq > 0 and self.iter_step % self.metrics_plot_freq == 0:
                self.plot_train_metrics_curves()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    # ===================== 辅助方法 =====================
    def get_optional_int(self, key, default=None):
        try:
            return self.conf.get_int(key)
        except Exception:
            return default

    def get_optional_config(self, key, default=None):
        try:
            return self.conf.get(key)
        except Exception:
            return default

    def parse_n_height_schedule(self):
        raw_schedule = self.get_optional_config('train.n_height_schedule', default=None)
        if raw_schedule is None:
            return []
        schedule = []
        for item in raw_schedule:
            try:
                start_iter = int(item['iter'])
                n_height = int(item['value'])
            except (TypeError, KeyError):
                start_iter = int(item[0])
                n_height = int(item[1])
            if start_iter < 0 or n_height < 2:
                raise ValueError('n_height_schedule entries must have iter >= 0 and value >= 2')
            schedule.append((start_iter, n_height))
        schedule.sort(key=lambda entry: entry[0])
        return schedule

    def scheduled_n_height(self, iter_step):
        if len(self.n_height_schedule) == 0:
            return None
        selected = self.n_height_schedule[0][1]
        for start_iter, n_height in self.n_height_schedule:
            if iter_step >= start_iter:
                selected = n_height
            else:
                break
        return selected

    def set_renderer_n_height(self, n_height, report_change=False):
        if n_height is None:
            return
        n_height = int(n_height)
        old_n_height = self.renderer.n_height
        self.renderer.n_height = n_height
        if report_change and old_n_height != n_height:
            print(f'n_height changed: {old_n_height} -> {n_height} at iter {self.iter_step}')

    def update_train_n_height(self, report_change=False):
        self.set_renderer_n_height(self.scheduled_n_height(self.iter_step), report_change=report_change)

    def apply_mode_n_height(self):
        if self.mode == 'train':
            self.update_train_n_height()
        else:
            self.set_renderer_n_height(self.validate_n_height)

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)
    def compute_image_loss(self, pred_image_norm, target_image_norm):
        if self.image_loss_type == 'l1':
            return F.l1_loss(pred_image_norm, target_image_norm)
        if self.image_loss_type in ('huber', 'smooth_l1'):
            return F.smooth_l1_loss(
                pred_image_norm,
                target_image_norm,
                beta=self.huber_beta
            )
        if self.image_loss_type == 'charbonnier':
            diff = pred_image_norm - target_image_norm
            return torch.sqrt(diff * diff + self.charbonnier_eps * self.charbonnier_eps).mean()
        raise ValueError(
            "train.image_loss_type must be one of: l1, huber, smooth_l1, charbonnier"
        )

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end and self.warm_up_end > 0:
            factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = min(1.0, (self.iter_step - self.warm_up_end) /
                           max(1.0, self.end_iter - self.warm_up_end))
            factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * factor

    def get_cos_anneal_ratio(self):
        if self.anneal_end <= 0.0:
            return 1.0
        return min(1.0, self.iter_step / self.anneal_end)

    def get_validate_view_ids(self):
        return self.conf.get_string('validate.view_ids', default='0')

    def get_final_view_ids(self):
        return self.conf.get_string('validate.final_view_ids', default=self.get_validate_view_ids())

    def save_final_results(self, reason='finished'):
        if self.final_results_saved:
            return
        self.final_results_saved = True
        print(f'Saving final results at iter {self.iter_step} ({reason})...')

        if self.export_final_checkpoint:
            self.save_checkpoint()

        old_n_height = self.renderer.n_height
        self.set_renderer_n_height(self.validate_n_height)
        try:
            if self.export_final_image:
                final_view_ids = self.get_final_view_ids()
                self.validate_views(
                    parse_view_ids(final_view_ids, self.dataset.n_images),
                    view_token=final_view_ids
                )

            if self.export_final_sdf:
                self.validate_sdf_z_plane()

            if self.export_final_mesh:
                self.validate_mesh(resolution=self.mesh_resolution)

            self.plot_train_metrics_curves()
        finally:
            self.renderer.n_height = old_n_height
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
                self.writer = None
    def get_mesh_resolution(self):
        return self.mesh_resolution

    def file_backup(self):
        dirs = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for d in dirs:
            target_dir = os.path.join(self.base_exp_dir, 'recording', d)
            os.makedirs(target_dir, exist_ok=True)
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith('.py'):
                        copyfile(os.path.join(d, f), os.path.join(target_dir, f))
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        if os.path.isabs(checkpoint_name):
            path = checkpoint_name
        else:
            path = os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(path, map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network'])
        self.variance_network.load_state_dict(checkpoint['variance_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        logging.info(f'Checkpoint loaded (iter {self.iter_step})')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network': self.sdf_network.state_dict(),
            'variance_network': self.variance_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        path = os.path.join(self.base_exp_dir, 'checkpoints',
                            f'ckpt_{self.iter_step:06d}.pth')
        torch.save(checkpoint, path)

    def metric_fieldnames(self):
        return [
            'iter', 'frame_idx', 'loss', 'image_loss', 'eikonal_loss',
            'image_loss_raw', 'eikonal_loss_raw', 'image_weight', 'igr_weight',
            'lr', 'inv_s', 'cos_anneal_ratio', 'n_height',
            'target_mean', 'target_max', 'pred_mean', 'pred_max',
            'alpha_mean', 'alpha_max', 'weight_mean', 'weight_max',
            'point_weight_mean', 'point_weight_max', 'sdf_min', 'sdf_max'
        ]

    def init_metrics_log(self):
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        if (not self.is_continue) and self.iter_step == 0:
            mode = 'w'
        else:
            mode = 'a' if os.path.exists(self.metrics_path) else 'w'

        if mode == 'a' and not self.metrics_header_matches(self.metrics_path):
            root, ext = os.path.splitext(self.metrics_path)
            self.metrics_path = root + '_weighted' + ext
            mode = 'a' if os.path.exists(self.metrics_path) else 'w'

        if mode == 'w':
            with open(self.metrics_path, mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.metric_fieldnames())
                writer.writeheader()

    def metrics_header_matches(self, path):
        try:
            with open(path, 'r', newline='', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                header = next(reader, None)
        except OSError:
            return False
        return header == self.metric_fieldnames()
    def tensor_stat(self, tensor, reducer):
        if tensor is None:
            return float('nan')
        return float(reducer(tensor.detach()).item())

    def collect_train_metrics(self, frame_idx, loss, image_loss, eikonal_loss,
                              image_loss_raw, eikonal_loss_raw,
                              cos_anneal_ratio, current_inv_s, target_image, render_out):
        pred_image = render_out.get('isar')
        alpha = render_out.get('alpha')
        weights = render_out.get('weights')
        point_weight = render_out.get('point_weight')
        sdf = render_out.get('sdf')
        sdf_min = render_out.get('sdf_min')
        sdf_max = render_out.get('sdf_max')
        return {
            'iter': self.iter_step,
            'frame_idx': int(frame_idx),
            'loss': float(loss.detach().item()),
            'image_loss': float(image_loss.detach().item()),
            'eikonal_loss': float(eikonal_loss.detach().item()),
            'image_loss_raw': float(image_loss_raw.detach().item()),
            'eikonal_loss_raw': float(eikonal_loss_raw.detach().item()),
            'image_weight': float(self.image_weight),
            'igr_weight': float(self.igr_weight),
            'lr': float(self.optimizer.param_groups[0]['lr']),
            'inv_s': float(current_inv_s),
            'cos_anneal_ratio': float(cos_anneal_ratio),
            'n_height': self.renderer.n_height,
            'target_mean': self.tensor_stat(target_image, torch.mean),
            'target_max': self.tensor_stat(target_image, torch.max),
            'pred_mean': self.tensor_stat(pred_image, torch.mean),
            'pred_max': self.tensor_stat(pred_image, torch.max),
            'alpha_mean': self.tensor_stat(alpha, torch.mean),
            'alpha_max': self.tensor_stat(alpha, torch.max),
            'weight_mean': self.tensor_stat(weights, torch.mean),
            'weight_max': self.tensor_stat(weights, torch.max),
            'point_weight_mean': self.tensor_stat(point_weight, torch.mean),
            'point_weight_max': self.tensor_stat(point_weight, torch.max),
            'sdf_min': self.tensor_stat(sdf_min, torch.min),
            'sdf_max': self.tensor_stat(sdf_max, torch.max),
        }

    def append_train_metrics(self, metrics):
        with open(self.metrics_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.metric_fieldnames())
            writer.writerow(metrics)

    def plot_train_metrics_curves(self):
        if not os.path.exists(self.metrics_path):
            return

        rows = []
        with open(self.metrics_path, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('iter') not in (None, ''):
                    rows.append(row)
        if len(rows) == 0:
            return

        def values(name):
            return [float(row[name]) for row in rows]

        iters = [int(row['iter']) for row in rows]
        fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

        axes[0].plot(iters, values('loss'), label='loss')
        axes[0].plot(iters, values('image_loss'), label='image_loss_weighted')
        axes[0].plot(iters, values('eikonal_loss'), label='eikonal_loss_weighted')
        axes[0].set_ylabel('loss')
        axes[0].legend()

        axes[1].plot(iters, values('image_loss'), color='tab:blue', label='image_loss_weighted')
        axes[1].set_ylabel('image')
        axes[1].legend()

        axes[2].plot(iters, values('eikonal_loss'), color='tab:green', label='eikonal_loss_weighted')
        axes[2].set_ylabel('eikonal')
        axes[2].legend()

        axes[3].plot(iters, values('inv_s'), color='tab:red', label='inv_s')
        axes[3].set_ylabel('inv_s')
        axes[3].legend()

        axes[4].plot(iters, values('sdf_min'), label='sdf_min')
        axes[4].plot(iters, values('sdf_max'), label='sdf_max')
        axes[4].axhline(0.0, color='black', linewidth=0.8, alpha=0.5)
        axes[4].set_ylabel('sdf')
        axes[4].set_xlabel('iter')
        axes[4].legend()


        for ax in axes:
            ax.grid(True, alpha=0.3)

        fig.suptitle('Training metrics')
        fig.tight_layout()
        out_path = os.path.join(self.base_exp_dir, 'logs', 'loss_curves.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    # ===================== 验证 =====================
    def _normalize_validation_image(self, img):
        img = np.asarray(img, dtype=np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = np.clip(img, 0.0, None)
        vmax = np.max(img)
        if vmax <= 1e-8:
            return np.zeros_like(img, dtype=np.uint8)
        return (np.clip(img / vmax, 0.0, 1.0) * 255).astype(np.uint8)

    def _render_validation_pair(self, idx):
        target_image, frame_meta = self.dataset.get_frame(idx)
        render_out = self.renderer.render_frame(
            frame_meta,
            self.sdf_network,
            self.variance_network,
            image_shape=target_image.shape
        )

        pred = render_out['isar'].detach().cpu().numpy()
        target = target_image.detach().cpu().numpy()
        return np.concatenate([
            self._normalize_validation_image(target),
            self._normalize_validation_image(pred)
        ], axis=1)

    def validate_image(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.dataset.n_images)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations'), exist_ok=True)
        out_path = os.path.join(self.base_exp_dir, 'validations',
                                f'iter_{self.iter_step:06d}_frame_{idx}.png')
        cv.imwrite(out_path, self._render_validation_pair(idx))
        print(f"Validation image saved to {out_path}")

    def validate_views(self, view_ids, view_token=None):
        if len(view_ids) == self.dataset.n_images:
            view_desc = 'all'
        else:
            view_desc = ','.join(str(idx) for idx in view_ids)
        if len(view_ids) > 1:
            iterator = tqdm(view_ids,
                            desc=f"Validating {len(view_ids)} view(s) [{view_desc}] iter {self.iter_step}",
                            leave=True)
        else:
            print(f"Validating 1 view [{view_desc}] at iter {self.iter_step}...")
            iterator = view_ids

        rows = []
        for idx in iterator:
            rows.append(self._render_validation_pair(idx))
        max_width = max(row.shape[1] for row in rows)
        padded_rows = []
        for row in rows:
            if row.shape[1] < max_width:
                pad = np.zeros((row.shape[0], max_width - row.shape[1]), dtype=row.dtype)
                row = np.concatenate([row, pad], axis=1)
            padded_rows.append(row)

        canvas = np.concatenate(padded_rows, axis=0)
        os.makedirs(os.path.join(self.base_exp_dir, 'validations'), exist_ok=True)
        if view_token is None:
            view_token = '_'.join(str(idx) for idx in view_ids)
        else:
            view_token = view_token.strip().replace(',', '_')
        out_path = os.path.join(self.base_exp_dir, 'validations',
                                f'iter_{self.iter_step:06d}_frames_{view_token}.png')
        cv.imwrite(out_path, canvas)
        print(f"Validation views saved to {out_path}")

    def validate_sdf_z_plane(self):
        sdf_plane_path = os.path.join(
            self.base_exp_dir,
            'sdf_z_planes',
            f'sdf_z0_{self.iter_step:06d}.png'
        )
        plot_sdf_z_plane(self.sdf_network, self.device, output_path=sdf_plane_path)

    def validate_mesh(self, resolution=256, threshold=0.0):
        print(f"Extracting mesh at iter {self.iter_step} "
              f"(resolution={resolution}, threshold={threshold})...")
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32, device=self.device)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32, device=self.device)

        vertices, triangles = self.renderer.extract_geometry(
            self.sdf_network, bound_min, bound_max,
            resolution=resolution, threshold=threshold
        )

        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        mesh_path = os.path.join(self.base_exp_dir, 'meshes',
                                 f'mesh_{self.iter_step:06d}.ply')
        # 乘以放大倍数 coord_scale，将归一化空间坐标转换为物理空间（米）
        vertices_scaled = vertices * self.renderer.coord_scale

        mesh = trimesh.Trimesh(vertices_scaled, triangles)
        mesh.export(mesh_path)
        logging.info(f'Mesh saved to {mesh_path}')
        print(f"Mesh saved to {mesh_path} "
              f"(vertices={len(vertices)}, triangles={len(triangles)}, "
              f"resolution={resolution}, threshold={threshold})")



if __name__ == '__main__':
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.WARNING, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/isar_fuyan.conf')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'validate_image', 'validate_views', 'validate_mesh'])
    parser.add_argument('--case', type=str, default='1999JV6')
    parser.add_argument('--is_continue', default=False, action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load a specific checkpoint, e.g. ckpt_000400.pth or a full path.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--mesh_resolution', type=int, default=None,
                        help='Override validate.mesh_resolution for validate_mesh.')
    parser.add_argument('--view_ids', type=str, default=None,
                        help='Override validate.view_ids for validate_views, e.g. 0,6,12, or all.')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    runner = ISARRunner(args.conf, args.mode, args.case, args.is_continue, args.checkpoint)
    if args.mode == 'train':
        try:
            runner.train()
        except KeyboardInterrupt:
            print('Training interrupted by user.')
            runner.save_final_results(reason='interrupted')
        else:
            runner.save_final_results(reason='finished')
    elif args.mode == 'validate_image':
        runner.validate_image()
    elif args.mode == 'validate_views':
        view_ids = args.view_ids if args.view_ids is not None else runner.get_validate_view_ids()
        runner.validate_views(parse_view_ids(view_ids, runner.dataset.n_images), view_token=view_ids)
    elif args.mode == 'validate_mesh':
        mesh_resolution = args.mesh_resolution if args.mesh_resolution is not None else runner.get_mesh_resolution()
        runner.validate_mesh(resolution=mesh_resolution, threshold=args.mcube_threshold)

