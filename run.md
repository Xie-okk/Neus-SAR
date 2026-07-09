# ISAR 训练/验证运行命令

以下命令默认在项目根目录运行：

```powershell
cd D:\A_master\AAA组会\AAAA研究方向\NeuS-main
```

Python 环境：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe
```

默认目标：`1999JV6`  
默认配置：`confs/isar_fuyan.conf`

---

## 1. 从头开始训练

确认配置里：

```hocon
train {
    end_iter = 1000
    save_freq = 200
}
```

然后运行：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode train --case 1999JV6
```

从头训练时不要加 `--is_continue`，也不要加 `--checkpoint`。

训练过程中会按配置保存：

```text
exp/1999JV6/checkpoints/ckpt_XXXXXX.pth
exp/1999JV6/validations/iter_XXXXXX_frames_*.png
exp/1999JV6/sdf_z_planes/sdf_z0_XXXXXX.png
exp/1999JV6/meshes/mesh_XXXXXX.ply
exp/1999JV6/logs/train_metrics.csv
exp/1999JV6/logs/loss_curves.png
```

---

## 2. 从断点继续训练

### 2.1 自动读取最新断点

程序会自动读取：

```text
exp/1999JV6/checkpoints/
```

里满足 `iter <= train.end_iter` 的最新 checkpoint。

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode train --case 1999JV6 --is_continue
```

注意：如果你想从 `ckpt_000800.pth` 继续，配置里的：

```hocon
train.end_iter
```

必须大于等于 `800`，否则自动读取最新断点时它会被跳过。

### 2.2 读取指定断点继续训练

指定 checkpoint 文件名：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode train --case 1999JV6 --checkpoint ckpt_000400.pth
```

也可以指定完整路径：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode train --case 1999JV6 --checkpoint D:\A_master\AAA组会\AAAA研究方向\NeuS-main\exp\1999JV6\checkpoints\ckpt_000400.pth
```

指定 `--checkpoint` 后，不需要再加 `--is_continue`。如果两者都加，优先使用 `--checkpoint` 指定的断点。

注意：继续训练时 `train.end_iter` 要大于 checkpoint 里的 `iter_step`。例如读取 `ckpt_000400.pth` 后，`end_iter` 必须大于 `400`，否则程序会认为已经训练结束，只保存 final 结果。

---

## 3. 读取断点，只验证图像

不训练，只加载 checkpoint 并输出验证拼接图。

### 3.1 自动读取最新断点，验证所有图像

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode validate_views --case 1999JV6 --is_continue --view_ids all
```

输出位置类似：

```text
exp/1999JV6/validations/iter_XXXXXX_frames_all.png
```

### 3.2 自动读取最新断点，验证指定图像

只看第 0 帧：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode validate_views --case 1999JV6 --is_continue --view_ids 0
```

看多帧：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode validate_views --case 1999JV6 --is_continue --view_ids 0,6,12
```

输出位置类似：

```text
exp/1999JV6/validations/iter_XXXXXX_frames_0.png
exp/1999JV6/validations/iter_XXXXXX_frames_0_6_12.png
```

### 3.3 读取指定断点，只验证图像

指定断点并验证全部图像：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode validate_views --case 1999JV6 --checkpoint ckpt_000400.pth --view_ids all
```

指定断点并验证部分图像：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode validate_views --case 1999JV6 --checkpoint ckpt_000400.pth --view_ids 0,6,12
```

也可以在配置里设置默认验证视角：

```hocon
validate {
    view_ids = "all"      # all 或 0 或 0,6,12
}
```

然后运行：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode validate_views --case 1999JV6 --is_continue
```

---

## 4. 读取断点，只导出 mesh

### 4.1 自动读取最新断点导出 mesh

使用配置里的 `validate.mesh_resolution`：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode validate_mesh --case 1999JV6 --is_continue
```

临时覆盖 mesh 分辨率，例如 `128`：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode validate_mesh --case 1999JV6 --is_continue --mesh_resolution 128
```

临时覆盖 marching cubes 阈值：

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode validate_mesh --case 1999JV6 --is_continue --mcube_threshold 0.0
```

### 4.2 读取指定断点导出 mesh

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py --conf .\confs\isar_fuyan.conf --mode validate_mesh --case 1999JV6 --checkpoint ckpt_000400.pth --mesh_resolution 128
```

输出位置：

```text
exp/1999JV6/meshes/mesh_XXXXXX.ply
```

---

## 5. 训练结束或 Ctrl+C 打断时自动保存最后结果

当前配置支持训练正常结束或手动 `Ctrl+C` 打断时保存最后结果：

```hocon
validate {
    export_final_checkpoint = true
    export_final_image = true
    export_final_sdf = true
    export_final_mesh = true
    final_view_ids = "all"
}
```

会保存：

```text
exp/1999JV6/checkpoints/ckpt_XXXXXX.pth
exp/1999JV6/validations/iter_XXXXXX_frames_all.png
exp/1999JV6/sdf_z_planes/sdf_z0_XXXXXX.png
exp/1999JV6/meshes/mesh_XXXXXX.ply
exp/1999JV6/logs/loss_curves.png
```

注意：这个功能只对重新启动后的训练进程生效。已经在旧代码下启动的训练，不会自动执行新的收尾逻辑。

---

## 6. 常用配置项

训练长度和保存频率：

```hocon
train {
    end_iter = 1000
    save_freq = 200
    val_freq = 20
    val_mesh_freq = 100
    val_sdf_freq = 20
    metrics_plot_freq = 20
}
```

验证视角：

```hocon
validate {
    view_ids = "all"      # all 或 0 或 0,6,12
    final_view_ids = "all"
}
```

验证/最终出图时的高度采样：

```hocon
validate {
    n_height = 64
}
```

训练时高度采样调度：

```hocon
train {
    n_height_schedule = [
        { iter = 0, value = 16 },
        { iter = 201, value = 32 },
        { iter = 401, value = 64 }
    ]
}
```