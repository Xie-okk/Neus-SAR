import os
from PIL import Image

def convert_left_right_to_up_down(input_path, output_name=None):
    """
    将左右对比图（每行左真右预）转为上下对比，并重排为2行9列。
    :param input_path: 输入图片路径
    :param output_name: 输出文件名（不含后缀），默认在原文件名后加 '_updown'
    """
    # 打开原图
    img = Image.open(input_path)
    width, height = img.size

    # 固定参数（可根据实际调整）
    tile_size = 64          # 每个子图边长
    n_rows_original = 18    # 原图行数（即样本数）
    n_cols_original = 2     # 原图列数（左真右预）

    # 校验尺寸（可选）
    expected_w = tile_size * n_cols_original
    expected_h = tile_size * n_rows_original
    if width != expected_w or height != expected_h:
        print(f"警告：输入图尺寸为 {width}x{height}，预期为 {expected_w}x{expected_h}，将按实际尺寸处理。")
        # 仍然按 tile_size 进行裁剪，但会以左上角为准

    # 1. 裁剪所有子图，并构建上下对
    up_down_pairs = []
    for row in range(n_rows_original):
        # 真值 (左)
        left = img.crop((0, row * tile_size, tile_size, (row + 1) * tile_size))
        # 预测 (右)
        right = img.crop((tile_size, row * tile_size, 2 * tile_size, (row + 1) * tile_size))
        
        # 上下拼接：真值在上，预测在下 (可调换顺序)
        pair = Image.new('RGB', (tile_size, tile_size * 2))
        pair.paste(left, (0, 0))
        pair.paste(right, (0, tile_size))
        up_down_pairs.append(pair)

    # 2. 重排为 2 行 x 9 列
    n_rows_new = 2
    n_cols_new = 9
    if len(up_down_pairs) != n_rows_new * n_cols_new:
        print(f"警告：总对数 {len(up_down_pairs)} 不等于 {n_rows_new}x{n_cols_new}，将按实际数量填充。")
        # 仍可处理，但可能留白

    # 计算最终画布尺寸
    tile_w = tile_size
    tile_h = tile_size * 2   # 上下拼接后高度
    canvas_w = n_cols_new * tile_w
    canvas_h = n_rows_new * tile_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))

    # 填充网格
    for idx, pair in enumerate(up_down_pairs):
        row = idx // n_cols_new
        col = idx % n_cols_new
        if row >= n_rows_new:
            break  # 超出部分忽略
        x = col * tile_w
        y = row * tile_h
        canvas.paste(pair, (x, y))

    # 3. 保存结果
    if output_name is None:
        base, ext = os.path.splitext(input_path)
        output_name = base + '_updown' + ext
    else:
        if not output_name.endswith(('.png', '.jpg', '.jpeg')):
            output_name += '.png'
        output_name = os.path.join(os.path.dirname(input_path), output_name)

    canvas.save(output_name)
    print(f"转换完成！结果保存为：{output_name}")
    return output_name

if __name__ == "__main__":
    # 请修改为您的实际路径
    input_img = r"D:\A_master\AAA组会\AAAA研究方向\网络重建\NeuS-main\iter_001000_frames_all.png"
    convert_left_right_to_up_down(input_img)