%% 批量裁剪：54个自定义矩形区域 (尺寸强制统一 w=108, h=79)，保存为 PNG
clear; clc; close all;
input_file = '2024ON_Sep16_3.75m_0.1 Hz.jpg';
output_dir = 'D:\A_master\AAA组会\AAAA研究方向\网络重建\NeuS-main\public_data\2024ON\image';   % 保存目录

% 读取图像
if ~exist(input_file, 'file')
    error('文件 %s 不存在，请检查路径。', input_file);
end
img = imread(input_file);
[img_h, img_w, ~] = size(img);
fprintf('原始图像尺寸：宽=%d, 高=%d\n', img_w, img_h);

%% 定义 54 个裁剪区域 (宽高统一设定为 w=108, h=79)
w_val = 107;
h_val = 79;

regions = [
    % === 第 1 行 (y0 = 37) ===
    struct('x0', 3,   'y0', 37,  'w', w_val, 'h', h_val);   % 000
    struct('x0', 111, 'y0', 37,  'w', w_val, 'h', h_val);   % 001
    struct('x0', 219, 'y0', 37,  'w', w_val, 'h', h_val);   % 002
    struct('x0', 327, 'y0', 37,  'w', w_val, 'h', h_val);   % 003
    struct('x0', 435, 'y0', 37,  'w', w_val, 'h', h_val);   % 004
    struct('x0', 543, 'y0', 37,  'w', w_val, 'h', h_val);   % 005
    struct('x0', 651, 'y0', 37,  'w', w_val, 'h', h_val);   % 006
    struct('x0', 759, 'y0', 37,  'w', w_val, 'h', h_val);   % 007
    
    % === 第 2 行 (y0 = 116) ===
    struct('x0', 3,   'y0', 116, 'w', w_val, 'h', h_val);   % 008
    struct('x0', 111, 'y0', 116, 'w', w_val, 'h', h_val);   % 009
    struct('x0', 219, 'y0', 116, 'w', w_val, 'h', h_val);   % 010
    struct('x0', 327, 'y0', 116, 'w', w_val, 'h', h_val);   % 011
    struct('x0', 435, 'y0', 116, 'w', w_val, 'h', h_val);   % 012
    struct('x0', 543, 'y0', 116, 'w', w_val, 'h', h_val);   % 013
    struct('x0', 651, 'y0', 116, 'w', w_val, 'h', h_val);   % 014
    struct('x0', 759, 'y0', 116, 'w', w_val, 'h', h_val);   % 015
    
    % === 第 3 行 (y0 = 195) ===
    struct('x0', 3,   'y0', 195, 'w', w_val, 'h', h_val);   % 016
    struct('x0', 111, 'y0', 195, 'w', w_val, 'h', h_val);   % 017
    struct('x0', 219, 'y0', 195, 'w', w_val, 'h', h_val);   % 018
    struct('x0', 327, 'y0', 195, 'w', w_val, 'h', h_val);   % 019
    struct('x0', 435, 'y0', 195, 'w', w_val, 'h', h_val);   % 020
    struct('x0', 543, 'y0', 195, 'w', w_val, 'h', h_val);   % 021
    struct('x0', 651, 'y0', 195, 'w', w_val, 'h', h_val);   % 022
    struct('x0', 759, 'y0', 195, 'w', w_val, 'h', h_val);   % 023
    
    % === 第 4 行 (y0 = 274) ===
    struct('x0', 3,   'y0', 274, 'w', w_val, 'h', h_val);   % 024
    struct('x0', 111, 'y0', 274, 'w', w_val, 'h', h_val);   % 025
    struct('x0', 219, 'y0', 274, 'w', w_val, 'h', h_val);   % 026
    struct('x0', 327, 'y0', 274, 'w', w_val, 'h', h_val);   % 027
    struct('x0', 435, 'y0', 274, 'w', w_val, 'h', h_val);   % 028
    struct('x0', 543, 'y0', 274, 'w', w_val, 'h', h_val);   % 029
    struct('x0', 651, 'y0', 274, 'w', w_val, 'h', h_val);   % 030
    struct('x0', 759, 'y0', 274, 'w', w_val, 'h', h_val);   % 031
    
    % === 第 5 行 (y0 = 353) ===
    struct('x0', 3,   'y0', 353, 'w', w_val, 'h', h_val);   % 032
    struct('x0', 111, 'y0', 353, 'w', w_val, 'h', h_val);   % 033
    struct('x0', 219, 'y0', 353, 'w', w_val, 'h', h_val);   % 034
    struct('x0', 327, 'y0', 353, 'w', w_val, 'h', h_val);   % 035
    struct('x0', 435, 'y0', 353, 'w', w_val, 'h', h_val);   % 036
    struct('x0', 543, 'y0', 353, 'w', w_val, 'h', h_val);   % 037
    struct('x0', 651, 'y0', 353, 'w', w_val, 'h', h_val);   % 038
    struct('x0', 759, 'y0', 353, 'w', w_val, 'h', h_val);   % 039
    
    % === 第 6 行 (y0 = 432) ===
    struct('x0', 3,   'y0', 432, 'w', w_val, 'h', h_val);   % 040
    struct('x0', 111, 'y0', 432, 'w', w_val, 'h', h_val);   % 041
    struct('x0', 219, 'y0', 432, 'w', w_val, 'h', h_val);   % 042
    struct('x0', 327, 'y0', 432, 'w', w_val, 'h', h_val);   % 043
    struct('x0', 435, 'y0', 432, 'w', w_val, 'h', h_val);   % 044
    struct('x0', 543, 'y0', 432, 'w', w_val, 'h', h_val);   % 045
    struct('x0', 651, 'y0', 432, 'w', w_val, 'h', h_val);   % 046
    struct('x0', 759, 'y0', 432, 'w', w_val, 'h', h_val);   % 047
    
    % === 第 7 行 (y0 = 511，缺失最后两个图，共 6 个) ===
    struct('x0', 3,   'y0', 511, 'w', w_val, 'h', h_val);   % 048
    struct('x0', 111, 'y0', 511, 'w', w_val, 'h', h_val);   % 049
    struct('x0', 219, 'y0', 511, 'w', w_val, 'h', h_val);   % 050
    struct('x0', 327, 'y0', 511, 'w', w_val, 'h', h_val);   % 051
    struct('x0', 435, 'y0', 511, 'w', w_val, 'h', h_val);   % 052
    struct('x0', 543, 'y0', 511, 'w', w_val, 'h', h_val);   % 053
];

% ==============================================================
% 在原始拼图上显示 54 个裁剪框的位置
figure('Name', '裁剪区域预览', 'Position', [100, 100, 1000, 700]);
imshow(img);
hold on;

for i = 1:length(regions)
    r = regions(i);
    % 绘制矩形框
    rectangle('Position', [r.x0, r.y0, r.w, r.h], ...
              'EdgeColor', 'r', 'LineWidth', 1.0);
    % 在矩形左上角显示编号，稍微缩小字体防止重叠
    text(r.x0 + 2, r.y0 + 12, sprintf('%03d', i-1), ...
         'Color', 'r', 'FontSize', 8, 'FontWeight', 'bold', ...
         'BackgroundColor', 'white', 'EdgeColor', 'none', 'Margin', 1);
end
hold off;
title(sprintf('54 个裁剪区域示意 (尺寸统一为 %d x %d)', w_val, h_val));

%% 创建输出目录
if ~exist(output_dir, 'dir')
    [status, msg] = mkdir(output_dir);
    if ~status
        error('创建目录失败: %s', msg);
    end
end

% 循环裁剪并保存
fprintf('开始裁剪...\n');
for i = 1:length(regions)
    r = regions(i);
    
    % 检查边界
    if r.x0 < 1 || r.y0 < 1 || r.x0 + r.w - 1 > img_w || r.y0 + r.h - 1 > img_h
        warning('区域 %03d 超出边界，已跳过 (x0=%d, y0=%d, w=%d, h=%d)', ...
                i-1, r.x0, r.y0, r.w, r.h);
        continue;
    end
    
    % 裁剪
    cropped = img(r.y0 : r.y0 + r.h - 1, r.x0 : r.x0 + r.w - 1, :);
    
    % 保存为 PNG
    output_filename = fullfile(output_dir, sprintf('%03d.png', i-1));
    imwrite(cropped, output_filename);
end
fprintf('全部完成！54张图片（均 %dx%d）已保存在 "%s" 目录下。\n', w_val, h_val, output_dir);