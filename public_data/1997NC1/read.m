%% 批量裁剪：13 个自定义矩形区域，保存为 PNG
clear; clc; close all;

input_file = '1997nc1.jun25.p5us.p08hz.13panel.labelled.collage.10minutes.jpg';
output_dir = 'image';   % 保存目录
% 读取图像
if ~exist(input_file, 'file')
    error('文件 %s 不存在，请检查路径。', input_file);
end
img = imread(input_file);
[img_h, img_w, ~] = size(img);
fprintf('原始图像尺寸：宽=%d, 高=%d\n', img_w, img_h);

%%
% 定义 13 个裁剪区域（每个区域：x0, y0, width, height）
% 注意：坐标是 MATLAB 索引（从 1 开始），x0 为列（水平），y0 为行（垂直）
regions = [
    struct('x0', 37,   'y0', 160, 'w', 200, 'h', 200);   % 000
    struct('x0', 288,  'y0', 170, 'w', 200, 'h', 200);   % 001
    struct('x0', 538,  'y0', 170, 'w', 200, 'h', 200);   % 002
    struct('x0', 789,  'y0', 170, 'w', 200, 'h', 200);   % 003
    struct('x0', 1040, 'y0', 170, 'w', 200, 'h', 200);   % 004
    struct('x0', 1291, 'y0', 170, 'w', 200, 'h', 200);   % 005
    struct('x0', 1541, 'y0', 170, 'w', 200, 'h', 200);   % 006
    struct('x0', 37,   'y0', 580, 'w', 200, 'h', 200);   % 007
    struct('x0', 288,  'y0', 580, 'w', 200, 'h', 200);   % 008
    struct('x0', 539,  'y0', 590, 'w', 200, 'h', 200);   % 009
    struct('x0', 790,  'y0', 590, 'w', 200, 'h', 200);   % 010
    struct('x0', 1040, 'y0', 590, 'w', 200, 'h', 200);   % 011
    struct('x0', 1291, 'y0', 590, 'w', 200, 'h', 200);   % 012
];
% ==============================================================
% 在原始拼图上显示 13 个裁剪框的位置
figure;
imshow(img);  % 假设 img 已读入
hold on;

% 定义颜色（红色矩形框）
colors = lines(13);  % 区分不同区域（可选）

for i = 1:length(regions)
    r = regions(i);
    % 绘制矩形框（注意 rectangle 的坐标为 [x, y, width, height]）
    rectangle('Position', [r.x0, r.y0, r.w, r.h], ...
              'EdgeColor', 'r', 'LineWidth', 1.5);
    % 在矩形左上角显示编号
    text(r.x0 + 5, r.y0 + 15, sprintf('%03d', i-1), ...
         'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold', ...
         'BackgroundColor', 'white', 'EdgeColor', 'none');
end
hold off;
title('13 个裁剪区域示意（红色框，编号 000~012）');
%%
% 创建输出目录
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
[status, msg] = mkdir(output_dir);
if ~status
    error('创建目录失败: %s', msg);
end

% 循环裁剪
for i = 1:length(regions)
    r = regions(i);
    
    % 检查边界
    if r.x0 < 1 || r.y0 < 1 || r.x0 + r.w - 1 > img_w || r.y0 + r.h - 1 > img_h
        warning('区域 %d 超出图像边界，已跳过 (x0=%d, y0=%d, w=%d, h=%d)', ...
                i-1, r.x0, r.y0, r.w, r.h);
        continue;
    end
    
    % 裁剪
    cropped = img(r.y0 : r.y0 + r.h - 1, r.x0 : r.x0 + r.w - 1, :);
    
    % 保存为 PNG
    output_filename = fullfile(output_dir, sprintf('%03d.png', i-1));
    imwrite(cropped, output_filename);
    fprintf('已保存 %s  [%d, %d] %dx%d\n', output_filename, r.x0, r.y0, r.w, r.h);
end

fprintf('全部完成！裁剪图片保存在 %s 目录下。\n', output_dir);


%%
% img0 = imread('000.png');
% figure;
% imagesc(img0);
% colormap("gray")