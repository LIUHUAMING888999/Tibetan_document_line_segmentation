% clear all
% clc
% close all;
% % -----采集与读取图像
% [fn,pn,fi] = uigetfile('*.png','请选择所要识别的图片');          %1.读取图像
% %fn表示图片的名字，pn表示图片的路径，fi表示选择的文件类型
% bw = imread([pn fn]); % 读取图像 参数为图像名称和图像路径
% subplot(121), imshow(bw);  %显示图像函数                     %2.显示原始图像
% title('原始图像'); %显示原始图像
% %---------------------
% qingxiejiao = rando_get(bw)                          %3.调用函数，获取倾斜角
% bw1 = imrotate(bw,qingxiejiao,'bilinear','crop');        %4.图像进行位置矫正
% % bw2 = imrotate(bw,qingxiejiao,'bilinear','crop');        %4.图像进行位置矫正
% %取值为负值向右旋转 并选区双线性插值 并输出同样尺寸的图像
% subplot(122), imshow(bw1); % 显示修正后的图像
% title('倾斜校正');

%%
function [rotate_image, rot_angle ]=randon_skew_correct(I)

I3= I;
theta = 1:180;                              %4.theta就是要投影方向的角度
[R,xp] = radon(I3,theta);          %5.沿某个方向theta做radon变换，结果是向量
%所得R(p,alph)矩阵的每一个点为对I3基于（p,alph）的线积分,其每一个投影的方向对应一个列向量
[height, width ] = size(R);
% [r,c] = find(R>=max(max(R)));  %检索矩阵R中最大值所在位置，提取行列标
% max(R)找出每个角度对应的最大投影角度 然在对其取最大值，即为最大的倾斜角即90度
% 通过前10 个求平均值
[x, index ] = sort(R(:));
[row,angle ] = ind2sub([height,width],index);
c2 = mean(angle(end-50 :end));
% J=c;  %由于R的列标就是对应的投影角度
qingxiejiao=90-c2; %计算倾斜角
rotate_image = imrotate(I,qingxiejiao,'bilinear','crop');        %4.图像进行位置矫正

rot_angle=qingxiejiao;
% 通过在水平方向的投影，对图像进行微调
%   [rot_angle ] =  find_best_angle(  rotate_image);


end

function [rot_angle ] =  find_best_angle(  mask)
% countX:水平方向上的直方图
% mask: 二值化图像
% 算法思路： 在较小的角度上，旋转图像，获得最佳角度。
[height, width ] = size(mask);

min_value =10000;
rot_angle=0;
for k = -0.5:0.01:0.5
    rot_mask = imrotate(mask,k,'bilinear','crop');
    countX_R =  sum(rot_mask,2 );
    max_value = max(countX_R);
    [count_value, index]= find(countX_R );
    
    s= sum(1./count_value);
    %    z = find(countX_R);
    %    D = 1./z;
    %    s = sum(D)+ 0.0008*size(z,2);
    if(s<min_value)
        min_value = s;
        rot_angle = k;
    end
end
end

