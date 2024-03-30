function [baseLinePoint,dis,index] = get_centroid_dis_from_upBaseLine(point,baseLine)
%根据上元音的重心位置，求其到各行基线的距离
%   此处显示详细说明
 lineNums = max(baseLine(:));
%  [h,w] = size(baseLine);
  
 point = ceil(point);%取整数     
 y =  point(2);
 x =  point(1);
 ys = baseLine(:,x);
 [rows,~] = find(ys>0);
 baseLinePoint= abs(rows -y );
 [dis, index  ] = min(baseLinePoint);
end

