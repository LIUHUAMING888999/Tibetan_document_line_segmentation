function [baseLinePoint,dis,index] = get_centroid_dis_from_upBaseLine(point,baseLine)
%������Ԫ��������λ�ã����䵽���л��ߵľ���
%   �˴���ʾ��ϸ˵��
 lineNums = max(baseLine(:));
%  [h,w] = size(baseLine);
  
 point = ceil(point);%ȡ����     
 y =  point(2);
 x =  point(1);
 ys = baseLine(:,x);
 [rows,~] = find(ys>0);
 baseLinePoint= abs(rows -y );
 [dis, index  ] = min(baseLinePoint);
end

