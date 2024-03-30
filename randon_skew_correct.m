% clear all
% clc
% close all;
% % -----�ɼ����ȡͼ��
% [fn,pn,fi] = uigetfile('*.png','��ѡ����Ҫʶ���ͼƬ');          %1.��ȡͼ��
% %fn��ʾͼƬ�����֣�pn��ʾͼƬ��·����fi��ʾѡ����ļ�����
% bw = imread([pn fn]); % ��ȡͼ�� ����Ϊͼ�����ƺ�ͼ��·��
% subplot(121), imshow(bw);  %��ʾͼ����                     %2.��ʾԭʼͼ��
% title('ԭʼͼ��'); %��ʾԭʼͼ��
% %---------------------
% qingxiejiao = rando_get(bw)                          %3.���ú�������ȡ��б��
% bw1 = imrotate(bw,qingxiejiao,'bilinear','crop');        %4.ͼ�����λ�ý���
% % bw2 = imrotate(bw,qingxiejiao,'bilinear','crop');        %4.ͼ�����λ�ý���
% %ȡֵΪ��ֵ������ת ��ѡ��˫���Բ�ֵ �����ͬ���ߴ��ͼ��
% subplot(122), imshow(bw1); % ��ʾ�������ͼ��
% title('��бУ��');

%%
function [rotate_image, rot_angle ]=randon_skew_correct(I)

I3= I;
theta = 1:180;                              %4.theta����ҪͶӰ����ĽǶ�
[R,xp] = radon(I3,theta);          %5.��ĳ������theta��radon�任�����������
%����R(p,alph)�����ÿһ����Ϊ��I3���ڣ�p,alph�����߻���,��ÿһ��ͶӰ�ķ����Ӧһ��������
[height, width ] = size(R);
% [r,c] = find(R>=max(max(R)));  %��������R�����ֵ����λ�ã���ȡ���б�
% max(R)�ҳ�ÿ���Ƕȶ�Ӧ�����ͶӰ�Ƕ� Ȼ�ڶ���ȡ���ֵ����Ϊ������б�Ǽ�90��
% ͨ��ǰ10 ����ƽ��ֵ
[x, index ] = sort(R(:));
[row,angle ] = ind2sub([height,width],index);
c2 = mean(angle(end-50 :end));
% J=c;  %����R���б���Ƕ�Ӧ��ͶӰ�Ƕ�
qingxiejiao=90-c2; %������б��
rotate_image = imrotate(I,qingxiejiao,'bilinear','crop');        %4.ͼ�����λ�ý���

rot_angle=qingxiejiao;
% ͨ����ˮƽ�����ͶӰ����ͼ�����΢��
%   [rot_angle ] =  find_best_angle(  rotate_image);


end

function [rot_angle ] =  find_best_angle(  mask)
% countX:ˮƽ�����ϵ�ֱ��ͼ
% mask: ��ֵ��ͼ��
% �㷨˼·�� �ڽ�С�ĽǶ��ϣ���תͼ�񣬻����ѽǶȡ�
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

