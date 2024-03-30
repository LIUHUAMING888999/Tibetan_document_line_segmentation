function [ line_UP_vowel,up_region_vowel ] = get_vowel_region( baseLineImage ,mask)
%
%baseLineImage : ����ͼ�����У�1��2��...,8�����е����ֱ�ʾ�еĻ���
% line_UP_vowel:����1����ʾ��Ԫ����������
%
[height,width] =  size(baseLineImage);
line_UP_vowel= zeros(height,width);
lineNums = max(baseLineImage(:));
for k =2:1:lineNums  % ���Ե�һ�У�����Ҫ����
    %mask,baseLineImage
    [up_rows,~]=find(baseLineImage==k-1);
    [bottom_rows,~] = find(baseLineImage==k);
    two_line_hight = bottom_rows - up_rows;%��һ�м�ȥ��һ��
    for t= 1:size(mask,2)
        line_UP_vowel(bottom_rows(t)- floor(two_line_hight(t)*0.3):bottom_rows(t),:)=1;
    end
    
end
figure;imshow(line_UP_vowel);
up_region_vowel = line_UP_vowel &~mask; %��ȡ��Ԫ���ַ�
%     show_result(img,line_UP_yuanyin &~mask,'up');
end