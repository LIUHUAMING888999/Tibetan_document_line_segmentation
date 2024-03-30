function [ line_UP_vowel,up_region_vowel ] = get_vowel_region( baseLineImage ,mask)
%
%baseLineImage : 基线图像，其中，1，2，...,8，其中的数字表示行的基线
% line_UP_vowel:其中1，表示上元音所在区域
%
[height,width] =  size(baseLineImage);
line_UP_vowel= zeros(height,width);
lineNums = max(baseLineImage(:));
for k =2:1:lineNums  % 忽略第一行，不需要处理
    %mask,baseLineImage
    [up_rows,~]=find(baseLineImage==k-1);
    [bottom_rows,~] = find(baseLineImage==k);
    two_line_hight = bottom_rows - up_rows;%下一行减去上一行
    for t= 1:size(mask,2)
        line_UP_vowel(bottom_rows(t)- floor(two_line_hight(t)*0.3):bottom_rows(t),:)=1;
    end
    
end
figure;imshow(line_UP_vowel);
up_region_vowel = line_UP_vowel &~mask; %获取上元音字符
%     show_result(img,line_UP_yuanyin &~mask,'up');
end