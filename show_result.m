function [z]=  show_result(img,bw,title_name)
% bw= imdilate(bw,strel('square',5));
%SHOW_RESULT 此处显示有关此函数的摘要 
   x = cat(3,bw*20,bw*59,bw*221); 
   y = cat(3,bw,bw,bw);
   
   z = img.*uint8(1-y) + uint8(x).* uint8(y);
   figure;imshow(uint8(z));title(title_name);
end

