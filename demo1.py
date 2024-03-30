#encoding:utf-8
close all; clear all;clc;
img = imread("001.18.png");
figure;imshow(img);title("image");

bw = img/255;
bw2=  1 - bw;
figure;imshow(bw2,[]);
hs = sum(bw2,2)
barh(hs);
hight = size(bw,1);
dis = 25;
figure;
hs2= hs*0;
for i = dis+1: 1:hight - dis
   hs2(i) = sum(hs(i-dis:i+dis))/( dis*2 + 1 );
    
end
figure;
barh(hs2);%ƽ�����ͼ��
[pks,locs] = findpeaks(hs2,'MinPeakDistance',100);
mx = max(hs2);
hs3  =  mx - hs2;
figure;barh(hs3);
hs3(1:locs(1))=0;
hs3(locs(8):hight) = 0 ;
figure;barh(hs3);

[pks2,locs2] = findpeaks(hs3,'MinPeakDistance',100);

x= 1;
for i =1: size(locs2)
   temp = bw*0;
   line = bw2(x:locs2(i),:);
   figure;imshow(line,[]);title("line");
   x = locs2(i)+1;
    
end




