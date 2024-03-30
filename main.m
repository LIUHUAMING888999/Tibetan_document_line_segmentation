clc; clear all; close all;
% addpath('tools');
imageDir = './non_borderImage\';
fileFolder=fullfile(imageDir);
dirOutput=dir(fullfile(fileFolder,'*.png'));
fileNames={dirOutput.name}';
image_nums = size(fileNames,1);
for num =  3: 3 %  image_nums
    imageName =  fileNames{num};
    imageNames = ['./non_borderImage/' imageName];
    maskName= ['./remove_border/' imageName];
    img = imread(imageNames);
    %     img0= img;
    %     img = medfilt3(img,[5,5,5]); % 中值滤波
    mask = imread(maskName);
    bw2 = 1-mask;
    
    hs = sum(bw2,2); %行投影
    % barh(hs);
    hight = size(bw2,1);
    dis = 25;
    figure;
    hs2= hs*0;
    for i = dis+1: 1:hight - dis
        hs2(i) = sum(hs(i-dis:i+dis))/( dis*2 + 1 ); %平滑
    end
    % figure;
    % barh(hs2);%平滑后的图像
    [pks,locs] = findpeaks(hs2,'MinPeakDistance',100);
    mx = max(hs2);
    hs3  =  mx - hs2;
    % figure;barh(hs3);
    hs3(1:locs(1))=0;
    hs3(locs(8):hight) = 0 ;  % 为了去除两端的干扰
    % figure;barh(hs3); % 可视化结果
    [pks2,locs2] = findpeaks(hs3,'MinPeakDistance',100); %求峰值
    x= 1;
    for i =1: size(locs2)
        temp = bw2*0;
        temp(x:locs2(i),:)  = bw2(x:locs2(i),:);
        %         figure;imshow(temp,[]);title("line");
        x = locs2(i)+1;
    end
    
    
    %***************-----1 step: 倾斜矫正---------------------------%
    % radon 拉东变换：
    % https://blog.csdn.net/corilei/article/details/80560608
    %**************************************************************%
    [height,width ] = size(mask);
    [pic, rot ]= randon_skew_correct(~mask); % 倾斜矫正
    img=imrotate(img,rot,'crop'); %旋转图像
    mask=imrotate(~mask,rot,'crop'); %旋转图像
    mask = ~mask;
    
    %*********************** 2 - Step:上元音提前处理 ********************%
    %  原因：上元音在藏文基线之上，在行切分的过程中，容易对行切分线，造成干扰；
    %  提前对这些元音进行所属行的划分：
    %  具体步骤：通过基线，对上元音进行切分，然后通过距离基线的长度，划分所属行。
    %
    %*******************************************************************%
    
    % 求基线
    [  baseLineImage] = find_syllable_line( mask);
    figure;imshow(baseLineImage);
    %可视化结果，与原图对应上
    show_result(img,baseLineImage,'显示基线的信息'); %可视化结果，测试你的数据是否可行，与目标是否已一致
    
    %统计基线之间的距离，以便求出行间的距离
    
    lineNums = max(baseLineImage(:));
    dis=zeros(lineNums,1);
    for k = 1:1: lineNums
        if k==1
            [rows,~] = find(baseLineImage==k);
        else
            [rows,~] = find(baseLineImage==k);
            dis(k) = mean(rows - uprows);
        end
        uprows=rows;
    end
    avg_Line_hight = ceil(  sum(dis)/(lineNums - 1) ); % 求所有行的平均值
    %求基线上方的元音，有可能有粘连的字符无法识别到；
    %也有可能，上方的断裂字符，认为是元音。
    %求基线上方的元音
    [line_UP_yuanyin , up_region_vowel] = get_vowel_region( baseLineImage ,mask);
    show_result(img,up_region_vowel,'up');
    
    % 基线下方要扩展一些区域
    [rows, cols ] = find(baseLineImage>0);
    baseLineImage2 = baseLineImage>1;
    for k=1:size(rows,1)
        baseLineImage2(rows(k):rows(k)+30,cols(k)) = 1;
        %获取基线下方的区域，为了去除一些干扰
    end
    %     show_result(img,baseLineImage2,'up');
    up_vowels = up_region_vowel - ( baseLineImage2>0 ); %分离出上元音
    %     figure;imshow(up_vowels);
    show_result(img,up_vowels,'up');
    
    % 对二值化（mask）图像进行区域属性求解
    L0 = bwlabel( ~mask-  ( baseLineImage2>0 ) );     % 得到分割行区域，做标签
    stats0 = regionprops(L0);
    
    % 上元音区域(up_vowels)的属性值
    L2 = bwlabel(up_vowels);
    stats2 = regionprops(L2);
    up_vowel_process = baseLineImage*0;
    for k =1 : size(stats2,1)
        if stats2(k).Area>2500  % || stats2(k).Area<50
            continue;
        end
        region = stats2(k);
        point=region.BoundingBox;
        point= ceil(point);
%         if point(3)<20|| point(4)<10
%             continue;
%         end
        [baseLinePoint,dis,index] = get_centroid_dis_from_upBaseLine(point,baseLineImage);
%         if dis>=50
%             continue;
%         end
        patchImg = up_vowels(point(2):point(2)+point(4)-1,point(1):point(1)+point(3)-1);
        %         zft= figure;imshow(patchImg);
        %         insert_word('文档1.doc',zft);
        filename=[ './up_vowel/' num2str(k) '_' 'area_' num2str(stats2(k).Area) '_h_' num2str(point(4)) ...
            '_w_'  num2str(point(3)) '_dis_' num2str(dis) '.png'];
        up_vowel_process(point(2):point(2)+point(4)-1,point(1):point(1)+point(3)-1) =index;
        
%                imwrite(patchImg,filename);
        %         pause(0.5);
        
%         close all;
    end
    figure;imshow(up_vowel_process.*up_vowels,[]);
    up_vowel_process= up_vowel_process.*up_vowels;
    mask2= ~ mask;
    mask2( up_vowel_process > 0) = 0;
    figure;imshow(mask2);
    show_result(img,mask2,'remove up vowel');
    
    
    
    
    
    
    
end









