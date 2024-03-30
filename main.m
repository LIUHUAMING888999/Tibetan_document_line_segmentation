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
    %     img = medfilt3(img,[5,5,5]); % ��ֵ�˲�
    mask = imread(maskName);
    bw2 = 1-mask;
    
    hs = sum(bw2,2); %��ͶӰ
    % barh(hs);
    hight = size(bw2,1);
    dis = 25;
    figure;
    hs2= hs*0;
    for i = dis+1: 1:hight - dis
        hs2(i) = sum(hs(i-dis:i+dis))/( dis*2 + 1 ); %ƽ��
    end
    % figure;
    % barh(hs2);%ƽ�����ͼ��
    [pks,locs] = findpeaks(hs2,'MinPeakDistance',100);
    mx = max(hs2);
    hs3  =  mx - hs2;
    % figure;barh(hs3);
    hs3(1:locs(1))=0;
    hs3(locs(8):hight) = 0 ;  % Ϊ��ȥ�����˵ĸ���
    % figure;barh(hs3); % ���ӻ����
    [pks2,locs2] = findpeaks(hs3,'MinPeakDistance',100); %���ֵ
    x= 1;
    for i =1: size(locs2)
        temp = bw2*0;
        temp(x:locs2(i),:)  = bw2(x:locs2(i),:);
        %         figure;imshow(temp,[]);title("line");
        x = locs2(i)+1;
    end
    
    
    %***************-----1 step: ��б����---------------------------%
    % radon �����任��
    % https://blog.csdn.net/corilei/article/details/80560608
    %**************************************************************%
    [height,width ] = size(mask);
    [pic, rot ]= randon_skew_correct(~mask); % ��б����
    img=imrotate(img,rot,'crop'); %��תͼ��
    mask=imrotate(~mask,rot,'crop'); %��תͼ��
    mask = ~mask;
    
    %*********************** 2 - Step:��Ԫ����ǰ���� ********************%
    %  ԭ����Ԫ���ڲ��Ļ���֮�ϣ������зֵĹ����У����׶����з��ߣ���ɸ��ţ�
    %  ��ǰ����ЩԪ�����������еĻ��֣�
    %  ���岽�裺ͨ�����ߣ�����Ԫ�������з֣�Ȼ��ͨ��������ߵĳ��ȣ����������С�
    %
    %*******************************************************************%
    
    % �����
    [  baseLineImage] = find_syllable_line( mask);
    figure;imshow(baseLineImage);
    %���ӻ��������ԭͼ��Ӧ��
    show_result(img,baseLineImage,'��ʾ���ߵ���Ϣ'); %���ӻ������������������Ƿ���У���Ŀ���Ƿ���һ��
    
    %ͳ�ƻ���֮��ľ��룬�Ա�����м�ľ���
    
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
    avg_Line_hight = ceil(  sum(dis)/(lineNums - 1) ); % �������е�ƽ��ֵ
    %������Ϸ���Ԫ�����п�����ճ�����ַ��޷�ʶ�𵽣�
    %Ҳ�п��ܣ��Ϸ��Ķ����ַ�����Ϊ��Ԫ����
    %������Ϸ���Ԫ��
    [line_UP_yuanyin , up_region_vowel] = get_vowel_region( baseLineImage ,mask);
    show_result(img,up_region_vowel,'up');
    
    % �����·�Ҫ��չһЩ����
    [rows, cols ] = find(baseLineImage>0);
    baseLineImage2 = baseLineImage>1;
    for k=1:size(rows,1)
        baseLineImage2(rows(k):rows(k)+30,cols(k)) = 1;
        %��ȡ�����·�������Ϊ��ȥ��һЩ����
    end
    %     show_result(img,baseLineImage2,'up');
    up_vowels = up_region_vowel - ( baseLineImage2>0 ); %�������Ԫ��
    %     figure;imshow(up_vowels);
    show_result(img,up_vowels,'up');
    
    % �Զ�ֵ����mask��ͼ����������������
    L0 = bwlabel( ~mask-  ( baseLineImage2>0 ) );     % �õ��ָ�����������ǩ
    stats0 = regionprops(L0);
    
    % ��Ԫ������(up_vowels)������ֵ
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
        %         insert_word('�ĵ�1.doc',zft);
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









