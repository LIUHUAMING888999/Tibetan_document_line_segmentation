function [  baseLineImage] = find_syllable_line(I)
%SYLLABLE_LINE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
% I: ���ֵĶ�ֵ��ͼ��0��ʾ������1��ʾǰ������������
   imgsize = size(I);
    
    % ʹ��ˮƽֱ�߼��������ֱ�߼�⡡ֻ�����ºڡ��ϰ׵�ֱ��
    h = zeros(3, 20);
    h(1,:) = -1;
    h(3,:) = 1;
    Y = filter2(h, I);
    [y,~] = size(Y);
    Y(y-2:y,:) = 0;
    ymax = max( abs(Y(:)));
    y = Y ./ ymax;
    y1 = abs(y .* (y< -0.5));
    y1(y1>0) = 1;

    % ������ͨ�������
    cc = bwconncomp(y1);
    graindata = regionprops(cc, {'Extrema','Centroid','Area','BoundingBox','PixelIdxList'});
    maxvalue = 99999;

    
    % �õ��������
    dm = GetDistanceMatrix(graindata, maxvalue);
    
    % ����׷��
    tc = GetTrackMatrix(dm, maxvalue, graindata, I);
    
    % ����׷�ٽ���õ�����
    [clp, cl, lineNum, dify, baseLineImage] = GetBaseLine(tc, graindata, y1);
%     RGB = label2rgb(baseLineImage);
%     figure, imshow(RGB);
end
 
function [clp, cl, lineNum, dify, baseLineImage] = GetBaseLine(tc, graindata, oriimg)
% ����׷�ٽ���õ����ϵ�������õĻ���
% ����ֵ clp һ��cell���� ������ÿ�����ߵ������λ��
%        cl ����x y�������ʽ������ÿ�����ߵ�λ��
%        lineNum ������Ŀ
%        avdify ������ÿ������֮��ľ���
%        baseLineImage ����ͼ

    % ��׷�ٽ������y����ֵ��������
    [lineNum, w] = size(tc);
    ct = [graindata.Centroid];
    oriy = zeros(1,lineNum);
    for i=1:lineNum
        idx = tc(i,1);
        oriy(i) = ct(2*idx); 
    end
    [~, idx] = sort(oriy, 'ascend');
    tc = tc(idx,:);
    
    ii = zeros( size(oriimg) );
    for i = 1:lineNum       
       for j = 1:w
           if tc(i,j) == 0
               break;
           end
           ii( graindata(tc(i,j)).PixelIdxList ) = i;
       end
    end
%     RGB = label2rgb(ii);
%     figure, imshow(RGB);
    
    % ��ÿһ��������������
    [cl, ~] = JoinAllBaseLine(ii, graindata, tc);
    clp = cell(lineNum,1);
    [h,w] = size(ii);
    baseLineImage = zeros(h,w);
    for j=1:lineNum
        clp{j} = sub2ind([h, w], int32(cl{j}(2,:)), int32(cl{j}(1,:)));
        baseLineImage(clp{j}) = j;
    end
    
    dify = zeros(lineNum, w);
    for i = 1:lineNum-1
        for j=1:w
           x1 = find( cl{i}(1,:) == j );
           y1 = cl{i}(2, x1(1));
           
           x2 = find( cl{i+1}(1,:) == j );
           y2 = cl{i+1}(2,x2(1));
           dify(i,j) = y2 - y1;
        end
    end
    dify(lineNum,:) = dify(lineNum-1, :);
    
end

function [cl, baseLineImage] = JoinAllBaseLine(ii, graindata, tc)
% �������еĻ��ߡ�����һ���в�ͬ��ǵĻ���ͼ
    bbx =  [graindata.Extrema]; 
    [h,w] = size(bbx);
    bbx = reshape(bbx,h,2,w/2);
    
    baseLineImage = ii;
    [lineNum, w] = size(tc);
    for i=1:lineNum
        for j=1:w-1
            if tc(i,j+1) == 0
                break;
            end
            
            leftOne = bbx(:,:,tc(i,j));
            rightOne = bbx(:,:,tc(i,j+1));
            
            dx = rightOne(8,1) - leftOne(3,1);
            dy = rightOne(8,2) - leftOne(3,2);
            dy = dy / dx;
            
            xstart = leftOne(3,1);
            ystart = leftOne(3,2);
            
            for x=0:dx-1
                baseLineImage(round(ystart+dy*x), round(xstart+x)) = i;
            end
        end
    end
    
    [h, w] = size(ii);
    resultImg = baseLineImage * 0;
    cl = cell(lineNum,1);
    for i=1:lineNum
        [i_y,i_x] = ind2sub([h, w], find(baseLineImage==i));
%         [i_x,idx] = sort(i_x);
%         i_y = i_y(idx);
        startx = min(i_x);
        endx = max(i_x);
        
        len = endx - startx + 1;
        newx = zeros(len,1);
        newy = zeros(len,1);
        
        for idx=1:len
            xvalue = startx + idx - 1;
            newx(idx) = xvalue;
            tmp = (i_x == xvalue);
            newy(idx) = mean(i_y(tmp));
        end
        newy = medfilt1(newy, 31);
        newy = medfilt1(newy, 31);
        resultImg(sub2ind([h, w], int32(newy), int32(newx))) = i;

%         cl{i} = [newx';newy'];
        
        resultImg( int32(newy(1)), 1:int32(newx(1))) = i;
        resultImg( int32(newy(len)), int32(newx(len)+1):w) = i;
    end
    
    for i=1:lineNum
        [i_y,i_x] = ind2sub([h, w], find(resultImg==i));
        cl{i} = [i_x';i_y'];
    end
    baseLineImage = resultImg;
end

function [tc] = GetTrackMatrix(dm, maxvalue, graindata, I)
% ׷�����е�ˮƽ��ͨ����
    [~, newIdx] = sort(dm, 2, 'ascend');

    % ׷��ÿһ����λ
    [h, w] = size(dm);
    tc = zeros(h); % ׷�ٶ���

    for curid1 = 1:h
        curid2 = curid1;
        tccount = 1;    % ��ǰ׷�ٶ��е����
        while curid2 <= h
            tc(curid1, tccount) = curid2;
            tccount = tccount + 1;

            % ȡ��һ����λ
            curid3 = curid2;
            bget = 0;
            while curid3 <= h
                if ( curid3 == newIdx(curid3,1) ) % ���������Ǳ��� ��ô�Ͳ�ȡ
                    break;
                end

                curid3 = newIdx(curid3, 1);
                if dm(curid2, curid3) == maxvalue
                    bget = 0;
                else
                    bget = 1;
                end
                break;
            end

            % �����ȡ����һ����λ
            if bget == 1
                dst = dm(curid2, curid3);
                if dst == maxvalue % �����һ����λ�޷�����Ҳ�˳�
                    break;
                end
                curid2 = curid3;
            else % ȡ�������˳�
                break;
            end
        end
    end

    ss = tc > 0;
    ssc = sum(ss, 2);
    [~, sidx] = sort(ssc, 'descend');
    newTc = tc(sidx,:);

    for i = 1:h-1       % ����Ŀ�ʼ�ж� �ϲ������Ӽ�
        baseL = newTc(i,:);
        if sum(baseL) == 0
            continue;
        end
        len = sum(baseL > 0);
        baseL = baseL( 1:len);    

        for j = i+1:h
            subL = newTc(j,:);
            if sum(subL) == 0
                continue;
            end
            len = sum(subL > 0);
            subL = subL( 1:len);        
            bsub = intersect(subL, baseL);
            if ~isempty(bsub)
                newTc(j,:) = 0;
            end
        end
    end

    
    pl = {graindata.PixelIdxList};
    pointNum = zeros(1,h);
    zk = zeros(1,h); % ��ռ��
    [hh, ww] = size(I);
    tmpImg = zeros([hh,ww]);
    for i=1:h
        baseL = newTc(i,:);
        if sum(baseL) == 0
            continue;
        end
        
        gminx = zeros(1,w);
        gmaxx = zeros(1,w);
        for j=1:w
           qyidx = newTc(i,j); % �����ͨ������
           if qyidx == 0
               continue;
           end
          
           [~, c_x] = ind2sub([hh, ww], pl{qyidx} );
           tmpImg( pl{qyidx} ) = i;
           minx = min(c_x);
           maxx = max(c_x);
           pointNum(i) = pointNum(i) + maxx - minx + 1;
           gminx(j) = minx;
           gmaxx(j) = maxx;
        end
        gminx = gminx(find ( gminx>0));
        len = max(gmaxx) - min(gminx) + 1;
        zk(i) = pointNum(i) / len;
    end
    
%     RGB = label2rgb( tmpImg);
%     figure,imshow(RGB);
    zk = zk';
    
    ss = newTc > 0;
    ssc = sum(ss,2);
    t1 = max(ssc) / 3;
    t2 = 0.3;
    
    idx1 = find( ssc > t1 );
    idx2 = find( zk > t2 );
    idx = intersect(idx1, idx2);    
    tc = newTc(idx, :);
end

function [dm] = GetDistanceMatrix(graindata, maxvalue)
% ����������ͨ����ľ������   

    bbx =  [graindata.BoundingBox];
    [h,w] = size(bbx);
    bbx = reshape(bbx, 4, w/4);
    bbx = bbx';
    [h,w] = size(bbx);

    dm = zeros(h);  % �������
    
    for i=1:h
        dx = bbx(:,1)' - (zeros(1,h) + bbx(i,1)+bbx(i,3));
        dy = (zeros(1,h) + (bbx(i,2)+bbx(i,4))/2) - ((bbx(:,2)+bbx(:,4)) / 2)';

        dy = abs(dy) < 5; % y�Ĳ���С��ĳһ��ֵ
        dy = ~dy;
        dx(dy) = maxvalue;
        dy = dx<0;  % �����ڵ�ǰ���ұ�
        dx(dy) = maxvalue;
        dx(i) = maxvalue;  % �����������
        dm(i,:) = dx;
    end
end

