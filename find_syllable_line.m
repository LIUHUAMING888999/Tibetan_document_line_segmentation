function [  baseLineImage] = find_syllable_line(I)
%SYLLABLE_LINE 此处显示有关此函数的摘要
%   此处显示详细说明
% I: 文字的二值化图像，0表示背景，1表示前景（文字区域）
   imgsize = size(I);
    
    % 使用水平直线检测器进行直线检测　只保留下黑　上白的直线
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

    % 进行连通区域分析
    cc = bwconncomp(y1);
    graindata = regionprops(cc, {'Extrema','Centroid','Area','BoundingBox','PixelIdxList'});
    maxvalue = 99999;

    
    % 得到距离矩阵
    dm = GetDistanceMatrix(graindata, maxvalue);
    
    % 进行追踪
    tc = GetTrackMatrix(dm, maxvalue, graindata, I);
    
    % 根据追踪结果得到基线
    [clp, cl, lineNum, dify, baseLineImage] = GetBaseLine(tc, graindata, y1);
%     RGB = label2rgb(baseLineImage);
%     figure, imshow(RGB);
end
 
function [clp, cl, lineNum, dify, baseLineImage] = GetBaseLine(tc, graindata, oriimg)
% 根据追踪结果得到从上到下排序好的基线
% 返回值 clp 一个cell数组 保存了每条基线的坐标点位置
%        cl 按照x y坐标的形式保存了每条基线的位置
%        lineNum 基线数目
%        avdify 保存了每条基线之间的距离
%        baseLineImage 基线图

    % 对追踪结果根据y坐标值进行排序
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
    
    % 把每一条基线连接起来
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
% 连接所有的基线　返回一个有不同标记的基线图
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
% 追踪所有的水平连通区域
    [~, newIdx] = sort(dm, 2, 'ascend');

    % 追踪每一个单位
    [h, w] = size(dm);
    tc = zeros(h); % 追踪队列

    for curid1 = 1:h
        curid2 = curid1;
        tccount = 1;    % 当前追踪队列的序号
        while curid2 <= h
            tc(curid1, tccount) = curid2;
            tccount = tccount + 1;

            % 取下一个单位
            curid3 = curid2;
            bget = 0;
            while curid3 <= h
                if ( curid3 == newIdx(curid3,1) ) % 如果最近的是本身 那么就不取
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

            % 如果能取到下一个单位
            if bget == 1
                dst = dm(curid2, curid3);
                if dst == maxvalue % 如果下一个单位无法到达也退出
                    break;
                end
                curid2 = curid3;
            else % 取不到就退出
                break;
            end
        end
    end

    ss = tc > 0;
    ssc = sum(ss, 2);
    [~, sidx] = sort(ssc, 'descend');
    newTc = tc(sidx,:);

    for i = 1:h-1       % 从最长的开始判断 合并所有子集
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
    zk = zeros(1,h); % 空占比
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
           qyidx = newTc(i,j); % 获得连通区域编号
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
% 计算所有连通区域的距离矩阵   

    bbx =  [graindata.BoundingBox];
    [h,w] = size(bbx);
    bbx = reshape(bbx, 4, w/4);
    bbx = bbx';
    [h,w] = size(bbx);

    dm = zeros(h);  % 距离矩阵
    
    for i=1:h
        dx = bbx(:,1)' - (zeros(1,h) + bbx(i,1)+bbx(i,3));
        dy = (zeros(1,h) + (bbx(i,2)+bbx(i,4))/2) - ((bbx(:,2)+bbx(:,4)) / 2)';

        dy = abs(dy) < 5; % y的差异小于某一阈值
        dy = ~dy;
        dx(dy) = maxvalue;
        dy = dx<0;  % 必须在当前点右边
        dx(dy) = maxvalue;
        dx(i) = maxvalue;  % 不计算自身点
        dm(i,:) = dx;
    end
end

