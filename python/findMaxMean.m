function [maxGain meanGain] = findMaxMean(rawImg,filtImg)
    thresh = 0.8;
    maxval = max(max(filtImg));
    limit = maxval*thresh;
    [ii,jj] = find(filtImg>limit);
    
    maxGain = max(max(rawImg(ii,jj)));
    meanGain = mean(mean(rawImg(ii,jj)));
end
