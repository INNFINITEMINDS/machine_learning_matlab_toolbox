function [data, normInfo] = normalizeData(data, normInfo)

warning off


dimFeatures=size(data,2);

% Find data with Inf values
[rows, cols] = find(isinf(data));
colsWithInfs = unique(cols);

% Loop through each column of data with an Inf
for i = 1:numel(colsWithInfs)
    % Isolate i'th column with Inf
    thisCol = colsWithInfs(i);

    % Isolate data in that column
    thisFeature = data(:, thisCol);

    % Calculate the mean of non-Inf, non-Nan values
    meanVal = nanmean(thisFeature(~isinf(thisFeature))); 
    
    % Find indices among inf cols and rows
    ii = find(cols == thisCol);

    % Replace infs with meanVal
    data(rows(ii), cols(ii)) = meanVal;
end

if nargin<2
    normInfo.data_upper=zeros(1,dimFeatures);
    kstat=zeros(6,dimFeatures);
    normInfo.data_lower=zeros(1,dimFeatures);

    %% Find features that need normalization
    for k=1:dimFeatures

        aa=data(:,k);
        aa=aa(~isnan(aa));
        
        normInfo.isBinary(k) = numel(unique(aa))<=2;
        
        if normInfo.isBinary(k)
            continue;
        end

        normInfo.data_upper(k) = quantile(aa,0.995);

        if normInfo.data_upper(k)==0
            normInfo.data_upper(k) = quantile(aa,0.999);
        end

        aa(aa>normInfo.data_upper(k)) = normInfo.data_upper(k); % remove outliers
        normInfo.data_lower(k) = quantile(aa,0.005); 
        aa(aa<normInfo.data_lower(k)) = normInfo.data_lower(k); % remove outliers
        
        [~,~,kstat(1,k)] = lillietest(aa - normInfo.data_lower(k) + 1); %check for normal
        [~,~,kstat(2,k)] = lillietest(log(aa - normInfo.data_lower(k) + 1 )); %check for log normal
        [~,~,kstat(3,k)] = lillietest(sqrt(aa - normInfo.data_lower(k) + 1)); %check for square root normal
        
        [~,~,kstat(4,k)] = lillietest(normInfo.data_upper(k) - aa + 1); %check for normal
        [~,~,kstat(5,k)] = lillietest(log(normInfo.data_upper(k) - aa + 1 )); %check for log normal
        [~,~,kstat(6,k)] = lillietest(sqrt(normInfo.data_upper(k) - aa + 1)); %check for square root normal
    end

    [~, normInfo.trans_type] = min(kstat);
    normInfo.trans_type(normInfo.isBinary) = NaN;
    
    % Transform features that need normalization
    for k=1:dimFeatures

        if normInfo.isBinary(k)
            continue;
        end

        aa=data(:,k);
        aa(aa>normInfo.data_upper(k)) = normInfo.data_upper(k);
        aa(aa<normInfo.data_lower(k)) = normInfo.data_lower(k);
        data(:,k)=aa;
    end

    for v=find(normInfo.trans_type==1),
        data(:,v) = data(:,v) - normInfo.data_lower(v) + 1;
    end

    for v=find(normInfo.trans_type==2), data(:,v) = log( data(:,v) - normInfo.data_lower(v) + 1 ); end
    for v=find(normInfo.trans_type==3), data(:,v) = sqrt( data(:,v)-normInfo.data_lower(v) + 1 ); end
    for v=find(normInfo.trans_type==4), data(:,v) =  normInfo.data_upper(v) - data(:,v) + 1; end
    for v=find(normInfo.trans_type==5), data(:,v) = log( normInfo.data_upper(v) - data(:,v) + 1 ); end
    for v=find(normInfo.trans_type==6), data(:,v) = sqrt( normInfo.data_upper(v) - data(:,v) + 1 ); end
    
    % Replace NaN & Normalized data
    normInfo.data_mean = nanmean(data); normInfo.data_std = nanstd(data);
    for k=1:dimFeatures
        ind=find(isnan(data(:,k)));
        if normInfo.isBinary(k)
            data(:,k)=2*data(:,k)-1;
            data(ind,k)=0;
        else
            data(ind,k) = normInfo.data_mean(k);
            data(:,k) = (data(:,k) - normInfo.data_mean(k))./(1e-2+2.5*normInfo.data_std(k));
        end
    end
else
    % 1) remove outliers, transform, replace NaN by mean, subtract mean, divide by std
    for k=1:dimFeatures,  if normInfo.isBinary(k), continue;end
        aa=data(:,k); aa(aa>normInfo.data_upper(k)) = normInfo.data_upper(k); aa(aa<normInfo.data_lower(k)) = normInfo.data_lower(k);  data(:,k)=aa;
    end
    %
    for v=find(normInfo.trans_type==1),  data(:,v) =  data(:,v) - normInfo.data_lower(v) + 1 ; end
    for v=find(normInfo.trans_type==2),  data(:,v) = log( data(:,v)-normInfo.data_lower(v) + 1 ); end
    for v=find(normInfo.trans_type==3),  data(:,v) = sqrt( data(:,v)-normInfo.data_lower(v) + 1 ); end
    for v=find(normInfo.trans_type==4),  data(:,v) =  normInfo.data_upper(v) - data(:,v) + 1; end
    for v=find(normInfo.trans_type==5),  data(:,v) = log( normInfo.data_upper(v) - data(:,v) + 1 ); end
    for v=find(normInfo.trans_type==6),  data(:,v) = sqrt( normInfo.data_upper(v) - data(:,v) + 1 ); end
    %
    for k=1:dimFeatures
        ind=find(isnan(data(:,k)));
        if normInfo.isBinary(k), data(:,k)=2*data(:,k)-1; data(ind,k)=0;
        else
            data(ind,k)=normInfo.data_mean(k);
            data(:,k) = (data(:,k)-normInfo.data_mean(k))./(1e-2+2.5*normInfo.data_std(k));
        end
    end
end
warning on



