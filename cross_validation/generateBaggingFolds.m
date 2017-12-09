function [idx, features_cv, labels_cv] = generateBaggingFolds(features, labels, normalizeFlag, p, numInstances)


% Default: bag five times
if nargin < 5
    numInstances = 5;
end

% Default: 70:30 training:test ratio
if nargin < 4
    p = 0.70;
end

% Default: normalize data
if nargin < 3
    normalizeFlag = 1;
end

for i = 1:numInstances

    features_cv(i).train = [];
    features_cv(i).test = [];
    labels_cv(i).train = [];
    labels_cv(i).test = [];
    
    % Initialize other values
    unique_labels = unique(labels)';
    idx.train = [];
    idx.test = [];

    for k = unique_labels
        ind_kth_O = find(labels==k);
        r_ind = ind_kth_O(randperm(length(ind_kth_O)));
        idx.train = [idx.train; r_ind(1:floor(p * length(r_ind)))];
        idx.test = [idx.test  ; r_ind(floor(p * length(r_ind))+1:end)];
    end
    
    idx.train = idx.train(randperm(length(idx.train)));
    idx.test = idx.test(randperm(length(idx.test)));

    if nargout > 1
        % Develop training set
        features_cv(i).train = features(idx.train, :);
        labels_cv(i).train = labels(idx.train);

        % Develop testing set
        features_cv(i).test = features(idx.test, :);
        labels_cv(i).test = labels(idx.test);
    end

    % Normalize data against training set
    if normalizeFlag
        [features_cv(i).train, norm_info] = normalizeData(features_cv(i).train);
        [features_cv(i).test] = normalizeData(features_cv(i).test, norm_info);
    end
    
end