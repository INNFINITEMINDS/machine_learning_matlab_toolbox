function [idx, features_cv, labels_cv] = generateBaggingFolds(features, labels, normalizeFlag, p)

% [idx, features_cv, labels_cv] = generateBaggingFolds(features, labels, validFold, p)

% Initialize default parameter values


% Default: 70:30 training:test ratio
if nargin < 4
    p = 0.70;
end

% Default: normalize data
if nargin < 3
    normalizeFlag = 1;
end

% Initialize other values
unique_labels = unique(labels)';
idx.train = [];
idx.test = [];

for k = unique_labels
    ind_kth_O = find(labels==k);
    r_ind = ind_kth_O(randperm(length(ind_kth_O)));
    idx.train = [idx.train       ; r_ind(1:floor(p * length(r_ind)))];
    idx.test = [idx.test         ; r_ind(floor(p * length(r_ind))+1:end) ];
end

idx.train = idx.train(randperm(length(idx.train)));
idx.test = idx.test(randperm(length(idx.test)));

if nargout > 1
    % Develop the training set
    features_cv.train = features(idx.train, :);
    labels_cv.train = labels(idx.train);

    % Develop the testing set
    features_cv.test = features(idx.test, :);
    labels_cv.test = labels(idx.test);
end

% Normalize data against training set
if normalizeFlag
    [features_cv.train, norm_info] = normalizeFeatures(features_cv.train);
    [features_cv.test] = normalizeFeatures(features_cv.test, norm_info);
end