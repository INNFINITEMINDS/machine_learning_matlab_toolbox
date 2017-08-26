function [features, labels] = createToyData(N, mCorr, m, p, effectSize, plotFeatures)

% function [features, labels] = createToyData(N, p, effectSize, plotFeatures)
% 
% Overview
%   Creates a toy dataset for testing machine learning algorithms.
%     
% Input
%   N - number of subjects (rows)
%   m - number of features that correlate with label
%   p - proportion of label balance.
%       can be a fraction [0, 1], resulting in N*p positive labels,
%       or a number >1, resulting in p positive labels.
%   effectSize - correlation between features and labels;
%                for no correlation (AUC=0.50), set to 1
%                for strong correlation, set >>1 or <<1
%   plotFeatures - boolean flag; true if want to plot histograms of features 
%
% Output
%   features - [N x 4] matrix
%   labels - [N x 1] array
%
% Example
%   [features, labels] = createToyData(1e3, 0.25, 1.1, false);
%
% Dependencies
%    https://github.com/cliffordlab/
%
% Reference(s)
% 
% Copyright (C) 2017 Erik Reinertsen <er@gatech.edu>
% All rights reserved.
%
% This software may be modified and distributed under the terms
% of the BSD license. See the LICENSE file in this repo for details.

if nargin < 6
    plotFeatures = true;
end

if nargin < 5
    % Set difference factor (mean #2 = effectSize * mean #1)
    effectSize = 1.5;
end

if nargin < 4
    % Set label balance to 50%
    p = 0.5;
end

if nargin < 3
    m = 9;
end

if nargin < 2
    mCorr = 3;
end

if nargin < 1
    % Set sample size
    N = 200;
end

% Create labels
labels = zeros(N, 1);
if p < 1
    labels(randperm(N, round(N*p))) = 1;
else
    labels(randperm(N, p)) = 1;
end

% Find indices of pos and neg classes
idxPos = find(labels == 1);
idxNeg = find(labels == 0);

% Count # of positive and negative classes
n_pos = size(idxPos, 1);
n_neg = size(idxNeg, 1);

% Create [N subjects x 10 features] matrix drawn from
% normal distribution with mean mu and variance sigma
mu = 1;
sigma = 0.3;
features = normrnd(mu, sigma, N, m);

% Make five of the features correlate with the data
chosenFeatures = randsample(m, mCorr);

% Adjust the chose features
features(idxPos, chosenFeatures) = features(idxPos, chosenFeatures) * effectSize;

% Make some features colinear by choosing two at random
idxColinearFeature = randsample(m, 2);

% Find indices of all the remaining features
otherFeatures = setdiff(1:m, idxColinearFeature);

% Select one of those features
idxOtherFeature = randsample(otherFeatures, 2);

% Make that selected feature colinear with the previous
features(:, idxOtherFeature) = features(:, idxColinearFeature) * 1.3;

% Plot data
if plotFeatures
    figure('position', [100 100 1000 500]);
    for i = 1:m
        subplot(3, 3, i);
        nhist({features(idxPos, i), features(idxNeg, i)});
        title(sprintf('Feature %d', i));
        set(gcf,'color','w');
    end
    tightfig;
end

end