function [featuresCv, labelsCv] = generateCrossfolds(features, labels, numFolds)

% [outputVar] = function_name(inputVar)
% 
% Overview
%    Divide data into 'numFolds' crossfolds, balanced by class
%    
%     
% Input
%    inputVar
%
% Output
%
% Example
%
% Dependencies
%    https://github.com/tminka/lightspeed
%
% Reference(s)
% 
% Copyright (C) 2017 Erik Reinertsen <er@gatech.edu>
% All rights reserved.
%
% This software may be modified and distributed under the terms
% of the BSD license.  See the LICENSE file in this repo for details.

% Perform 5-fold CV by default
if nargin < 3
    numFolds = 5;
end

% Determine number of labels
numLabels = numel(labels);

% Isolate features for positive labels
featuresPos = features(logical(labels), :);
featuresNeg = features(logical(~labels), :);

% Initialize other values
uniqueLabels = unique(labels)';
mapFold = NaN(numLabels, 1);

% Loop through each unique label
for i = uniqueLabels
    
    % Isolate indices for i'th label
    idxiLabel = find(labels == i);
    
    % Generate crossfold indices for these labels
    idxCrossFolds = crossvalind('Kfold', length(idxiLabel), numFolds);
    
    % Map the i'th index to the appropriate fold
    mapFold(idxiLabel) = idxCrossFolds; 
end

% Loop through each crossfold
for k = 1:numFolds
    
	% Find indices for elements matching i'th fold (test set)
    idxTest = find(mapFold == k);
    
	% Isolate k'th fold test set
    featuresCv(k).test = features(idxTest, :);
    labelsCv(k).test = labels(idxTest);
    
    % Find indices for elements not matching i'th fold (training set)
    idxTrain = find(mapFold ~= k);
    
	% Isolate k'th fold training set
    featuresCv(k).train = features(idxTrain, :);
    labelsCv(k).train = labels(idxTrain);
end

end