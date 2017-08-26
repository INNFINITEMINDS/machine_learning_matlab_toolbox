function featuresCat = categorizeFeatures(features)

% featuresCat = categorize_features(features)
%
% Overview
%		Categorizes feature values to integer levels by z-scoring
%       and shifting to a positive range instead of centered at 0.
%       Used prior to mRMR feature selection.
%
% Input
%		features - matrix
%
% Output
%		featuresCat - categorized
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

featuresCat = NaN(size(features,1), size(features,2));
features = zscore(features);
featuresCat(features < -2) = 1;
featuresCat(features >= -2 & features < -1.5) = 2;
featuresCat(features >= -1.5 & features < -1.0) = 3;
featuresCat(features >= -1.0 & features < -0.5) = 4;
featuresCat(features >= -0.5 & features < 0) = 5;
featuresCat(features >= 0.0 & features < 0.5) = 6;
featuresCat(features >= 0.5 & features < 1.0) = 7;
featuresCat(features >= 1.0 & features < 1.5) = 8;
featuresCat(features >= 1.5 & features < 2.0) = 9;
featuresCat(features >= 2.0) = 10;

end