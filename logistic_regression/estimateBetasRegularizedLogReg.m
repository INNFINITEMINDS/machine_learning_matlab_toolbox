function [lambda, beta] = estimateBetasRegularizedLogReg(features, labels, performanceMetric, regularizationMethod, numFolds)

%   [lambda, beta] = estimateBetasRegularizedLogReg(features, labels, performanceMetric, regularizationMethod, numFolds)
%
%	OVERVIEW
%       Estimate optimal lambdas and betas via regularized logistic regression
%       using parallelized loop over all folds
%
%   INPUT
%       features    - [n subjects x m features] matrix
%       labels      - [n subjects x 1] vector
%                     neg class: 0
%                     pos class: 1
%       performanceMetric       - 'auc': area under ROC curve (default)
%                               - 'acc': accuracy
%                           'vp': validation performance
%                           'auc': area under ROC curve
%       regularizationMethod    - 'L1L2': L1 and L2 elastic net (default)
%                               - 'L1': L1 norm (LASSO)
%                               - 'L2': L2 norm
%       numFolds        - number of folds for interal cross-validation
%
%   OUTPUT
%       lambda       - [n+1 subjects x 1] vector of optimized lambda values
%       beta         - [n+1 subjects x 1] vector of optimized beta coefficients
%
%	AUTHORS
%       Shamim Nemati       <shamim.nemati@gmail.com>
%       Erik Reinertsen     <er@gatech.edu>
%
%	REPO
%       n/a
%
%   DEPENDENCIES & LIBRARIES
%
%	COPYRIGHT (C) 2016 AUTHORS (see above)
%       This code (and all others in this repository) are under the GNU General Public License v3
%       See LICENSE in this repo for details.

% Set default parameters if not specified at function call
if nargin < 3, performanceMetric = 'auc'; end
if nargin < 4, regularizationMethod = 'L1L2'; end
if nargin < 5, numFolds = 10; end

% Ensure labels are col
labels = labels(:);

% Count # of distinct classes
num_classes = length(unique(labels));

% Count number of subjects (n) and features (m)
n = size(features, 1);
m = size(features, 2);

% minFunc settings
options.Display = 0; % reduce verbosity
options.MaxIter = 100;

% Insert a column of 1's into features
features = [ones(n, 1) features];

% Initialize other variables
opt_lambda = zeros(1, numFolds);
wvec0 = zeros((m+1) * (num_classes-1),1);

% Shift labels from (0 or 1) to (1 or 2) for `minFunc`
labels = labels + 1;

% Generate indices for crossfolds before running parpool
[features_cv, labels_cv] = generateCrossfolds(features, labels, numFolds);


for i_fold = 1:numFolds
    features_train = features_cv(i_fold).train;
    features_test  = features_cv(i_fold).test;
    labels_train   = labels_cv(i_fold).train;
    labels_test    = labels_cv(i_fold).test;
    
    funObj = @(w)SoftmaxLoss2(w, features_train, labels_train, num_classes);
    lam_ind = 0;
    lambda_vector = linspace(1e-3,5,10);
%    lambda_vector = [1e-3 5e-3 1e-2 5e-2 1e-1 0.5 1.0 2.0 3.0 4.0 5.0];
    val_perf = zeros(1,length(lambda_vector));
    
    for lambda = lambda_vector
        lam_ind = lam_ind + 1;
        lambda = lambda * ones(m+1, num_classes-1);
        lambda(1,:) = 0; % Don't penalize biases (intercept coefficient)
        
        % L2 performanceMetric
        if strcmpi(regularizationMethod,'L2')
            funObjL2 = @(w)penalizedL2(w,funObj,lambda(:));
            WL = minFunc(@penalizedL2, wvec0, options, funObjL2, lambda(:));
            
        % L1 & L2 regularization: Elastic Net
        elseif strcmpi(regularizationMethod,'L1L2')
            funObjL2 = @(w)penalizedL2(w,funObj,lambda(:));
            funObjL1L2 = @(w)pseudoGradL1(@(w)funObjL2(w), w, lambda(:));
            WL = minFunc(@(w)funObjL1L2(w), wvec0, options);
        
        % L1 regularization: Least absolute shrinkage and selection operator (LASSO)
        elseif strcmpi(regularizationMethod,'L1')
            funObjL1 = @(w)pseudoGradL1(funObj,w,lambda(:));
            WL = minFunc(@(w)funObjL1(w), wvec0, options);
        else
            WL = minFunc(funObj, wvec0, options);
        end
        
        WL = reshape(WL, [m+1 num_classes-1]);
        
        val_perf(lam_ind) = ...
            getValidationPerformance_Multi(WL, features_test(:, 2:end), labels_test, performanceMetric);
    end
    [~, ind_mx] = max(val_perf);
    opt_lambda(i_fold) = lambda_vector(ind_mx);
end

lambda = median(opt_lambda);
lambda = lambda*ones(m+1,num_classes-1); lambda(1,:) = 0; % Don't penalize biases

funObj = @(w)SoftmaxLoss2(w, features, labels, num_classes);

% L1 regularization: Least absolute shrinkage and selection operator (LASSO)
if strcmpi(regularizationMethod,'L1')
    funObjL1 = @(w)pseudoGradL1(funObj,w,lambda(:));
    beta = minFunc(@(w)funObjL1(w), wvec0, options);
end

% L1 & L2 regularization: Elastic Net
if strcmpi(regularizationMethod,'L1L2')
    funObjL2 = @(w)penalizedL2(w,funObj,lambda(:));
    funObjL1L2 = @(w)pseudoGradL1(@(w)funObjL2(w),w, lambda(:));
    beta = minFunc(@(w)funObjL1L2(w), wvec0, options);
end

% L2 regularization
if strcmpi(regularizationMethod,'L2')
    funObjL2 = @(w)penalizedL2(w,funObj,lambda(:));
    beta = minFunc(@penalizedL2, wvec0, options, funObjL2, lambda(:));
end

beta = reshape(beta,[m+1 num_classes-1]);

end % end function