function [cp_train, cp_test, labels_train, labels_test, betas, idx_test] = ...
         reg_lr(features, labels, reg_method, crossfolds, label_flag)

%   class_perf_lr = reg_lr(features, labels, reg_method)
%   
%   Overview
%       Wrapper for regularized logistic regression
%       
%   Input
%
%       label_flag: 'label' returns binary label output, 1 for p(label = 1) > threshold
%                   'p' returns p(label) for labels output
%
%   Output
%       output_var:
%
%   Dependencies
%
%   Reference(s)
%   
%   Copyright (C) 2017 Erik Reinertsen <er@gatech.edu>
%   All rights reserved.
%   
%   This software may be modified and distributed under the terms
%   of the BSD license.  See the LICENSE file in this repo for details.


% Output discrete classes as labels by default
if nargin < 5; label_flag = 'label'; end

% Use 10 internal crossfolds by default
if nargin < 4; crossfolds = 10; end

% Initialize structs for output
labels_train.real = [];
labels_train.esti = [];
labels_test.real = [];
labels_test.esti = [];
labels_test.esti = [];
betas = [];
idx_test = [];

% Loop through each crossfold
for i_fold = 1:crossfolds
    
    % Generate training and test sets
    [idx, features_cv, labels_cv] = generate_crossfolds(features, labels, i_fold);
    
    % Estimate beta coefficients for logistic regression on training set
    [lambdas, betas_i] = estimateBetasRegularizedLogReg(features_cv.train, labels_cv.train, 'auc', reg_method);
    
    % Append growing matrix of beta coefficients
    betas = [betas, betas_i];
    
	% Estimate training set labels
    labels_train_est = mnrval(betas_i, features_cv.train);
    
    % Calculate AUC and optimal point for training set
    [x, y, threshold, auc, optrocpt] = ...
        perfcurve(labels_cv.train, labels_train_est(:,2), 1);
        
    % Assess other measures of classifer performance for training set
    % and save all as fields in struct
    cp_train(i_fold) = assess_classifier_perf(x, y, threshold, auc, ...
                    optrocpt, labels_cv.train, labels_train_est(:,2));

	if strcmp(label_flag, 'label')
        % Determine threshold from best classifier (pos are >=): training labels
        % and convert probabilities to discrete classes
        labels_train_rounded = zeros(length(labels_train_est(:,2)), 1);
        idx_pos = find(labels_train_est(:,2) >= threshold((x==optrocpt(1)) & (y==optrocpt(2))));
        labels_train_rounded(idx_pos) = 1;

        % Append output with training labels (real and estimated)
        labels_train.real = [labels_train.real; labels_cv.train];
        labels_train.esti = [labels_train.esti; labels_train_rounded];
    end
    
    if strcmp(label_flag, 'p')
        % Append output with training labels (real and estimated)
        labels_train.real = [labels_train.real; labels_cv.train];
        labels_train.esti = [labels_train.esti; labels_train_est(:,2)];
    end
                
    %% Estimate test set labels
    labels_test_est = mnrval(betas_i, features_cv.test);
    
    % Calculate AUC and optimal point for test set
    [x, y, threshold, auc, optrocpt, suby] = ...
        perfcurve(labels_cv.test, labels_test_est(:,2), 1);
    
    % Assess other measures of classifer performance for test set
    % and save all as fields in struct
    cp_test(i_fold) = assess_classifier_perf(x, y, threshold, auc, ...
                                             optrocpt, labels_cv.test, ...
                                             labels_test_est(:,2));
    if strcmp(label_flag, 'label')
        % Determine threshold from best classifier (pos are >=): training labels
        % and convert probabilities to discrete classes
        labels_test_rounded = zeros(length(labels_test_est(:,2)), 1);
        idx_pos = find(labels_test_est(:,2) >= threshold((x==optrocpt(1)) & (y==optrocpt(2))));
        labels_test_rounded(idx_pos) = 1;

        % Append output with training labels (real and estimated)
        labels_test.real = [labels_test.real; labels_cv.test];
        labels_test.esti = [labels_test.esti; labels_test_rounded];
    end
    
    if strcmp(label_flag, 'p')
        % Append output with training labels (real and estimated)
        labels_test.real = [labels_test.real; labels_cv.test];
        labels_test.esti = [labels_test.esti; labels_test_est(:,2)];
    end
    
    %% Append index object so we know which ID (as indices of `labels` input vector) are in test set
    idx_test = [idx_test; idx.test];
end

end % end function