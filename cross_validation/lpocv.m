function [class_perf] = lpocv(features, labels, reg_method)

% Determine indices of + & - labels
idx_pos = find(labels);
idx_neg = find(~labels);

% Count total # of combinations
n_combos = length(idx_pos)*length(idx_neg);

% Initialize arrays to store training and test labels
all_test_labels_true = [];
all_test_labels_est = [];

% Initialize iteration counter just to track algorithmic progress
ii = 1;

% Loop through each + label
for j = 1:5

    % Loop through each - label
	for k = 1:5

        fprintf('    LPOCV combo #%1.0f/%1.0f...\n', ii, n_combos);
        ii = ii + 1;
        
        % Determine training features
        features_cv.train = features([idx_pos(idx_pos ~= idx_pos(j)); ...
                                      idx_neg(idx_neg ~= idx_neg(k))], :); 

        % Determine training labels
        labels_cv.train = labels([idx_pos(idx_pos ~= idx_pos(j)); ...
                                  idx_neg(idx_neg ~= idx_neg(k))]); 
                              
        % Determine test features
        features_cv.test = features([idx_pos(j) idx_neg(k)], :);
        
        % Determine test labels
        labels_cv.test = labels([idx_pos(j) idx_neg(k)]);
        
        % Estimate beta coefficients from training set
        % via regularized logistic regression
        [lambdas, betas] = estimate_betas_reg_lr(features_cv.train, labels_cv.train, 'auc', reg_method);
        
        % Estimate labels corresponding to training features
        label_est_train = mnrval(betas, features_cv.train);
        
        % Estimate labels corresponding to test features
        label_est_test = mnrval(betas, features_cv.test);
        
        % Append arrays holding true and estimated test labels
        all_test_labels_true = [all_test_labels_true; labels_cv.test];
        all_test_labels_est = [all_test_labels_est; label_est_test(:, 2)];
    end
end

% Using all true and estimated labels, calculate AUC and optimal point
[x, y, threshold, auc, optrocpt] = ...
    perfcurve(all_test_labels_true, all_test_labels_est, 1);

% Assess other measures of classifer performance
class_perf = assess_classifier_perf(x, y, threshold, auc, ...
                    optrocpt, all_test_labels_true, all_test_labels_est);

% Add AUC to class_perf structure
class_perf.auc = auc;

fprintf('    LPCOV AUC: %1.3f\n', auc);

end % end function