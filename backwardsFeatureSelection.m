function [good_features] = backwardsFeatureSelection(cp_train, features_matrix, labels);

good_features = struct;

% Run reg LR using all features
[cp_train, cp_test, labels_train, labels_test, betas, idx_test] = ...
    reg_lr(features_matrix, labels, 'L1L2', 10, 'p');

% Current best training AUC is using all features
best_train_auc_original = median([cp_train.auc]);

for i_bfs = 1:10

    fprintf('Backwards feature selection: iteration #%1.0f\n', i_bfs);
    
    % Reset best_train_auc
    best_train_auc = best_train_auc_original;
    
    % Create temp features matrix just for b.f.s.
    x = features_matrix;

    % Create vector of indices for all features at random
    idx_features = randperm(size(x, 2));

    % Initialize index to number of features so we start at the end
    i = size(x, 2);

    % Loop through features from the end to the start
    % until you reach the first element
    while i > 0

%             fprintf('Index #%1.0f: removing index %1.0f\n', i, idx_features(i));

        % Isolate all features for this iteration except idx_features(i)
        i_idx_features = idx_features(idx_features ~= idx_features(i));
        x_except_i_feature = x(:, i_idx_features);

        % Run reg LR using `x_except_i_feature` feature set
        [cp_train, cp_test, labels_train, labels_test, betas, idx_test] = ...
            reg_lr(x_except_i_feature, labels, 'L1L2', 10, 'p');

        temp = best_train_auc;

        % If removing feature improved AUC
        if median([cp_train.auc]) > best_train_auc

            % Update `best_train_auc`
            best_train_auc = median([cp_train.auc]);

            % Remove the withheld feature
            idx_features(i) = [];
        end

%             fprintf('    Train AUC = %1.2f | Old AUC = %1.2f\n\n', ...
%                     median([cp_train.auc]), temp);

        % Decrement i
        i = i - 1;
    end

    good_features(i_bfs).idx = i_idx_features;
end