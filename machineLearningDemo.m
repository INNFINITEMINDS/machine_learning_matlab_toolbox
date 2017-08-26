function machineLearningDemo()

% machineLearningDemo()
% 
% Overview
%    Showcases several useful machine learning toolbox functions
%     
% Input
%    n/a
%
% Output
%    n/a
%
% Dependencies
%    https://github.com/cliffordlab/machineLearning
%    https://github.com/tminka/lightspeed
%    https://github.com/erikrtn/dataviz
%
% Reference(s)
% 
% Copyright (C) 2017 Erik Reinertsen <er@gatech.edu>
% All rights reserved.
%
% This software may be modified and distributed under the terms
% of the BSD license.  See the LICENSE file in this repo for details.

ml.svm = true;
ml.lr = false;

% Create toy data
N = 200;
mCorr = 3;
m = 9;
p = 0.4;
effectSize = 1.25;
histFeatures = false;

fprintf('Generating toy data with %1.0f subjects and 4 features...\n', N);
[features, labels] = createToyData(N, mCorr, m, p, effectSize, histFeatures);

% Categorize features for mRMR feature selection
featuresCat = categorizeFeatures(features);

% Perform mRMR feature selection
idxFeaturesMrmr = mrmr_mid_d(featuresCat, labels, m)';

% Back up features
featuresOriginal = features;

% Try up to 'numFeatures' at a time
for numFeatures = 1:length(idxFeaturesMrmr)

    % Isolate up to 'numFeatures' top features
    features = featuresOriginal(:, idxFeaturesMrmr(1:numFeatures));
    
    % Divide data into training and test sets
    numFolds = 5;
    [featuresCv, labelsCv] = generateCrossfolds(features, labels, numFolds);

    % Initialize arrays to store results across all simulations
    testAucs = [];
    trainAucs = [];
    
    numSimulations = 10;

    for i = 1:numSimulations

        % Perform k-fold CV
        for k = 1:numFolds

            % Save labels to simpler variable names for consistency
            labelsTrain = labelsCv(k).train;
            labelsTest = labelsCv(k).test;

            % Normalize training set features
            [featuresTrain, normInfo] = normalizeFeatures(featuresCv(k).train);

            % Normalize test set features using data from training set
            featuresTest = normalizeFeatures(featuresCv(k).test, normInfo);

            % Train logistic regression
            if ml.lr
                % Estimate beta coefficients of a logistic regression
                b = glmfit(featuresTrain, labelsTrain, 'binomial', 'link', 'logit');

                % Estimate labels given beta coefficients and training set features
                labelsTrainEst = glmval(b, featuresTrain, 'logit');

                % Estimate labels given beta coefficients and test set features
                labelsTestEst = glmval(b, featuresTest, 'logit');
            end

            % Train SVM
            if ml.svm
                model = fitcsvm(featuresCv(k).train, labelsCv(k).train, ...
                                           'KernelFunction', 'linear', ...
                                           'Standardize', true, ...
                                           'ClassNames', {'1','0'});
                % Estimate training set classes
                [estLabel, score] = predict(model, featuresCv(k).train);

                % Isolate probability of belonging to positive class
                labelsTrainEst = score(:, 1);

                % Estimate test set classes
                [estLabel, score] = predict(model, featuresCv(k).test);

                % Isolate probability of belonging to positive class
                labelsTestEst = score(:, 1);
            end

            % Assess classifier performance on training set
            classPerfTrain(k) = assessClassifierPerformance(labelsTrain, labelsTrainEst);

            % Assess classifier performance on test set
            classPerfTest(k) = assessClassifierPerformance(labelsTest, labelsTestEst);

            % Print results
%             fprintf('Simulation #%d, cv #%d | Train AUC = %1.3f | Test AUC = %1.3f\n', ...
%                 i, k, classPerfTrain(k).auc, classPerfTest(k).auc);

            % Save AUCs to array
            trainAucs = [trainAucs; classPerfTrain(k).auc];
            testAucs = [testAucs; classPerfTest(k).auc];
        end
    end

    fprintf('Using %d top features (mRMR):\n', numFeatures);
    fprintf('Train AUC = %1.3f +- %1.3f\n', ...
            median(trainAucs), iqr(trainAucs));
    fprintf('Test AUC = %1.3f +- %1.3f\n\n', ...
            median(testAucs), iqr(testAucs));
        
    % Save results to a master array
    aucsAllTrain(numFeatures) = median(trainAucs);
    aucsAllTest(numFeatures) = median(testAucs);
end
    
end % end function