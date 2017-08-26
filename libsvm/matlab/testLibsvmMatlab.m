function testLibsvmMatlab()

% Tests if LibSVM is installed and works

% Reset environment
clear all; close all; clc;

% Generate toy data
N = 200;
mCorr = 3;
m = 10;
p = 0.50;
effectSize = 1.25;
plotFeatures = false;
[features, labels] = createToyData(N, mCorr, m, p, effectSize, plotFeatures);

% Convert negative labels to -1 for libsvm
labels(labels == 0) = -1;

% Generate crossfolds
numFolds = 5;
[featuresCv, labelsCv] = generateCrossfolds(features, labels, numFolds);

% Initialize arrays to store labels
yTrueTrainAll = [];
yTrueTestAll = [];
yHatTrainAll = [];
yHatTestAll = [];

% Perform 5-fold CV
for k = 1:numFolds
    % Train SVM using libsvm
    model = libsvmtrain(labelsCv(k).train, featuresCv(k).train, '-b 1');
    
    % Evaluate SVM model on training data
    [yHatTrainLabels, acc, probs] = ...
        libsvmpredict(labelsCv(k).train, featuresCv(k).train, model, '-b 1');
    
    % Isolate P(positive label)
    yHatTrain = probs(:, 1);
    
    % Evaluate SVM model on test data
    [yHatTestLabels, acc, probs] = ...
        libsvmpredict(labelsCv(k).test, featuresCv(k).test, model, '-b 1');
    
    % Isolate P(positive label)
    yHatTest = probs(:, 1);
    
    % Append arrays
    yTrueTrainAll = [yTrueTrainAll; labelsCv(k).train];
    yTrueTestAll = [yTrueTestAll; labelsCv(k).test];
    yHatTrainAll = [yHatTrainAll; yHatTrain];
    yHatTestAll = [yHatTestAll; yHatTest];
end


% Assess libsvm performance
[x, y, t, trainAuc, optrocpt] = perfcurve(yTrueTrainAll, yHatTrainAll, 1);
[x, y, t, testAuc, optrocpt] = perfcurve(yTrueTestAll, yHatTestAll, 1);

fprintf('\nTrain AUC = %1.3f | Test AUC = %1.3f\n', trainAuc, testAuc);

end