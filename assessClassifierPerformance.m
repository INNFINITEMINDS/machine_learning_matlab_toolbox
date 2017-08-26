function classPerf = assessClassifierPerformance(y, yHat)

% classPerf = assessClassifierPerformance(y, yHat)
% 
% Overview
%    Calculates different classifier performance metrics
%    given real labels 'y' and estimated labels 'yHat'
%     
% Input
%    y - array of real labels
%    yHat - array of estimated labels
%
% Output
%    classPerf      - structure with various fields
%    classPerf.auc  - area under receiver operating characteristic curve
%    classPerf.tpr  - true positive rate
%    classPerf.fpr  - false positive rate
%    classPerf.spec - specificity
%    classPerf.sens - sensitivity
%    classPerf.acc  - accuracy 
%    classPerf.ppv  - positive predictive value
%    classPerf.npv  - negative predictive value
%    classPerf.fdr  - false discovery rate
% 
% Copyright (C) 2017 Erik Reinertsen <er@gatech.edu>
% All rights reserved.
%
% This software may be modified and distributed under the terms
% of the BSD license.  See the LICENSE file in this repo for details.

% Set positive class values
posClass = 1;

% User perfcurve to compute AUC
[~, ~, ~, classPerf.auc, optimumRocPoint] = perfcurve(y, yHat, 1);

% Count number of true positives and negatives
numPos = sum(y == posClass);
numNeg = sum(y ~= posClass);

classPerf.fpr = optimumRocPoint(1);
classPerf.spec = 1 - classPerf.fpr;
fp = classPerf.fpr * numNeg;
tn = numNeg - fp;

classPerf.tpr = optimumRocPoint(2);
classPerf.sens = classPerf.tpr;
tp = classPerf.tpr * numPos;
fn = numPos - tp;

classPerf.acc = (tp + tn) / (numPos + numNeg);
classPerf.ppv = tp / (tp + fp);
classPerf.npv = tn / (tn + fn);
classPerf.fdr = fp / (fp + tp);

classPerf.f1 = (2 * classPerf.sens * classPerf.spec) ...
               / (classPerf.sens + classPerf.spec);

end