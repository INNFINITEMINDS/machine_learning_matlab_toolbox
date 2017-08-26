function p = compare_paired_models(oldPred, newPred, outcome, BstrapNbre, fcn)

% This function compare difference between old versus new model with
% boostrap estimate of the difference
% Usage: p = compare_paired_models(old, new, outcome, BstrapNbre, fcn)
% 
% By default: uses log likelihood if fcn is empty
%   Specify fcn = @metric(prediction, outcome)  otherwise
%   - BstrapNbre, is the number of bootstraps (defaut is 1000)
%
% (C) Louis Mayaud - University of Oxford - 2013
%     louis.mayaud@gmail.com
%
% Please cite:  ***
%

doPlot = false; % change if you want to plot histgram

if nargin < 3
    error('Nor enough input arguments')
elseif nargin <4
    BstrapNbre = 1e3;
elseif nargin < 5
    fcn = @loglikelihood;
end

N = size(outcome,1);
diff = nan(BstrapNbre,1);

% Bootstrap
for b=1:BstrapNbre
    
    bIdx = ceil(rand(N,1)*N);
    
    % Get performance old model
    oldPerf = fcn( oldPred(bIdx) , outcome(bIdx) );
    
    % Get performance new model
    newPerf = fcn( newPred(bIdx) , outcome(bIdx) );
    
    % Get difference
    diff(b) = newPerf-oldPerf;
    
end

if doPlot
    figure, hist(diff,100);
end

p = 1-sum(diff>0)/BstrapNbre;


end

function ll = loglikelihood(pred,outcome)

ll = sum(pred.*outcome + (1-pred).*(1-outcome));

end