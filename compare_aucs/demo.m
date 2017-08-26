% demo.m
%
%   OVERVIEW
%       Demo to showcase:
%           wilcoxonEXACT.m
%           idi.m
%           nri.m
%              
%       Creates a vector of labels and targets
%       First vector of targets is a bad estimate
%       Second vector of targets is a better estimate
%       Calculates AUC of each vector of estimated labels
%       Quantifies improvement via IDI and NRI
%   
%	AUTHORS
%       Erik Reinertsen <er@gatech.edu>
%

% Set vector length
n = 80;

% Generate random vector of labels
y = round(rand(n,1));

% Flip a few elements to lower the concordance between the estimated
% and real vector of labels
yhat1 = flip_percent(y, 3)
yhat2 = flip_percent(y, 5)

w = wilcoxonEXACT(yhat1, y);
fprintf('Wilcoxon statistic (yhat1): %1.3f\n', w);

w = wilcoxonEXACT(yhat2, y);
fprintf('Wilcoxon statistic (yhat2): %1.3f\n', w);

[nri_score, pval] = nri(yhat1, yhat2, y);
fprintf('Net reclassification improvement: %1.3f (p = %1.3f)\n', nri_score, pval);

[idi_score, pval] = idi(yhat1, yhat2, y);
fprintf('Integrated discrimination improvement: %1.3f (p = %1.3f)\n', idi_score, pval);

function y = flip_percent(y, pct)
    pct = round(pct);
    for i = 1:pct:length(y)
        if y(i) == 1
            y(i) = 0;
        else
            y(i) = 1;
        end
    end
end