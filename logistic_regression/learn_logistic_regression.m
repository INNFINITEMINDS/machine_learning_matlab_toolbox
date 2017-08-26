
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Logistic Regression
% 
%flip_odds_ratio_array: array of 1 and -1, b will be multiplied by this
%array before generating odds ratio (1 if keep fv odds ratio as is, -1 if
%want to "flip" the odds ratio from its natural direction
%
% NOTE:  flip_odds_ratio_array is optional
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% replaces learn_multivariate_aki and learn_multivariate_LR
				     
function [auc Pvalues betas OddsRatio lowerOdds upperOdds HL_P y_est] = learn_logistic_regression(features_matrix, labels, flip_odds_ratio_array)

[betas, dev, stats] = glmfit(features_matrix, labels);
y_est = glmval(betas, features_matrix,'logit');
y = labels;
[X,Y,~, auc, OPTROCPT] = perfcurve(y, y_est, 1);
HL_P = HosmerLemeshowTest(y_est,y);

% b=b*-1; %7/1/2010

if (nargin<=2)
  flip_odds_ratio_array = [];
end

if (~isempty(flip_odds_ratio_array))
    betas(2:end) = betas(2:end) .* flip_odds_ratio_array';
end

OddsRatio = exp(betas);
Zvalues = betas./stats.se;
%Pvalues = cdf('Normal',-1*Zvalues,0,1)*2;
lowerBound = betas - 1.96*stats.se;
upperBound = betas + 1.96*stats.se;
lowerOdds = exp(lowerBound);
upperOdds = exp(upperBound);

% Report everything after the intercept
maxrow = size(OddsRatio,1);
betas = [betas(2:maxrow)];
OddsRatio = [OddsRatio(2:maxrow)];
upperOdds = [upperOdds(2:maxrow)];
lowerOdds = [lowerOdds(2:maxrow)];
Pvalues   = [stats.p(2:maxrow)];

%numDecimals = 6;
%num = 10^(numDecimals);
%OddsRatio = round(OddsRatio*num)/num;
%upperOdds = round(upperOdds*num)/num;
%lowerOdds = round(lowerOdds*num)/num;
return;

     
%%see http://www.mathworks.com/matlabcentral/newsreader/view_thread/256534
%for glmfit problems 
