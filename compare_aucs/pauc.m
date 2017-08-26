function [pvalue Wold Wnew]= pauc(predOld, predNew, outcome)

%	OVERVIEW
%       Compares two AUCs derived from same cases.
%       Instead, author recommands the use of:
%           - NetReclassificationImprovement.m
%           - IntegratedDiscriminationImprovement.m
% 
%   USAGE
%       [pvalue Wold Wnew] = pauc(pred_old, pred_new, outcome)
% 
%   DEPENDENCIES
%       https://github.com/cliffordlab/compare_aucs
%
%	REFERENCE
%   	Mayaud, Louis, et al. "Dynamic Data During Hypotensive Episode Improves
%       Mortality Predictions Among Patients With Sepsis and Hypotension."
%       Critical care medicine 41.4 (2013): 954-962.
%
%	COPYRIGHT
%       Louis Mayaud, 2011 (louis.mayaud@gmail.com)

[Wold SEold] = wilcoxon_statistic(predOld,outcome);
[Wnew SEnew] = wilcoxon_statistic(predNew,outcome);

Revent = abs(corr(predOld(outcome==1),predNew(outcome==1),'type','Kendall'));
RNevent = abs(corr(predOld(outcome==0),predNew(outcome==0),'type','Kendall'));

Ravg = (Revent + RNevent)/2 ;
Wavg = (Wold + Wnew)/2;

load Rcoeff-AUC-sign.mat
AUCs = 0.7:0.025:0.975;
Rs = 0.02:0.02:0.9;

[m RIdx] = min(sqrt((Rs-Ravg).^2));
[m WIdx] = min(sqrt((AUCs-Wavg).^2));
R = correlationAUC(RIdx,WIdx); % Get the value from the table 
    % See Table I in Hanley et Mc Neil, A method of Comparing the AUROC derived from same cases
    % Radiology vol. 148, N. 3, pp839-843, Sept. 1983
    
    if SEold^2 + SEnew^2 - 2*R*SEold*SEnew <0
        pvalue = NaN;
        return
    end
    
z = abs(Wnew-Wold)/sqrt( SEold^2 + SEnew^2 + 2*R*SEold*SEnew ) ;
pvalue = Ztest(z);

end  % end function
