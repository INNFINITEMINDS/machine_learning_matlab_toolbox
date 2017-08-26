function [nri_score, pval] = nri(pred_old, pred_new, labels)
%    [nri_score, pval] = nri(pred_old, pred_new, labels)
%
%    OVERVIEW
%       Calculates net reclassification improvement (NRI).
%       Compares classification of an old vs new classification technique
%
%    REFERENCES
%       Mayaud, Louis, et al. Dynamic Data During Hypotensive Episode Improves Mortality Predictions Among Patients With Sepsis and Hypotension. Critical Care Medicine. 2013.
%
%       Evaluating the added predictive ability of a new marker: From area under the ROC curve to reclassification and beyond
%       Michael J. Pencina Ralph B. D'agostino Sr Ralph B. D'Agostino Jr and Ramachandran S. Vasan
%
%    AUTHOR(S)
%       Louis Mayaud <louis.mayaud@gmail.com>


% By default, set positive labels as 1 and negative labels as 0
if nargin < 3
    labels = 1;
end

% Remove NaNs
Idx = (isnan(pred_old) & isnan(pred_new));
pred_old(Idx) = [];
pred_new(Idx) = [];

% Need to define the following probabilities
PEventUp = sum(labels==1 & pred_old==0 & pred_new==1)/sum(labels==1); 
PEventDown = sum(labels==1 & pred_old==1 & pred_new==0)/sum(labels==1); 
PNoneventUp= sum(labels==0 & pred_old==0 & pred_new==1)/sum(labels==0); 
PNoneventDown= sum(labels==0 & pred_old==1 & pred_new==0)/sum(labels==0); 

nri_score = (PEventUp - PEventDown) - (PNoneventUp - PNoneventDown) ;
z = abs(nri_score)/sqrt( (PEventUp + PEventDown)/sum(labels==1) + (PNoneventUp + PNoneventDown)/sum(labels==0)  );
[~,pval] = ztest(z,0,1);

end
