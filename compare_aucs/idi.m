function [idi_score, pval] = idi(pred_old, pred_new, labels)

%   [idi_score, pval] = idi(pred_old, pred_new, labels)
%
%   OVERVIEW
%       Calculates integrated discrimination improvement (IDI) to compare old vs new classification technique 
%
%    REFERENCES
%       Mayaud, Louis, et al. Dynamic Data During Hypotensive Episode Improves Mortality Predictions Among Patients With Sepsis and Hypotension. Critical Care Medicine. 2013.
%
%       Michael J. Pencina Ralph B. D'agostino Sr Ralph B. D'Agostino Jr and Ramachandran S. Vasan. Evaluating the added predictive ability of a new marker: From area under the ROC curve to reclassification and beyond. Statistics in Medicine. 2008.
%
%    AUTHOR(S)
%       Louis Mayaud <louis.mayaud@gmail.com>

% Remove NaNs
Idx = (isnan(pred_old) & isnan(pred_new));
pred_old(Idx) = [];
pred_new(Idx) = [];

P_new_ev = mean(pred_new(labels==1));
P_new_Nev = mean(pred_new(labels==0));
P_old_ev = mean(pred_old(labels==1));
P_old_Nev = mean(pred_old(labels==0));
idi_score = (P_new_ev - P_new_Nev) - ( P_old_ev - P_old_Nev ) ;

SEevent = std( pred_old(labels==1)-pred_new(labels==1) ) / sqrt(sum(labels==1)) ; 
SENevent = std( pred_old(labels==0)-pred_new(labels==0) ) / sqrt(sum(labels==0)) ; 
z = abs(idi_score)/ sqrt( SEevent.^2 +  SENevent.^2 ) ;
[~,pval] = ztest(z,0,1);

end % end idi.m

function [Sen, Spe] = SenSpe(pred,out)

    [pred , idx] = sort(pred,'ascend');
    out = out(idx);
    for i=1:length(pred)
        pred_bin = pred > pred(i);
        tp = sum(pred_bin == 1 & out == 1 ); 
        tn = sum(pred_bin == 0 & out == 0 );
        fp = sum(pred_bin == 1 & out == 0 );
        fn = sum(pred_bin == 0 & out == 1 );
        Sen(i) = tp/(tp+fn);
        Spe(i) = tn/(tn+fp);
    end
    %Sen = [0 Sen 1];
    %Spe = [1 Spe 0];

end
