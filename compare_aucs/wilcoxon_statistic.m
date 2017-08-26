function [W SE] = wilcoxon_statistic(pred,outcome)
% from Hanley et McNeil Apr. 1982
% Radiology 143, The meaning and use of the area under the ROC curve
% Returns Wilcoxon Stat (equivelent to AUC) and its standard error

X = pred(outcome==1);
Y = pred(outcome==0);

x=length(X);
y = length(Y);


Xa = pred(outcome==1);
Xn = pred(outcome==0);
w=0;
q1=0; q2 = 0;
for i=1:length(Xa)
    for j=1:length(Xn)
        if Xa(i) > Xn(j)
            w = w+1;
        elseif Xa(i)==Xn(j)
            w = w + 1/2;
        end
        
    end
end
W = w/(length(Xa)*length(Xn)) ; 
q1=W/(2-W);
q2=2*W^2/(1+W);
SE = sqrt( (W*(1-W)+(x-1)*(q1-W^2) + (y-1)*(q2-W^2)) / (x*y) ) ;            
