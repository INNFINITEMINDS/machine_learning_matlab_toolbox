function p = pauc_discrete(Wold,Nold,mortOld,Wnew,Nnew,mortNew)

% SE1
Q1old=Wold/(2-Wold);
Q2old=2*Wold^2/(1+Wold);
x = floor(Nold*mortOld);
y = floor(1-Nold*mortOld);
SEold = sqrt( (Wold*(1-Wold)+(x-1)*(Q1old-Wold^2) + (y-1)*(Q2old-Wold^2)) / (x*y) ) ;     

% SE2
Q1new=Wnew/(2-Wnew);
Q2new=2*Wnew^2/(1+Wnew);
x = floor(Nnew*mortNew);
y = floor(1-Nnew*mortNew);
SEnew = sqrt( (Wnew*(1-Wnew)+(x-1)*(Q1new-Wnew^2) + (y-1)*(Q2new-Wnew^2)) / (x*y) ) ;     

z = (Wnew-Wold)/sqrt( SEold^2 + SEnew^2   ) ;
p = Ztest(z);