function [selected_variables] = forwardFeatureSelection(feature_matrix,  outcome)

% feature_matrix should be NxM matrix where N is number of patients, M is
% number of features current feature_matrix
% selected_variables is a vector of index to columns of feature_matrix 

variables = [];
p_min = 0;
ALL_FV=feature_matrix;   
possibleVariables = 1:1:size(feature_matrix, 2); %caution: possibleVariables are 1 to size of possible Variables

j = 0;
while (p_min <= 0.05)  %j < 10
     p_vals = zeros(size(possibleVariables));       
     for i = 1:numel(possibleVariables)
         tempVars = [variables; possibleVariables(i)];
         X = ALL_FV(:,tempVars);
         %disp(ALL_FV_STR(tempVars));
         
         [b,dev,stats] = glmfit(X,outcome,'binomial');
         %variables
         %disp(['i is ' num2str(i) 'fv is ' ALL_FV_STR(possibleVariables(i))]);
         p_vals(i) = stats.p(numel(tempVars)+1);
     end
     p_min = min(p_vals);
    
     
     if (p_min <= 0.05)
         index = find(p_vals == p_min);
     
         variables = [variables; possibleVariables(index)'];
         possibleVariables(index)=[];

         %possibleVariables = removerows(possibleVariables,index);
     %else
     %    a=0;
     end
     j = j +1;
end

 selected_variables = variables;

 return;
