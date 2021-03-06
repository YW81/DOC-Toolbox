%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Cross Validation Demo on EMotion Classification Task from the CK dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('DOC')
ck = load('./data/ck.mat');

tr = ck.data(1:100);
va = ck.data(101:120);
te = ck.data(121:end);

set.iter = 100;
set.parallel = 0;

L = [0 10.^[-4:4]];

% select the models 
model = {};
model{end+1} = 'HCRF';
model{end+1} = 'HCORF';
model{end+1} = 'VSL_CRF';


% run cross validation for each model
for i = 1:length(model)

    % use two sepparate regularization parameter if it is a variable state model
    % one for the ordinal and one for the nominal stes
    if ~isempty(findstr(model{i},'VSL'))
        [p,q] = meshgrid(L, L);
    else
        [p,q] = meshgrid(L, [0]);
    end
    L_set = [p(:) q(:)];


    % train a model for each regularization parameter 
    parfor j = 1:length(L_set)
        l1 = L_set(j,1);
        l2 = L_set(j,2);
        mod{j} = seq.tr(model{i},tr,[l1 l2],set);
        pre{j} = seq.pr(va,mod{j});
        res = packages.EVAL(va,pre{j});
        F1(j)  = res.Y.F1;
    end

    % find that regularization parameter that 
    % performs best on the validation set
    [~,j] = max(F1);

    % apply this setting on the training set
    pre{j} = seq.pr(te,mod{j});
    res = packages.EVAL(te,pre{j});
    F1 = res.Y.F1;

    % save results in a table
    table{i,1} = model{i};
    table{i,2} = F1;
    table{i,3} = L_set(j,1);
    table{i,4} = L_set(j,2);

end

% save result
cell2table(table,'VariableNames',{'model', 'F1','L1','L2'})



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%      model        F1       L1      L2 
%    _________    _______    ___    ____
%    'HCRF'       0.65974    1      0
%    'HCORF'      0.68968    10     0
%    'VSL_CRF'    0.80509    0      0.01
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
