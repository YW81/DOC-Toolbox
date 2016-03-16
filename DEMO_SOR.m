%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the data and split it into training set and a test set.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('DOC')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the data and split it into training and test sets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('DOC')
ck = load('./data/ck.mat'); % ck dataset [INPUT: PCA processed locations of 20 facial points / OUTPUT: 6 emotion classes (and their  temporal segments,i.e., neutral->onset->apex)]
tr_data = ck.data(1:100);  %use first 100 sequences for training/validation 
te_data = ck.data(101:end);  %use the rest of sequences for testing (all in person independent manner)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train a SOR model (one sequence label, and  multiple states per frame)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set.iter = 200;     % set the number of iteration for the gradient descent optimization
set.parallel = 0;   % apply method in parallel on all sequences
l1 = 1;                  % weight for l2 regularization

% Train one of the methods: 
% MLR:        Multinomial Logistic Regression
% SOR:       Static Ordinal Regression
% CRF:        Conditional Random Field 
% CORF:     Conditional Ordinal Random Field 

mod = seq.tr('SOR',tr_data,l1,set); % train a CRF model
pre = seq.pr(te_data,mod);           % predict labels using CRF model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% print prediction of sequence 4 class 1 (SOR has only class)
% pre{4}.H{1}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
res = packages.EVAL(te_data,pre)
res.H
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% [1] M. Kim and V. Pavlovic. "Structured output ordinal regression for dynamic facial emotion 
% intensity prediction". Computer Vision - ECCV 2010. Daniilidis, Kostas, Maragos, Petros, 
% Paragios and Nikos eds. 2010. pp. 649-662.
