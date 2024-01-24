
%%
clearvars, close all
load('headers.mat','headers'); 

%loading features extracted via FBCSP 
load('features_csp.mat','feature_vector'); %in case of 7 subj, w/o nr. 2 and 3
%load('features_csp_all.mat','feature_vector'); %in case of all 9 subj

%% parameters
nsessions=2;
ntrials=120; 
nsubjects=7; %9 in case of all subjects

labels=cell(9,2);
inputData=cell(1,1); %here we will store features in format compatible with model architecture
lab=[]; %here we will store labels in single vector

%% reorganize features + labels
index = 0;
for sub = [1, 4, 5, 6, 7, 8, 9] % use 1:9 in case of all subjects    
    dataname="BCI_2b_mat/B0"+sub+"T.mat";
    load(dataname); % load data from each subject

    for n=1:nsessions
        h=headers{sub,n}; % select subject's header for current session
        s=feature_vector{sub,n}; %select subject's feature vector for current session

        labels{sub,n}=h.Classlabel(1:ntrials); %1 = left, 2 = right
        lab=[lab; labels{sub,n}];

        for j=1:ntrials
            index=index+1;
            inputData{index,1}=s(j,:);
        end
    end   
end

%% create architecture
category={'1','2'}
lab=categorical(lab);

dimFeatures = 1; %our features are stored as 1D vectors
numHiddenUnits = 100; 
numClasses = 2;

% Set up the LSTM architecture
layers = [ ...
    sequenceInputLayer(dimFeatures)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    dropoutLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% Classification with leave one subject out (LOSO) + LSTM
idxtot=1:240*nsubjects;
Y_Pred_total = [];
Y_Real_total = [];
accuracy=[];

seed = 7;

for sub = 1:nsubjects
    rng(seed);

    % Split the data into training and testing sets
    idx=240*sub-239:240*sub; %indexes related to current subject
    not_idx=idxtot(setdiff(1:end,idx)); %indexes related to all other subjects

    XTrain=inputData(not_idx,1);
    YTrain=lab(not_idx);

    XTest=inputData(idx,1);
    YTest=lab(idx);
    
    %Define training options for LSTM
    options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize', 60, ...
    'InitialLearnRate', 0.002, ...
    'ValidationFrequency',10, ... 
    'GradientThreshold', 1, ...
    'plots','training-progress', ...
    'Verbose',1, ValidationData={XTest,YTest});

    % Train LSTM network
    net = trainNetwork(XTrain,YTrain,layers,options);
    % Evaluate the LSTM network on the test set
    YPred=classify(net,XTest);
    
    % Compute accuracy for each subject
    correct = nnz(YPred == YTest);
    accuracy(sub) = correct/numel(YTest);
    
    disp(['Accuracy for subject ', num2str(sub), ': ', num2str(accuracy(sub))]);
    Y_Pred_total=[Y_Pred_total; categorical(YPred)];
    Y_Real_total=[Y_Real_total; categorical(YTest)];
end

%% compute the validation metrics for all considered subjects
averageAccuracy = mean(accuracy);
disp(['Average accuracy: ', num2str(averageAccuracy)]);

[C_t_total, order_t] = confusionmat(Y_Real_total,Y_Pred_total, 'order', category);

TP=C_t_total(1,1);
FP=C_t_total(2,1);
FN=C_t_total(1,2);
TN=C_t_total(2,2);

accuracy_total = sum(diag(C_t_total))/sum(sum(C_t_total))
precision = TP/(TP+FP)
recall = TP/(TP+FN)

f1_measure_tmp_poly = 2 *(precision.*recall)./(precision+recall)

