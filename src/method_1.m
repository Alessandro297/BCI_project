
%%
clearvars, close all
%loading headers
load('headers.mat','headers');
%% parameters
mi_duration = 1000; % duration of motor imagery task
adapt_duration = 375; % 1.5 seconds after mi task
cue_dur= 313; % duration of cue presentation -> at beginning of each trial, before mi task
event_dur=mi_duration+adapt_duration+cue_dur;

nsessions=2;
nchannels = 3;  % excluding the eog channels, keep only eeg
freqs=[8:4:32]; %6 different freq ranges for csp filtering
nfreq=numel(freqs)-1;

feature_vector = cell(1, nsessions);
fs=250; %Hz sampling frequency from documentation
ntrials=120;
nsubjects=7; %excluding subjects 2 and 3 -> use 9 for all subjs 
labels=cell(9,2);
 
seed=7; %chosen seed to make results more stable
rng(seed); %set seed
%% preprocessing + segmentation + csp filtering
for sub = [1, 4, 5, 6, 7, 8, 9] %use 1:9 for all subjs
        
    dataname="BCI_2b_mat/B0"+sub+"T.mat";
    load(dataname); % load data from each subject
    
    % PREPROCESSING
    for n=1:nsessions
        h=headers{sub,n}; % select subject's header for current session
        s=data{1,n}.X; % select subject's signal for current session
        % channels order c4 cZ c3 ch01 ch02 ch03 (3 EEG 3 EOG)
        
        labels{sub,n}=h.Classlabel(1:ntrials); % 1 = left, 2 = right
        
        % select event type 'start of a new run'
        idx_runs=find(h.EVENT.TYP==32766);
        
        D=[]; 
        if (sub==1)&&(n==2) 
            start=1; % session 2 of subj 1 doesn't have the calibration run
        else 
            start=2; % all other sessions have it -> don't consider it in pipeline
        end
    
        for i = start:length(idx_runs)
            if(i < length(idx_runs))
                current_run=s(h.EVENT.POS(idx_runs(i)): h.EVENT.POS(idx_runs(i+1))-1,:)';
            else
                current_run=s(h.EVENT.POS(idx_runs(i)): end,:)';
            end
        
            raw_eeg = current_run(1:3, :); % consider first 3 channel = EEG ones
            D=[D, raw_eeg]; % store raw data of the runs w/o 100 NaN values
        end
        
        idx_trials=find(h.EVENT.TYP==768); % select event type 768 = start of a trial
        trials_pos=h.EVENT.POS(idx_trials); % select the position of the start of each trial
        
        for i = start : length (idx_runs)
            if (i<length(idx_runs))
                ind=find((trials_pos >= idx_runs(i)) & (trials_pos < idx_runs(i+1)));
            else
                ind=find((trials_pos >= idx_runs(i)));
            end
            % here consider the removal of 100 NaN that separate each run
            % rescale index of trial position
            trials_pos(ind) = trials_pos(ind)-100*(i-1)*ones(length(ind),1);
        end
        % rescale the position of trials after removing index of
        % calibration run
        trials_pos=trials_pos-ones(length(trials_pos),1)*(trials_pos(1)-1);
        
        % SEGMENTATION
        jseg=[];
        Lseg=cell(nfreq,1);
        Rseg=cell(nfreq,1);
        segms=cell(nfreq,ntrials);
        
        for j=1:ntrials
            % store the current segment
            if j < ntrials
                jseg=D(:,trials_pos(j):trials_pos(j)+event_dur-1);
            else
                jseg= D(:,trials_pos(end):trials_pos(end)+event_dur-1);
            end
            % NOISE REMOVAL
            fc=7; % cut off frequency
            wn=2.*fc./250; % normalized cutoff angular frequency (-> w/o pi)
            [B,A]=butter(6,wn,'high'); % high pass filter to remove ocular artifacts
            filseg(1,:)=filtfilt(B,A,jseg(1,:));
            filseg(2,:)=filtfilt(B,A,jseg(2,:));
            filseg(3,:)=filtfilt(B,A,jseg(3,:));

            %SIGNAL VISUALIZATION 
            %on choosen signals that have artifacts as stated in headers{1,2}
            if sub == 1 && n == 2 && (j == 8) % || j == 15 || j == 19 || j==28
               %import and plot raw data
               EEG_eeg= pop_importdata('dataformat', 'array', 'nbchan', 0, ...
               'data', 'jseg', 'srate', 250, 'pnts', 0, 'xmin', 0);
               EEG_eeg.setname = 'raw eeg';
               EEG_eeg = eeg_checkset(EEG_eeg);
               pop_eegplot(EEG_eeg, 1, 1, 1) 
               %import and plot filtered data
               EEG_eeg_cor= pop_importdata('dataformat', 'array', 'nbchan', 0, ...
               'data', 'filseg', 'srate', 250, 'pnts', 0, 'xmin', 0);
               EEG_eeg_cor.setname = 'corrected eeg';
               EEG_eeg_cor = eeg_checkset(EEG_eeg_cor);
               pop_eegplot(EEG_eeg_cor, 1, 1, 1) 
            end
    

            l=labels{sub, n};
            for f=1:nfreq
                % extraction of frequency bands for csp + normalization
                [H,G]=butter(4,[freqs(f),freqs(f+1)].*(2/fs), 'bandpass');
                for ch=1:3
                    band(ch,:)=filtfilt(H,G,filseg(ch,:)); 
                    band(ch,:)=zscore(band(ch,:));
                end
                
                if  l(j) == 1 % left segments
                    Lseg{f,1}= [Lseg{f,1}, band];
                else % right segments
                    Rseg{f,1}= [Rseg{f,1}, band];
                end             
                % Save the acquired information
                % store all the frequency bands
                segms{f,j} = band;
            end         
        end
        % CSP
        V=[];
        for f=1:nfreq
            W = csp(Lseg{f,1}, Rseg{f,1});
            vv=[];
            for tr=1:ntrials
                E=segms{f,tr}; % segment bandpassed of the current trial
                Wbar=W(:,[1,3]);
                % vertical concatenation of features of each trial in the current band
                vv = [vv; log(diag(Wbar'*E*E'*Wbar)./trace(Wbar'*E*E'*Wbar))'];
            end
            % store the features of each band in a different column of V
            V=[V, vv];
        end
     
        feature_vector{sub, n} = V;
    end   
end
%% Classification with leave one subject out (LOSO) + SVM
category={'1','2'}
Y_Pred_total = [];
Y_Real_total = [];
accuracy=[];
for nsub = [1, 4, 5, 6, 7, 8, 9] %use 1:0 for all subjs
    X_Train = [];
    Y_Train = [];
    % consider both sessions for each subject
    for sub = 1 : nsubjects
        if sub == nsub
            X_Test = [feature_vector{sub, 1}; feature_vector{sub,2}];
            Y_Test = [labels{sub, 1}; labels{sub, 2}];
        else
            X_Train = [X_Train ; feature_vector{sub, 1}; feature_vector{sub,2}];
            Y_Train = [Y_Train ; labels{sub, 1}; labels{sub, 2}];
        end
    end
    % Define Cubic Polynomial SVM 
    template = templateSVM(...
        'KernelFunction', 'polynomial', ...
        'PolynomialOrder', 3, ...
        'KernelScale', 'auto', ...
        'Standardize', true);
    mdl = fitcecoc(...
        X_Train, ...
        Y_Train, ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', category);
    YPred_test = predict(mdl, X_Test);
    YPred_test = categorical(YPred_test);
    YReal_test = categorical(Y_Test);
    
    % compute the accuracy for each subject
    [C_t_subj, order_t] = confusionmat(YReal_test, YPred_test, 'order', category);
    accuracy(nsub) = sum(diag(C_t_subj))/sum(sum(C_t_subj));
    %display obtained results
    disp(['Accuracy for subject ', num2str(nsub), ': ', num2str(accuracy(nsub))]);

    Y_Pred_total = [Y_Pred_total; YPred_test];
    Y_Real_total = [Y_Real_total; YReal_test];
 end
%compute + display validation metrics for all subjects
[C_t_total, order_t] = confusionmat(Y_Real_total, Y_Pred_total, 'order', category)
TP=C_t_total(1,1);
FP=C_t_total(2,1);
FN=C_t_total(1,2);
TN=C_t_total(2,2);
accuracy_total = sum(diag(C_t_total))/sum(sum(C_t_total))
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1_measure = 2 *(precision.*recall)./(precision+recall)
