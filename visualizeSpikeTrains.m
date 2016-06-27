clear all
close all
clc;

% Load data
load('firingTimes')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% DESCRIPTIVE ANALYSIS %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. Raster Plot

% from firing times
subplot(2,1,1)
plot(firings(:,1),firings(:,2),'.'); hold on;
plot(firings2(:,1),firings2(:,2),'r.');

% from spiking matrix
subplot(2,1,2)
[neurons, time] = find(spikes==1);
plot(time, neurons,'.'); hold on;

[neurons2, time2] = find(spikes2==1);
plot(time2, neurons2,'r.'); 

%% 2. Firing rate as spike count
mean(sum(spikes,2)./totalTime*1000) % spikes per sec
mean(sum(spikes2,2)./totalTime*1000) % spikes per sec

% if we find on average N spikes on a temporal window of time t, then
% the mean inter-spike interval (time between two spikes) is on
% average t/n

%% 3. Instantaneous firing rate 
integrationWindow = 10; %ms

psth  = [];
psth2 = [];
for inIDX = 1:integrationWindow:totalTime-2*integrationWindow
    iniIDX      = inIDX+integrationWindow;
    endIDX      = inIDX+2*integrationWindow;
%     disp(['ini ' num2str(iniIDX) ' end ' num2str(endIDX)]);
    psth    = [psth  sum(spikes(:,iniIDX:endIDX),2)./integrationWindow ];
    psth2   = [psth2 sum(spikes2(:,iniIDX:endIDX),2)./integrationWindow ];
end
% first samples
psth  = [ sum(spikes(:,iniIDX:endIDX),2)./integrationWindow  psth];
psth2 = [ sum(spikes2(:,iniIDX:endIDX),2)./integrationWindow psth2];

rate    = mean(spikes);
rate_v  = mean(v_mat);

subplot(3,1,1);
plot(rate_v, 'r');
title('mean firing rate from voltage');
subplot(3,1,2);
plot(rate); 
title('mean firing rate from spikes')
subplot(3,1,3);
h = plot(mean(psth)); hold on
plot(mean(psth2),'r');
title('instantaneous firing rate - PSTH')
set(gca, 'XTickLabel', 0:totalTime/10:totalTime)

%% 4. Correlation on voltage trace
% Correlation is disrupted by the non-linearity of the spike
[correlation,lags] = xcorr(v_mat(1,:), v_mat2(1,:),'coeff');
plot(lags, correlation)

max(correlation)
lags(correlation==max(correlation))

figure;
plot(v_mat(1,:)); hold on;
plot(v_mat2(1,:),'r'); 

%% 4. Correlation on spikes
% Not really high correlations on spikes either
allData     = [spikes ; spikes2]';
[RHO,PVAL]  = corr(allData);
imagesc(RHO)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% UNSUPERVISED LEARNING %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. K-means
numberNeur  = size(psth, 1);
allData     = [psth ; psth2];
k_cluster   = 2;
[idx,C]     = kmeans(allData,k_cluster);

% Number of neurons of each group classified correctly 
sum(idx(1:numberNeur)==1)./numberNeur*100
sum(idx(numberNeur+1:end)==2)./numberNeur*100

% Here we need crossvalidation + goodness of fit measures

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% SUPERVISED LEARNING %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. GLM - TO DO
addpath(genpath('glmnet_matlab'))
numberNeur      = size(spikes, 1);
allData         = [sum(spikes,2)  sum(spikes2,2)];
teacherSignal   = [zeros(numberNeur,1) ; ones(numberNeur,1)];
w_linReg        = glmfit(allData,teacherSignal,'binomial');
y_pred          = allData*w_linReg;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% OTHER CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inter-spike intervals
figure
NR_spikes       = length(firings);
Delta_times   = firings(2:NR_spikes)-firings(1:NR_spikes-1);
ISI_Grid        = 1/((max(Delta_times)-min(Delta_times))*totalTime/10000);
ISI_Grid_vec    = 0:ISI_Grid:max(Delta_times);

histogram_ISI  = hist(Delta_times,ISI_Grid_vec);
hist(Delta_times,ISI_Grid_vec)
title('ISI Distribution for  Neurons');
xlabel('dt(close msec)');
set(gca,'XLim',[0 max(Delta_times)]);


