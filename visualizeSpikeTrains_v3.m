clear all
close all
clc;

% Load data
load('firingTimes')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% DESCRIPTIVE ANALYSIS %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. Raster Plot
% Q: How does the data look like in this trial?
[neurons, time] = find(allSpikes==1);
plot(time, neurons,'.'); hold on;
 


%% 2. Firing rate as spike count
% Q: Is there a difference in the average firing rate?
sp_per_sec = sum(allSpikes,2)./totalTime*1000; % spikes per sec
figure
hist(sp_per_sec);


% if we find on average N spikes on a temporal window of time t, then
% the mean inter-spike interval (time between two spikes) is on
% average t/n

%% 3. Instantaneous firing rate 
% Q: can we gain temporal resolution in the firing rate estimate?
integrationWindow = 200; %ms

psth  = [];
psth2 = [];
for inIDX = 1:integrationWindow:totalTime-2*integrationWindow
    iniIDX      = inIDX+integrationWindow;
    endIDX      = inIDX+2*integrationWindow;
    psth    = [psth  sum(allSpikes(:,iniIDX:endIDX),2)./integrationWindow ];
   
end
% lazy solution: complete the rate in the first and last integration
% window
psth    = [ sum(allSpikes(:,1:integrationWindow),2)./integrationWindow  psth]; 
psth    = [ psth    sum(allSpikes(:,endIDX:end),2)./integrationWindow  ]; 

% compare to a simple approach of averaging
rate    = mean(allSpikes);
rate_v  = mean(v_mat);

figure;
subplot(3,1,1);
plot(rate_v, 'r');
title('mean firing rate from voltage');
subplot(3,1,2);
plot(rate); 
title('mean firing rate from spikes')
xlim([0 10000]);
subplot(3,1,3);
h = plot(mean(psth)); hold on
plot(mean(psth),'r');
title('instantaneous firing rate - PSTH')
set(gca, 'XTickLabel', 0:totalTime/10:totalTime)
ylabel('Firing probability')

% Resolution of our data is reduced by integrationWindow
% there seem to be an impact in the firing rate

% % %% 4 Linear correlation on psth
% % % Are neurons that respond to stimuli correlated to each other
% % % Not really high correlations on spikes either
% % % Pearson's correlation
% % [RHO,PVAL]  = corr(psth');
% % figure;
% % imagesc(RHO);
% % colorbar;
% % caxis([0 0.05]);

% % %% 4 Linear correlation on spikes
% % % Are neurons that respond to stimuli correlated to each other
% % % Not really high correlations on spikes either
% % % Pearson's correlation by default
% % [RHO,PVAL]  = corr(allSpikes');
% % figure;
% % imagesc(RHO);
% % colorbar;
% % caxis([0 0.05]);

% % There are many ways we could continue exploring this, but by know, we
% % would say that, as corrleations beween neurons is low, we move on
% % 
% % %% Dimensionality reduction
% % [pcoeff,score,latent,tsquare] = princomp(psth');
% % 
% % most_var = cumsum(latent)./sum(latent);
% % most_var(1:40)
% % % most_var(1:40) --> you need 
% % figure;
% % biplot(pcoeff(:,1:2),'Scores',score(:,1:2))
     

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% UNSUPERVISED LEARNING %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. K-means on synthetic data

%% 5. K-means on the psth
clc
numberNeur  = size(psth, 1);
k_cluster   = 2;
[idx,C, sums, D] = kmeans(psth,k_cluster);

[idxspike,C]    = kmeans(allSpikes,k_cluster, 'Distance', 'hamming');
 

subplot(2,1,1);
h = plot(mean(psth(idx==1,:))); hold on
plot(mean(psth2),'r');
title('instantaneous firing rate - PSTH for cluster 1 ')
set(gca, 'XTickLabel', 0:totalTime/10:totalTime)
ylabel('Firing probability')

subplot(2,1,2);
h = plot(mean(psth(idx==2,:))); hold on
plot(mean(psth2),'r');
title('instantaneous firing rate - PSTH for cluster 2')
set(gca, 'XTickLabel', 0:totalTime/10:totalTime)
ylabel('Firing probability')

% How meaningful is this classification?
% Here we need a measure of goodness of fit

%% Distance to centroid 
clc
numberNeur  = size(psth, 1);
k_cluster   = 2;
PermVal     = 100;
clustDist    = zeros(PermVal,k_cluster);
clustDistR   = zeros(PermVal,k_cluster);

for crs = 1:PermVal
    [idx,C, sums, D]         = kmeans(psth,k_cluster);
    [idxspike,C, sums, DR]   = kmeans(allSpikes,k_cluster, 'Distance', 'hamming');

    % Number of neurons within cluster 1
    clustDist(crs,:)   = mean(D);
    clustDistR(crs,:)  = mean(DR);
     
end

mean(clustDist)
mean(clustDistR) 

% % is the difference between randomized psth and real psth?
p_value         = ranksum(clustDist(:,1), clustDistR(:,1)) 

% although we cannot visualize, seems like distance of real psth vs
% randomized psth, is smaller

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% INCLUDE EXTRA INFO %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% stimulus information
stimulus_in     = 500; %ms
stimulus_out    = 3500;
 
plot(mean(psth),'r');
title('instantaneous firing rate - PSTH')
set(gca, 'XTickLabel', 0:totalTime/10:totalTime)
ylabel('Firing probability')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% SUPERVISED LEARNING %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GLM on spikes
addpath(genpath('glmnet_matlab'))

teacherSignal   = [zeros(stimulus_in,1) ; ones(3000,1) ; zeros(totalTime-stimulus_out,1)];
w_linReg        = glmfit(allSpikes',teacherSignal,'binomial','constant','off');


% Find highest relevance neurons
highestRelevance    = sort(w_linReg);
nhighestRelevance   = 50;
neuronIDX_highestRelevance  = zeros(nhighestRelevance,1);
neuronIDX_highestRelevanceR = zeros(nhighestRelevance,1);
for idx =1:nhighestRelevance
    neuronIDX_highestRelevance(idx)     = find(w_linReg==highestRelevance(idx)); 
end

% THIS IS SYNTHETIC DATA -
% NEURONS THAT HAVE INCREASING FR AS A RESULT OF STIMULI ARE neuronsWithInput

correct_classified = length(intersect(neuronIDX_highestRelevance, neuronsWithInput));
perent_CorrectClas = correct_classified/50*100

% Compare to random labels
neuronsWithInputR   = randperm(160);
neuronsWithInputR   = neuronsWithInputR (1:80);
correct_classifiedR = length(intersect(neuronIDX_highestRelevance, neuronsWithInputR));
perent_CorrectClasR = correct_classifiedR/50*100

% A linear model SEEMS to be able to do the classification
% What are we missing? Model validation
% Can we do it? We would need WAY MORE DATA



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%% OTHER CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Inter-spike intervals
% figure
% NR_spikes       = length(firings);
% Delta_times     = firings(2:NR_spikes)-firings(1:NR_spikes-1);
% ISI_Grid        = 1/((max(Delta_times)-min(Delta_times))*totalTime/10000);
% ISI_Grid_vec    = 0:ISI_Grid:max(Delta_times);
% 
% histogram_ISI  = hist(Delta_times,ISI_Grid_vec);
% hist(Delta_times,ISI_Grid_vec)
% title('ISI Distribution for  Neurons');
% xlabel('dt(close msec)');
% set(gca,'XLim',[0 max(Delta_times)]);


break;

% figure;
% % from spiking matrix
% [neurons, time] = find(allSpikes==1);
% plot(time, neurons,'.'); hold on;
% 
% [neurons, time] = find(allSpikes(neuronIDX_highestRelevance,:)==1);
% plot(time, neurons,'r.'); hold on;
% 
% figure;
% nonhighestRelevance = setdiff(1:160,neuronIDX_highestRelevance);
% plot(mean(psth(nonhighestRelevance(1:50),:))); hold on;
% plot(mean(psth(neuronIDX_highestRelevance,:)),'r'); hold on;
% title('instantaneous firing rate - PSTH')
% set(gca, 'XTickLabel', 0:totalTime/8:totalTime)
% ylabel('Firing probability')


