close all
clear all
clc
dbstop if error;
% Created by Eugene M. Izhikevich, February 25, 2003
% Excitatory neurons    Inhibitory neurons
Ne = 80;                 Ni = 0;
re = rand(Ne,1);         ri = rand(Ni,1);

% Parameter neuron type
a=[0.02*ones(Ne,1);     0.02+0.08*ri];
b=[0.2*ones(Ne,1);      0.25-0.05*ri];
% Parameter neuron dynamic
c=[-65+15*re.^2;        -65*ones(Ni,1)];
d=[8-6*re.^2;           2*ones(Ni,1)];
% Paramater connectivity
S=[0.5*rand(Ne+Ni,Ne),  -1.2*rand(Ne+Ni,Ni)];

totalTime  = 10200;
offsetTime = 1000;
%% Offset - run for 1000ms
% initialize
v_ini  =-65*ones(Ne+Ni,1);    % Initial values of v
v_iniM = zeros(Ne+Ni,offsetTime);
u_ini=b.*v_ini;             % Initial values of u
offset=[];                  % spike timings
for t=1:offsetTime            % simulation of 1000 ms
  I             = [5*randn(Ne,1);2*randn(Ni,1)]; % thalamic input
  fired         = find(v_ini>=30);    % indices of spikes
  offset        = [offset; t+0*fired,fired];
  v_ini(fired)  = c(fired);
  u_ini(fired)  = u_ini(fired)+d(fired);
  I     = I+sum(S(:,fired),2);
  v_ini = v_ini +0.5*(0.04*v_ini.^2+5*v_ini+140-u_ini+I); % step 0.5 ms
  v_ini = v_ini +0.5*(0.04*v_ini.^2+5*v_ini+140-u_ini+I); % for numerical
  v_iniM(:,t) = v_ini; 
  u_ini = u_ini+a.*(b.*v_ini-u_ini);                 % stability
end;
plot(offset(:,1),offset(:,2),'.');
 
%% Run1 for totalTime (ms)
v       = v_ini;
v_mat   = -65*ones(Ne+Ni,totalTime);
u       = b.*v_ini;
firings=[];
for t=1:totalTime-1           % simulation of 1000 ms
    I   = [5*randn(Ne,1);2*randn(Ni,1)]; % thalamic input
%     if and(t>900, t<3900)
%         I_in = [3*randn(Ne,1);2*randn(Ni,1)]; % external input
%         I    = I + I_in;
%     end
    fired       = find(v>=30);    % indices of spikes
    firings     = [firings; t+0*fired,fired];
    v(fired)    = c(fired);
    u(fired)    = u(fired)+d(fired);
    I       = I+sum(S(:,fired),2);
    v       = v+0.5*(0.04*v.^2+5*v+140-u+I); % step 0.5 ms
    v       = v+0.5*(0.04*v.^2+5*v+140-u+I); % for numerical
    v_mat(:,t) = v;
    u       = u+a.*(b.*v-u);                 % stability
end;
discardIDX              = firings(:,1)<200;
firings(discardIDX,:)   = [];
firings(:,1)            = firings(:,1)-199;

plot(firings(:,1),firings(:,2),'.'); hold on;

% Make spike matrix
spikes = zeros(Ne+Ni, totalTime-200);
for frIDX = 1:length(firings)
    spikes(firings(frIDX,2), firings(frIDX,1)) = 1;
end

%% Run1 for totalTime (ms)
v       = v_ini;
v_mat2  = -65*ones(Ne+Ni,totalTime);
u       = b.*v_ini;

I_inScale = 0+3/3000:3/3000:3;
I_inScale = [zeros(1,500) I_inScale zeros(1,totalTime-3500) ];

firings2=[];
for t=1:totalTime-1            % simulation of 1000 ms
  I     = [5*randn(Ne,1);2*randn(Ni,1)]; % thalamic input
  if and(t>900, t<3900)
      I_in  = [I_inScale(t)*randn(Ne,1); 2*randn(Ni,1)]; % external input
      I     = I + I_in;
  end
  fired    = find(v>=30);    % indices of spikes
  firings2 = [firings2; t+0*fired,fired];
  v(fired) = c(fired);
  u(fired) = u(fired)+d(fired);
  I = I+sum(S(:,fired),2);
  v = v+0.5*(0.04*v.^2+5*v+140-u+I); % step 0.5 ms
  v = v+0.5*(0.04*v.^2+5*v+140-u+I); % for numerical
  v_mat2(:,t) = v;
  u = u+a.*(b.*v-u);                 % stability
end;
discardIDX              = firings2(:,1)<200;
firings2(discardIDX,:)  = [];
firings2(:,1)           = firings2(:,1)-199;

plot(firings2(:,1),firings2(:,2),'r.');

% Make spike matrix
spikes2 = zeros(Ne+Ni, totalTime-200);
for frIDX = 1:length(firings2)
    spikes2(firings2(frIDX,2), firings2(frIDX,1)) = 1;
end

%% Visualize single trace
v_mat(:,1:200)      = [];
v_mat2(:,1:200)     = [];

v_mat(v_mat>30)     = 30;
v_mat2(v_mat2>30)   = 30;
figure;
plot(v_mat(1,:)); hold on;
plot(v_mat2(1,:),'r');

all_v_mat = [v_mat ; v_mat2];


%% Randomize stuff
randidx     = randperm(Ne*2);
allSpikes   = [spikes ; spikes2];

firingShift      = firings2;
firingShift(:,2) = firingShift(:,2)+80;
allFirings       = [firings ; firingShift];

neuronsWithInput = find(randidx<80);

allSpikes   = allSpikes(randidx,:);

figure
plot(allFirings(:,1),allFirings(:,2),'.');

totalTime= totalTime-200;
%% Save

save('firingTimes', 'neuronsWithInput', 'totalTime', 'firings', 'firings2', 'spikes', 'spikes2', 'v_mat', 'v_mat2', 'allFirings', 'allSpikes', 'all_v_mat')