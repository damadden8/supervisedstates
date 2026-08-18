
function [simulatedTS, statesequence,cov] = discretesimulation(nreg,ntime,nsub,odds,duration)
%   nreg: Number of simulated regions in the dataset. Behavior-related regions will be the final 2 regions
%   ntime: Length of timeseries
%   nsub: Number of simulated subjects
%   Odds: Probability that regions will take on discrete values. Should be a 3 value vector summing to 1 [0.2, 0.6, 0.2]
%   Duration: 2 value vector with minimum and maximum duration bounds [5,30].

if nargin < 4
    odds = [0.2,0.6,0.2];
    duration = [5,30];
    duration(2) = duration(2) - duration(1);
elseif nargin == 4
    duration = [5,30];
    duration(2) = duration(2) - duration(1);
end

seed = 1;
rng(seed); %For reproducibility

oddsdep = odds;

choices = [-1,0,1]; %Potential timeseries values. Do not change.

simts = zeros(nreg,ntime+duration(2),nsub); %Preallocate timeseries array

cov = normrnd(0,1,[nsub,1]); %Generate covariates from standard normal
cov = (cov - min(cov))/(max(cov)-min(cov)); %Min-max transform covariates

for n = 1:nsub
    for i = 1:nreg-2
        t = 1;
        while t < ntime
            dur = duration(1)+randi(duration(2),1); %Establish duration for this region
            simts(i,t:t+dur,n) = randsample(choices,1,true,odds); %Assign value to region for duration
            t = t+dur+1; %Align t value
        end
    end
    t = 1;
    while t < ntime
        dur = duration(1)+randi(duration(2),1); %Establish duration for region A
        act = randsample(choices,1,true,odds); %Determine value for region A
        simts(nreg-1,t:t+dur,n) = act; %Asign A's value for duration
        if act == 0
            oddsdep(1) = odds(1); %If A is neutral, keep odds the same
            oddsdep(3) = odds(3);
        else
            oddsdep(1) = 0.4-(0.4*cov(n)); %If A is extreme, determine odds based on cov
            oddsdep(3) = 0.4*cov(n);
        end
        simts(nreg,t:t+dur,n) = randsample(choices,1,true,oddsdep); %Assign region B its value for duration
        t = t+dur+1;%Align t value
    end
end
simts(:,ntime+1:end,:) = [];
simulatedTS = simts;
statesequence = zeros([nsub,ntime]); %preallocate state sequence
for k = 1:nsub
    Apos = find(simts(6,:,k)==1); %Find where A and B both take extreme values
    Aneg = find(simts(6,:,k)==-1);
    Bpos = find(simts(7,:,k)==1);
    Bneg = find(simts(7,:,k)==-1);
    statesequence(k,intersect(Apos,Bpos)) = 1; %State 1: High A High B
    statesequence(k,intersect(Apos,Bneg)) = 2; %State 2: High A Low B
    statesequence(k,intersect(Aneg,Bpos)) = 3; %State 3: Low A High B
    statesequence(k,intersect(Aneg,Bneg)) = 4; %State 4: Low A Low B
end

end
