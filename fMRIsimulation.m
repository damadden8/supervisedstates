function [simulatedTS, statesequence, cov, meta] = fMRIsimulation(nreg,ntime,nsub,varargin)

%nreg: Number of regions in simulated timeseries. Final three will be A,B,C
%ntime: Length of simulated timeseries
%nsub: Number of simulated subjects

seed = 1;
rng(seed) %Reproducibility

y = normrnd(0,1,[nsub,1]); %Randomly generate covariates according to standard normal

% ADDITIONAL PARAMETERS (name/value) (Default Shown)
% 
% -- Timing --
% 'TR'            (2.0)
%
%   % Latent-network (modular FC):
% 'Knet'          (5)          % Number of modules
% 'NetAssign'     ([])         % Predefined network assignment if desired
% 'WithinLoad'    (0.7)        % Baseline strength of intra-module connections 
% 'BetweenLoad'   (-0.8, 0.4)  % Range of values for baselin inter-module connection strength
% 'NetSD'         (0.8)        % Standard deviation of module signal
%   
%   % Target-FC (exact stationary covariance):
% 'RhoAR'         (0.4)        % AR(1) coefficient in multivariate AR
% 'PSDdelta'      (1e-6)       % diagonal jitter if near-singular
%
% -- Neural local dynamics --
% 'NeuralSD'      (0.8)        % Standard deviation of neural signal
% 'PinkAlpha'     (1.0)        % 1/f^alpha
% 'PinkSD'        (0.5)        % Standard deviation of pink noise
%
% -- A-extreme detection & pulses --
% 'Smooth'        (true)       % smooth A before z-score
% 'Win'           (7)          % moving-average window
% 'Thr'           (1.0)        % z-threshold for extremes
% 'Kappa'         (3.0)        % sigmoid slope for p_high
% 'Pulse'         (2.2)        % neural pulse amplitude for (A,B,C) when pulsing
% 'PulseJit'      (0.25)       % jitter SD per pulsed TR
% 'PulseProbBC'   (0.40)       % probability that B or C receives a pulse when A is extreme
% 'eventMinDur'   (10)         % minimum duration of extreme events (states
% 'eventMaxDur'   (30)         % maximum duration of extreme events
%
% -- HRF --
% 'HrfLen'        (32)         
% 'HrfP1'         (6)          
% 'HrfP2'         (16)        
% 'HrfK1'         (1)          
% 'HrfK2'         (1)          
% 'HrfRatio'      (6)          
%
% -- BOLD-level nuisances --
% 'DriftSD'       (0.7)        % Noise parameters
% 'PhysSD'        (0.35)
% 'RespHz'        (0.28)
% 'CardHz'        (1.10)
% 'WhiteSD'       (0.35)
% 'BoldPinkSD'    (0.25)
%
% -- Post-processing --
% 'Detrend'       (true)
% 'Bandpass'      ([0.01 0.1]) % Hz; [] to disable
% 'Zscore'        (true)
% -------------------------------------------------------------------------

%% Parse
p = inputParser; p.KeepUnmatched = true;
addParameter(p,'TR',2.0);

% FC
addParameter(p,'FCMode','latent'); %May provide ideal FC matrix if desired, otherwise latent
addParameter(p,'Knet',5);
addParameter(p,'NetAssign',[]);
addParameter(p,'WithinLoad',0.7);
addParameter(p,'BetweenLoadLower',-0.8); 
addParameter(p,'BetweenLoadHigher',0.4);
addParameter(p,'NetSD',0.8);

addParameter(p,'Rtarget',[]);
addParameter(p,'RhoAR',0.4);
addParameter(p,'PSDdelta',1e-6);

% Neural local
addParameter(p,'NeuralSD',0.8);
addParameter(p,'PinkAlpha',1.0);
addParameter(p,'PinkSD',0.5);

% A-extreme & pulses
addParameter(p,'Smooth',true);
addParameter(p,'Win',7);
addParameter(p,'Thr',1.0);
addParameter(p,'Kappa',3.0);
addParameter(p,'Pulse',2.2);
addParameter(p,'PulseJit',0.25);
addParameter(p,'PulseProbBC',0.40); 
addParameter(p,'eventMinDur',10);
addParameter(p,'eventMaxDur',30);

% HRF
addParameter(p,'HrfLen',32);
addParameter(p,'HrfP1',6);
addParameter(p,'HrfP2',16);
addParameter(p,'HrfK1',1);
addParameter(p,'HrfK2',1);
addParameter(p,'HrfRatio',6);

% BOLD nuisances
addParameter(p,'DriftSD',0.7);
addParameter(p,'PhysSD',0.35);
addParameter(p,'RespHz',0.28);
addParameter(p,'CardHz',1.10);
addParameter(p,'WhiteSD',0.35);
addParameter(p,'BoldPinkSD',0.25);

% Post
addParameter(p,'Detrend',true);
addParameter(p,'Bandpass',[0.01 0.1]);
addParameter(p,'Zscore',true);

parse(p,varargin{:});
pr = p.Results;

%% y

%% Indices
Aidx = nreg - 2; Bidx = nreg - 1; Cidx = nreg;

%% Allocate
data = zeros(nreg, ntime, nsub);

%% ---------- Helpers ----------
    function x = ar1_series(rho_, sd_, T) %Generate autoregressive, latent (neural) signal
        x = zeros(T,1); eps = randn(T,1)*sd_;
        for ttt=2:T, x(ttt) = 0.4*x(ttt-1) + eps(ttt); end
    end
    function g = gamma_pdf(t,a,b) %Gamma pdf for HRF construction
        g = (t.^(a-1).*exp(-t./b)) ./ (gamma(a)*(b^a)); g(t<0)=0;
    end
    function h = make_hrf(TR_, lenSec, P1, P2, K1, K2, ratio) %Construct HRF
        t=(0:TR_:lenSec)'; a1=P1/K1; b1=K1; a2=P2/K2; b2=K2;
        h1=gamma_pdf(t,a1,b1); h2=gamma_pdf(t,a2,b2);
        h=(h1 - h2/ratio); h = h/(sum(abs(h))+1e-12);
    end
    function x = pink_noise_fft(T, alpha, sd_) %Add nuisance pink noise
        w = randn(T,1); F = fft(w); k=(0:T-1)'; k(1)=1;
        mag = 1./(k.^(alpha/2)); F = F.*mag;
        x = real(ifft(F)); x = x-mean(x); x = x/(std(x)+1e-12)*sd_;
    end
    function x = detrend_lin(x) %Detrend simulated signal
        T=numel(x); t=(1:T)'; A=[ones(T,1) t]; b=A\x; x = x - A*b;
    end
    function x = fft_bandpass(x, TR_, f_lo, f_hi) %Frequency filtering
        T=numel(x); fs=1/TR_; X=fft(x); f=(0:T-1)'*(fs/T);
        keep=(f>=f_lo & f<=f_hi) | (f>=(fs-f_hi) & f<=(fs-f_lo));
        X(~keep)=0; x=real(ifft(X));
    end
    function [Y] = mvAR1_from_targetR(T, R, rhoAR, deltaPSD) %Construct latent signal based on provided FC matrix
        % y_t = rhoAR*y_{t-1} + e_t, choose Q=(1-rhoAR^2)*R so Cov=R
        Rpsd=(R+R')/2; [V,D]=eig(Rpsd); D=diag(D); D(D<deltaPSD)=deltaPSD;
        Rpsd=V*diag(D)*V';
        Q=(1-rhoAR^2)*Rpsd;
        [L,pch]=chol(Q,'lower');
        if pch>0
            [V2,D2]=eig((Q+Q')/2); D2=diag(D2); D2(D2<deltaPSD)=deltaPSD;
            L = V2*diag(sqrt(D2));
        end
        Y=zeros(T,size(R,1));
        for t=2:T
            et = randn(1,size(R,1))*L';
            Y(t,:) = rhoAR*Y(t-1,:) + et;
        end
    end

%% HRF
hrf = make_hrf(pr.TR, pr.HrfLen, pr.HrfP1, pr.HrfP2, pr.HrfK1, pr.HrfK2, pr.HrfRatio); %Construct HRF function for convolution

%% Meta
meta = struct(); %Set up meta structure for state identification
meta.hrf = hrf; meta.FCMode = pr.FCMode; meta.params = pr;
meta.p_high = zeros(nsub,1);
meta.A_extreme_sign = cell(nsub,1);
meta.NetAssign = []; meta.Rtarget_used = [];

%% Time vector
tsec = (0:ntime-1)' * pr.TR; 

% Network modules
K = pr.Knet;
if isempty(pr.NetAssign)
    NetAssign = randi([1,K], [nreg,1]); %Randomly assign each region to a module
    Netweights = zeros([K,K]);
    for kk = 1:K
        Netweights(kk,kk) = pr.WithinLoad; %Within module base connection strength
        for jj = kk+1:K
            Netweights(kk,jj) = (randi(120)-80)/100; %Between network loads
            Netweights(jj,kk) = Netweights(kk,jj);
        end
    end

else
    NetAssign = pr.NetAssign(:);
    assert(numel(NetAssign)==nreg, 'NetAssign must have length nreg');
    K = max(NetAssign);
    Netweights = zeros([K,K]);
    for kk = 1:K
        Netweights(kk,kk) = pr.WithinLoad; %Within module base connection strength
        for jj = kk+1:K
            Netweights(kk,jj) = (randi(120)-80)/100; %Between network loads
            Netweights(jj,kk) = Netweights(kk,jj);
        end
    end
end
meta.NetAssign = NetAssign;

seq = zeros([nsub,ntime]);
%% ====================== Main loop ==========================
for s = 1:nsub

    % -------- Resting-state NEURAL baseline with FC --------
    switch lower(pr.FCMode)
        case 'latent'
            % Network factors
            nets = zeros(K, ntime);
            for kk=1:K
                nets(kk,:) = (ar1_series(pr.RhoAR, pr.NetSD, ntime) + ... %Generate autoregressive series for module
                             pink_noise_fft(ntime, pr.PinkAlpha, pr.NetSD*0.6))';
            end

            

            W = zeros(nreg, K);
            for r = 1:nreg
                for rr = r+1:nreg
                    W(r,rr) = normrnd(Netweights(NetAssign(r),NetAssign(rr)),0.4); %Randomly generate regional connection strength based on networks
                    W(rr,r) = W(r,rr);
                end
            end


            neural = zeros(nreg, ntime);
            for r = 1:nreg
                regional(r,:) = ar1_series(pr.RhoAR, pr.NeuralSD, ntime); %Generate regional neural signal
                colored  = pink_noise_fft(ntime, pr.PinkAlpha, pr.PinkSD);
                
            end
            for r = 1:nreg
                neural(r,:) = W(r,:)*regional+ regional(r,:) + colored'; %Add noise sources and weighted combination of other signals to instill connections
                neural(r,:) = zscore(neural(r,:));
            end

        case 'target'
            R = pr.Rtarget;
            assert(~empty(R) && all(size(R)==[nreg,nreg]), 'Rtarget must be (nreg x nreg)');
            Y = mvAR1_from_targetR(ntime, R, pr.RhoAR, pr.PSDdelta); % T x nreg
            neural = Y';
            % small local terms for realism
            for r=1:nreg %Generate neural signals based on provided FC matrix
                neural(r,:) = neural(r,:) + ...
                              ar1_series(0.3, 0.2*pr.NeuralSD, ntime)' + ...
                              pink_noise_fft(ntime, pr.PinkAlpha, 0.2*pr.PinkSD)';
                
            end
            meta.Rtarget_used = R;

        otherwise
            error('FCMode must be ''latent'' or ''target''.');
    end

    % -------- A extreme detection (neural) --------
    A = neural(Aidx,:)';
    Aproc = A;
    if pr.Smooth && pr.Win>1 %Convolve A signal with moving window average to identify persistent extremes
        k = ones(pr.Win,1)/pr.Win;
        Aproc = conv(Aproc, k, 'same');
    end
    Az = (Aproc - mean(Aproc)) / (std(Aproc) + 1e-12);
    signA = zeros(ntime,1);
    signA(Az >  pr.Thr) = +1;
    signA(Az < -pr.Thr) = -1; %Identify extreme points and directions
    meta.A_extreme_sign{s} = signA;

    % -------- y -> mapping probability --------
    p_high = 1 / (1 + exp(-pr.Kappa * y(s)));  % high y => A&B match, C oppose
    meta.p_high(s) = p_high;

    
% ===================== EVENT-BASED PULSE INJECTION =====================

    pulseAmp = pr.Pulse;
    jitSD    = pr.PulseJit;
    p_gate   = pr.PulseProbBC;
    
    
    tt = 1;
    
    while tt <= ntime
    
        % Only start an event when entering an extreme state
        if signA(tt) == 0 || (tt > 1 && signA(tt) == signA(tt-1))
            tt = tt + 1;
            continue;
        end
    
        % Sign of A for this event
        eventSign = signA(tt);
    
        % Draw event length uniformly from 10:30 TRs
        dur = randi([pr.eventMinDur pr.eventMaxDur]);
    
        % Don't exceed scan length
        tEnd = min(tt + dur - 1, ntime);
        
        % Decide mapping ONCE for the whole event
        high_map = (rand < p_high);
    
        % Decide whether B and C participate ONCE for the whole event
        pulseB = (rand < p_gate);
        pulseC = (rand < p_gate);
        pulseRes = pulseB | pulseC;
   

        Aext = pulseAmp * eventSign;
        extA = 1 * eventSign;
    
        for tEvent = tt:tEnd
    
            % A always follows the event if B or C do
            if pulseB || pulseC
                neural(Aidx,tEvent) = neural(Aidx,tEvent) + ...
                    extA + randn()*jitSD;
            end
    
            % B
            if pulseB
                if high_map
                    % A and B match
                    neural(Bidx,tEvent) = neural(Bidx,tEvent) + ...
                        Aext + randn()*jitSD;
                else
                    % B opposes A
                    neural(Bidx,tEvent) = neural(Bidx,tEvent) - ...
                        Aext + randn()*jitSD;
                end
            end
    
            % C
            if pulseC
                if high_map
                    % C opposes A
                    neural(Cidx,tEvent) = neural(Cidx,tEvent) - ...
                        Aext + randn()*jitSD;
                else
                    % A and C match
                    neural(Cidx,tEvent) = neural(Cidx,tEvent) + ...
                        Aext + randn()*jitSD;
                end
            end
            
            %Align state sequences

            if eventSign==1 && pulseRes && high_map==1
                seq(s,tEvent) = 1;
            elseif eventSign==1 && pulseRes && high_map==0
                seq(s,tEvent) = 2;
            elseif eventSign==-1 && pulseRes && high_map==1
                seq(s,tEvent) = 3;
            elseif eventSign==-1 && pulseRes && high_map==0
                seq(s,tEvent) = 4;
            end
    
        end
    
        % Jump to end of event
        tt = tEnd + 1;
    
    end

% ================= END EVENT-BASED PULSE INJECTION =====================


    % -------- HRF convolution -> BOLD --------
    bold = zeros(nreg, ntime);
    for r = 1:nreg
        tmp = conv(neural(r,:)', hrf, 'full');
        bold(r,:) = tmp(1:ntime)';
    end

    % -------- Add BOLD-level nuisances --------
    % Drift
    drift = cumsum(randn(ntime,1)); drift = drift - mean(drift);
    drift = drift / (std(drift)+1e-12) * pr.DriftSD;

    % Physiological signals
    phiR = 2*pi*rand; phiC = 2*pi*rand;
    resp = sin(2*pi*pr.RespHz*tsec + phiR);
    card = sin(2*pi*pr.CardHz*tsec + phiC);
    phys = resp + 0.5*card; phys = phys - mean(phys);
    phys = phys / (std(phys)+1e-12) * pr.PhysSD;

    % Colored + white
    boldPink = pink_noise_fft(ntime, pr.PinkAlpha, pr.BoldPinkSD);
    white    = randn(ntime,1) * pr.WhiteSD;

    for r = 1:nreg
        scale = 0.8 + 0.4*rand;
        bold(r,:) = bold(r,:) + (scale*drift)' + (scale*phys)' + boldPink' + white';
    end

    % -------- Post-processing --------
    for r = 1:nreg
        x = bold(r,:)';
        if pr.Detrend,  x = detrend_lin(x); end
        if ~isempty(pr.Bandpass), x = fft_bandpass(x, pr.TR, pr.Bandpass(1), pr.Bandpass(2)); end
        if pr.Zscore,   x = (x - mean(x)) / (std(x)+1e-12); end
        bold(r,:) = x';
    end

    data(:,:,s) = bold;
end

simulatedTS = data;
cov = y;
statesequence = seq;

end



