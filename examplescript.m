%%%% Example Script for GA-based state selection on simulated data %%%


% Generate discrete simulation data
[simulatedTS, statesequence,cov] = discretesimulation(7,250,100,[0.2,0.6,0.2],[5,30]);

% Use GA to find covariate-related states
[discreteC, discretetraincorr, discreteoccupancy,discreteocccorr] = gastates(simulatedTS,cov,4,"lb",-1,'ub',1,'tol',0.0001,'stall',25,'MaxIter',200,'Trainprop',1,'disp','off','Par',0);

fprintf("Discrete prediction of randomly generated covariate: %f",discretetraincorr);

% Generate fMRI simulation data
[simulatedTS, statesequence, cov, meta] = fMRIsimulation(50,250,250);

% Use GA to find covariate-related states
[fmriC,fmritraincorr,fmrioccupancy,fmriocccorr] = gastates(simulatedTS,cov,4,'lb',-3,'ub',3,'disp','iter');

fprintf("fMRI prediction of randomly generated covariate: %f",fmritraincorr);

% Use GA on a subset of data, test on unseen subjects
[~,~,~,~,fmritestcorr] = gastates(simulatedTS,cov,4,'lb',-3,'ub',3,'Trainprop',0.8);