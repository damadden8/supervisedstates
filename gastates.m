function [centroids,traincorr,occupancy,occcorr,testcorr] = gastates(ts,y,n,options)
%%% INPUTS %%%
% ts: Timeseries. Should be in shape regions x time x subjects
% y: covariate of interest
% n: number of states
% lb: lower bound (for z-scored ts, -3. For discrete, -1)
% ub: upper bound (for z-scored ts, 3. For discrete, 1)
% tol: tolerance threshold (0.0001)
% stall: number of generations without average improvement greater than tol (25)
% MaxIter: maximum number of iterations
% Trainprop: proportion of data to fit states (1)
% disp: display option to show ga performance, 'iter' or 'off' ('off')
% Par: Use parallel (0)
% Additional GA options can be modified, refer to GA options on MATLAB site

%%% OUTPUTS %%%
%Centroids: State centroids
%traincorr: training subset prediction performance of y
%occupancy: Occupancy time for each subject
%occcorr: Correlation between each state's occupancy times and y
%testcorr: test subset prediction performance of y

arguments
    ts
    y
    n
    options.lb (1,1) double = -3
    options.ub (1,1) double = 3
    options.tol (1,1) double = 1e-4
    options.stall (1,1) double = 25
    options.MaxIter(1,1) double = 200
    options.Trainprop (1,1) double = 1
    options.disp (1,1) string = 'off'
    options.Par (1,1) logical = 0;
end


[i1,t1,k1] = size(ts);
lb = ones(i1*n,1);
ub = ones(i1*n,1);
lb = lb*options.lb;
ub = ub*options.ub;
if options.Trainprop < 1
    cv = cvpartition(length(y),'Holdout',1-options.Trainprop);
    idxtrain = training(cv,1);
    idxtest = test(cv,1);
    tstrain = ts(:,:,idxtrain);
    ytrain = y(idxtrain);
    ytest = y(idxtest);
else
    tstrain = ts;
    ytrain = y;
end


tempfunc = @(B)tempfunction(B,tstrain,ytrain,n);
optionsga = optimoptions('ga','StallGenLimit',options.stall,'MaxGenerations',options.MaxIter,'CreationFcn','gacreationuniform','CrossoverFcn','crossoverscattered','SelectionFcn','selectionstochunif','FitnessScalingFcn','fitscalingrank','MutationFcn',{@mutationuniform,0.05},'PopulationSize',40,'TolFun',options.tol,'Display',options.disp,UseParallel=options.Par);%,'SelectionFcn',{@selectiontournament,10}
[centroids,traincorr] = ga(tempfunc,i1*n,[],[],[],[],lb,ub,[],[],optionsga);
traincorr = 1 - traincorr;
%Testing GA Output
[~,occupancy,occcorr] = tempfunction(centroids,ts,y,n);
if options.Trainprop < 1
    occtrain = occupancy(idxtrain,:);
    occtest = occupancy(idxtest,:);
    mdl = fitglm(occtrain,ytrain);
    yhat = predict(mdl,occtest);
    testcorr = corr(yhat,ytest);
else
    testcorr = 0;
end
end

function [val,occupancy,occcorr] = tempfunction(B,ts,y,n)
[i1,t1,k1] = size(ts);
C = reshape(B,[i1,n]);
warning('off')

for k = 1:k1
    mat = ts(:,:,k);
    distmat = internal.stats.pdist2mex(mat,C,'sqeuclidean',[],[],[],[]);
    [~,stateinds] = min(distmat,[],2);
    for m = 1:n
        occupancy(k,m) =length(find(stateinds==m));
    end
end
occcorr = corr(occupancy,y);
mdl = fitglm(occupancy,y);
val = 1 - sqrt(mdl.Rsquared.Ordinary);
end