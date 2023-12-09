# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 01:29:10 2023

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import scipy.stats
from scipy.optimize import minimize
from scipy.optimize import fsolve

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

### test and learn ABC method
# here we start with a simple Poisson GLM
# the goal is to move on to explore simple (3-5 neuron) circuit and chemotaxis
# this should outperform EA method classically used in worms and also provide uncertainty measure

# %% functions for Bayesian~~

def makeHypers(hypersOld, params, nupdate):
    """
    updates the prior distributions of the ribbon
    ---
    :param hypersOld: old hyperparameters of the ribbon distributions
    :param params: sampled parameters, already sorted by loss. shape: (n_samples, dim)
    :param nupdate_raw: taking into account the n best samples (for updating)
    ---
    :return: updated priors
    """

    ## multivariate normal distribution
    mu0s, Lambda0, kappa0, nu0 = hypersOld

    dim = len(mu0s) # dimension of parameter space

    mean_samples = np.mean(params[:nupdate], axis=0)
    muns = 1 / (kappa0 + nupdate) * (kappa0 * mu0s + nupdate * mean_samples)
    kappan = kappa0 + nupdate
    nun = nu0 + nupdate

    # compute matrix with y_i - mean(y)
    yi_y = params[:nupdate] - np.tile(mean_samples, nupdate).reshape(nupdate,dim)
    # compute s: sum (y - mean(y)) * (y - mean(y)).T (dim x dim corr matrix * n)
    s = np.sum([yi_y[0].reshape(dim, 1) *yi_y[0].reshape(1, dim) for i in range(len(yi_y))], axis=0)

    Lambdan = (Lambda0
               + s
               + (kappa0 * nupdate) / (kappa0 + nupdate) * np.dot((mean_samples - np.array(mu0s)).reshape(dim, 1),
                                                                  (mean_samples - np.array(mu0s)).reshape(1, dim)))
    hypers = [muns, Lambdan, kappan, nun]

    return hypers

def drawSamps(hyperMat, nSamps):
    """
    draw samples from priors using updated hyper parameters
    including drawing of Sigma for normal distribution
    ---
    :param hyperMat: parameters from makeHypers
    :param nSamps: how many samples to draw
    ---
    :return:
    """
    muns, Lambdan, kappan, nun = hyperMat

    sigmas = sp.stats.invwishart(df=nun, scale=Lambdan).rvs(
        size=nSamps)  # take care scale is here the "precision" E[~] = ...*Lambdan

    draws = np.array([sp.stats.multivariate_normal(mean=muns, cov=sigmas[i]).rvs() for i in range(nSamps)])
    
    return draws

def makeHypersPrior():
    """
    :return: hyper parameters for the prior distributions
    ---
    the dimension k of the parameter space is indirectly specified here
    """

    prior_means = np.array([0,0,0,0,0])
    lambda0 = np.eye(5)*.1 # approximatley cov of prior (param. for Inv-Wishart-distribution)
    
    # parameters for Inv-Wishart should be changed if the dimension k changes
    kappa0 = 1
    nu0 = 5  #### study this~~~ ####

    hypers = [prior_means, lambda0, kappa0, nu0]
    
    return hypers

# %% simulation functions
def NL(x):
    return 1/(1 + np.exp(-x))

def model_glm(params):
    conv = np.convolve(params, stim, mode='same')
    p_spk = NL(conv)
    spk = np.zeros(T)
    spk[p_spk > np.random.rand(T)] = 1
    return spk, p_spk  ### reutrn spikes or probability!

def model_mult(params_list, batchsize):
    dim_data = T*1
    nr_traces = len(params_list)
    traces = np.zeros((nr_traces, batchsize, dim_data))
    for i in range(nr_traces):
        for j in range(batchsize):
            _,traces[i,j,:] = model_glm(params_list[i])
    return traces

def extract_sumstat(data):
    return data

epsilon = 1e-10  # Small constant to prevent taking the log of zero
def poisson_loss(y_pred, y_true):
    y_pred = np.maximum(y_pred, epsilon)  # Add epsilon to predicted values
    loss = y_pred - y_true * np.log(y_pred)
#    loss = (y_pred - y_true)**2
    return loss.mean()  # Return the mean loss across samples

def loss(traces, sumstat_data):
     # extract summary stats for traces
    sumstat_traces = extract_sumstat(traces)
    
    n = len(sumstat_data)
    m = len(sumstat_traces)
    
    # mean of pairwise distance
    MSE_pairwise = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            MSE_pairwise[i,j] = 1/(len(sumstat_data[0]))*poisson_loss(sumstat_traces[j], sumstat_data[i])
            #1/(len(sumstat_data[0]))* np.sum( (sumstat_data[i]-sumstat_traces[j])**2 )
                
    return np.mean(MSE_pairwise)

# %% setup ground truth data and parameters
T = 1000  # length of simulation
K_true = np.array([0.1,0.2,0.5,-1,0.1])*10  # kernel weights
stim = np.random.randn(T)  # random stimuli

ndata = 100
data = np.zeros((ndata, T))
for i in range(ndata):
    data[i],_ = model_glm(K_true)  # ndata x T spikes
    
# %% setup for sampling
# extract summary statistics for data
sumstat_data = extract_sumstat(data)

## number of samples
# if nr is to small error araises from taking best 1000 values etc.
nSamps0 = 500 # samples in first round
nSampsLate = 500 # samples in later rounds
# number of sample to update from (n_accept)
nupdate0 = 10 
nupdate_late = 50 
# number of simulations per parameter draw
batchsize = 3
# number of rounds
rounds = 20
# initialize list to save all hyper paramters
hypersSave = []
# make prior distribution
hypers = makeHypersPrior() 
hypersSave.append(hypers)

# %% actual computation here
# to print or not to print...
verbose = True

for noRun in range(0,rounds):
    if noRun ==0:
        nSamps = nSamps0
        nupdate = nupdate0
    else:
        nSamps = nSampsLate
        nupdate = nupdate_late
    
    if verbose:
        print()
        print('run',noRun)
    
    
    """
    sample parameter
    """
    params = drawSamps(hypers, nSamps)
    if verbose:
        print('finished sampling, now run simulations...')

    
    """
    run the model
    """
    traces = model_mult(params, batchsize)
    
    if verbose:
        print('finished model evaluations.')
        
        
    """
    calculate summary statistics
    """
    loss_params = np.zeros(nSamps)
    for i in range(nSamps):
        loss_params[i] = loss(traces[i], sumstat_data)

    
    """
    sort parameters
    """
    sorted_params = params[loss_params.argsort()]    
    
    if verbose:
        print('mean of accepted parameters:', np.mean(sorted_params[:nupdate], axis=0))
    
    """
    update the hyperparameters
    """
    hypers = makeHypers(hypers, sorted_params, nupdate)
    # save hypers
    hypersSave.append(hypers)

# %% check sampling
means = np.zeros((rounds, 5))
for ii in range(rounds):
    means[ii,:] = hypersSave[ii][0]
plt.plot(means)

# %% analysis
plt.figure()
K_abc = np.mean(sorted_params[:nupdate],axis=0)
#test = np.median(sorted_params[:nupdate],axis=0)
plt.plot(K_abc)
plt.plot(K_true,'--')

# %% simple MLE
def poisson_nLL(param, y_observed):
#    lamb_pred,_ = model_glm(param)
    conv = np.convolve(param, stim, mode='same')
    lamb_pred = NL(conv)
    log_likelihood = np.sum(y_observed * np.log(lamb_pred) - lamb_pred)
    return -log_likelihood

K0 = np.random.randn(len(K_true))
result = minimize(poisson_nLL, K0, args=(data[0,:]),method='L-BFGS-B')
K_inf = result.x
plt.figure()
plt.plot(K_inf)
plt.plot(K_true,'--')

# %% compare uncertainty
### Hessian for MLE
y_observed = data[0,:]*1
def log_likelihood(x):
    return -poisson_nLL(x, y_observed)

def compute_hessian(func, x):
    n = len(x)
    hessian = np.zeros((n, n))
    eps = 1e-5  # Small epsilon for numerical differentiation
    
    for i in range(n):
        for j in range(n):
            # Compute second partial derivatives using central finite differences
            f1 = func(np.copy(x))
            x[i] += eps
            f2 = func(np.copy(x))
            x[i] -= eps
            x[j] += eps
            f3 = func(np.copy(x))
            x[j] -= eps
            
            hessian[i, j] = (f3 - f2 - f2 + f1) / (eps ** 2)
    
    return hessian
initial_x = K_inf*1
hessian_matrix = compute_hessian(log_likelihood, initial_x)
std_deviations = np.sqrt(np.diagonal(hessian_matrix))

### compute for ABC
#K_abc_std = np.std(sorted_params[:nupdate],axis=0)
def get_stds(hypers):
    k = 5 # dim of parameters
    stds = np.zeros(k)

    for i in range(k):
        stds[i] = (1/(hypers[3] - k) * hypers[1][i,i] )**0.5 #  [muns,Lambdan, kappan, nun]
        #### study this too~~
    
    return stds
K_abc_std = get_stds(hypersSave[-1])

# %%
xx = np.arange(len(K_abc))
plt.figure()
plt.plot(xx, K_abc,'-o', label='ABC')
plt.fill_between(xx, K_abc-K_abc_std, K_abc+K_abc_std, alpha=0.2,color='blue')
plt.plot(xx, K_inf,'-o', label='MLE')
plt.fill_between(xx, K_inf-std_deviations, K_inf+std_deviations, alpha=0.2,color='red')
plt.legend(fontsize=20)
plt.ylabel('kernel weights', fontsize=25)
