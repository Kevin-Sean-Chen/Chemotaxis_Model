# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:53:00 2021

@author: kevin
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import ndimage

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 
import h5py

# %% load dynamics Keito
fname = 'WT_Stim'
f = h5py.File('C:/Users/kevin/Downloads/'+fname+'.mat')
print(f[fname].keys())
all_trace = []
for ii in range(len(f[fname]['traces'])):
    temp = f[fname]['traces'][ii,0]
    all_trace.append(f[temp][:])

plt.figure()
plt.imshow(all_trace[0],aspect='auto')

# %% load recoding Leifer
from scipy import io
fname = 'C:/Users/kevin/Downloads/WBI_data/heatDataMS_1.mat'
mat = io.loadmat(fname)
behavior = mat['behavior']
Beh = behavior['v'][0][0].squeeze()
R2 = mat['Ratio2']

### preprocss nan
pos = np.argwhere(np.isnan(Beh))
Beh[pos] = np.nanmean(Beh)
Beh = Beh[None,:]

r2_ = R2.reshape(-1)
pos = np.argwhere(np.isnan(r2_))
r2_[pos] = np.nanmean(r2_)
R2 = r2_.reshape(R2.shape[0],R2.shape[1])

# %%
### recordings
#neuraldynamics = all_trace[0].copy()  # recordings

### connectome simulation
#neuraldynamics = Vs.copy()  #connectome simulation
#nn, timelen = neuraldynamics.shape  
#targ_dim = nn

### behavioral decoding (var x T)
neuraldynamics =Beh.copy()# np.arctanh(R2/5)#R2.copy()  #Beh.copy()#
nn, timelen = neuraldynamics.shape  
nn = R2.shape[0]*1
targ_dim = neuraldynamics.shape[0]
nn = 400

### initial params
N = nn  #number of neurons
p = .2  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.  #learning initial constant
dt = 0.1
simtime_len = timelen
learn_every = 2  #effective learning rate
tau = 5  #time scale

scale = 1.0/np.sqrt(p*N)  #scaling connectivity
M = np.random.randn(N,N)*g*scale
sparse = np.random.rand(N,N)
mask = np.random.rand(N,N)
mask[sparse>p] = 0
mask[sparse<=p] = 1
M = M*mask
#M, per,radius, theta = R_FORCEDistribution(N,g,0)

#uu,ss,vv = np.linalg.svd((JJ))
#JJ_ = uu[:,:20] @ np.diag(ss[:20]) @ vv[:20,:]
#M = M*J_s*J_dale #JJ.copy()

nRec2Out = N
wo = np.zeros((nRec2Out,targ_dim))
dw = np.zeros((nRec2Out,targ_dim))
#wf = np.ones((nRec2Out,targ_dim))*1
#wf = np.random.randn(nRec2Out,targ_dim)*.1
wf = 2.0*(np.random.rand(nRec2Out,targ_dim)-0.5)*.2

###target pattern
smooth = 200
#simtime = np.linspace(0,100,simtime_len)
ft = neuraldynamics*1#/np.max(neuraldynamics)*1.7 - 0.5
ft = ft - np.mean(ft)#,axis=0)
kern = np.hanning(smooth)   # a Hanning window with width 50
kern /= kern.sum()      # normalize the kernel weights to sum to 1
ft = ndimage.convolve1d(ft, kern, 1)*3
ft = ft/np.max(np.abs(ft),1)[:,None]*1.5

#test = np.sin(np.arange(0,simtime_len)/50)
#ft = np.repeat(test[None,:], targ_dim,axis=0)
#ft[ft<-1] = -1

#ft = np.convolve(ft[0,:],np.ones(smooth),'same')[None,:]/smooth*10
#amp = 0.7;
#freq = 1/60;
#rescale = 5.
#ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime*rescale) + \
#    (amp/2.0)*np.sin(2.0*np.pi*freq*simtime*rescale) + \
#    (amp/6.0)*np.sin(3.0*np.pi*freq*simtime*rescale) + \
#    (amp/3.0)*np.sin(4.0*np.pi*freq*simtime*rescale)
#ft = ft[None,:]/.5
plt.figure()
plt.plot(ft.T)

wo_len = np.zeros((1,simtime_len))    
zt = np.zeros((targ_dim,simtime_len))
x0 = 0.5*np.random.randn(N,1)
z0 = 0.5*np.random.randn(targ_dim)
xt = np.zeros((N,simtime_len))
rt = np.zeros((N,simtime_len))

x = x0
r = np.tanh(x)
z = z0

# %%
plt.figure()
ti = 0
#vE, vI = np.zeros(N), np.zeros(N)
#vE[pos_ex] = 1
#vI[pos_in] = 1
#P = (1.0/alpha)*np.eye(nRec2Out) + 1/2.*(np.outer(vE,vE)+np.outer(vI,vI)) + .1*np.cov(J_s)  #initialize with row-sum prior!
P = (1.0/alpha)*np.eye(nRec2Out)
for t in range(simtime_len-1):
    ti = ti+1
    x = (1.0-dt/tau)*x + M @ (r*dt)/tau #+ wf * (z*dt)
    r = np.tanh(x)
    rt[:,t] = r[:,0]  #xt[:,t] = x[:,0]
    z = wo.T @ r
    
    if np.mod(ti, learn_every) == 0:
        k = P @ r;
        rPr = r.T @ k
        c = 1.0/(1.0 + rPr)
        P = P - k @ (k.T * c)  #projection matrix
        
        # update the error for the linear readout
        e = z-ft[:,ti][:,None]
        
        # update the output weights
        dw = -k*e.T*c.squeeze()
        wo = wo + dw
        
        # update the internal weight matrix using the output's error
#        M = M + np.repeat(dw,N,1).T
#        M = M + dw
        M = M + (wf @ wo.T).T/1  #+ 0.005*J_s
#        M = M*J_dale
    
    # Store the output of the system.
    zt[:,ti] = np.squeeze(z)
    wo_len[0,ti] = np.sqrt(wo.T @ wo).sum()


zt = np.squeeze(zt)
error_avg = sum(abs(zt-ft))/simtime_len
print(['Training MAE: ', str(error_avg)])   
print(['Now testing... please wait.'])

plt.plot(ft.T)
plt.plot(zt.T,'--')

plt.figure()
plt.imshow(rt,aspect='auto')

# %%
zpt = np.zeros((targ_dim,simtime_len))
rpt = np.zeros((N,simtime_len))
ti = 0
x = x0
r = np.tanh(x)
z = z0
for t in range(simtime_len-1):
    ti = ti+1 
    
    x = (1.0-dt/tau)*x + M @ (r*dt)/tau #+ wf * (z*dt)
    r = np.tanh(x)
    rpt[:,t] = r.squeeze()
    z = wo.T @ r
    
    zpt[:,ti] = z.squeeze()


zpt = np.squeeze(zpt)
plt.figure()
plt.plot(zpt.T,label='model',linewidth=8)
plt.plot(ft[0,50:].T,'k--',label='data',linewidth=5)
plt.legend(fontsize=40)
plt.xlabel('time',fontsize=40)
plt.ylabel('velocity',fontsize=40)

plt.figure()
plt.subplot(211)
plt.imshow(rpt,aspect='auto')#(ft,aspect='auto')
plt.xticks([])
plt.ylabel('neuron',fontsize=40)
plt.subplot(212)
#plt.imshow(zpt,aspect='auto')
plt.plot(zpt[3,:].T,label='model',linewidth=8)
plt.plot(ft[3,:].T,'k--',label='data',linewidth=5)
plt.xlim([0,rpt.shape[1]])
plt.legend(fontsize=40)
plt.xlabel('time',fontsize=40)
plt.ylabel('velocity',fontsize=40)

# %%
plt.figure()
plt.plot(M[np.nonzero(JJ)], JJ[np.nonzero(JJ)],'.')
#plt.plot(M, JJ,'.')

# %% scan through N vs. decoding!
import statsmodels.api as sm
# %% fits
rhos = np.zeros(N)
for rr in range(N):
    temp = np.corrcoef(ft.reshape(-1), rt[rr,:])
    rhos[rr] = temp[0,1]
    
#    X2 = sm.add_constant(ft.reshape(-1))
#    est = sm.OLS(rt[rr,:], X2)
#    est2 = est.fit()
#    rhos[rr] = est2.rsquared
    
plt.figure()
plt.hist((rhos),30)

plt.figure()
plt.plot(rhos, wo[:,0],'o')

# %%
from sklearn.linear_model import Ridge
# %% single vs. population
sort_i = np.argsort(rhos)
X2 = sm.add_constant(ft.reshape(-1))
nr2 = np.zeros(N)
for rr in range(1,N):
    poss = sort_i[:rr]
    rt_i = rt[poss,:]
    
    ### via reconstruction
#    rec_n = wo[poss,:].T @ rt[poss,:]
#    est = sm.OLS(rec_n.squeeze(), X2)
    
    ### via regression
    clf = Ridge(alpha=1.)
    clf.fit(rt_i.T, ft.T)
    nr2[rr] = clf.score(rt_i.T, ft.T)
# %%
plt.figure()
plt.plot(nr2)
plt.ylabel(r'$R^2$',fontsize=40)
plt.xlabel('number of neurons',fontsize=40)
    
# %%
### Use known connecome and do pump-probe
### Use pump probe info as target to learn connectome
### (VAE??)

# %% Pump-probe test
N = 50
p = 0.4
g = 1.5
M0 = M.copy()
M0 = np.random.randn(N,N)*g/np.sqrt((1-p)*N)
sparse = np.random.rand(N,N)
M0[sparse>p] = 0
#M0 = JJ*.05

# %%
window = 200
ipt_du = 50
amp = 2.1
stim = np.zeros(window)
stim[int(window/2-ipt_du/2):int(window/2+ipt_du/2)] = amp
pp = []
xx = []  #try subthreshold 

for ii in range(N):
    rt = np.zeros((N,window))
    xt = rt*1
    x = np.random.randn(N)*0.1  #random initial potential
    r = np.tanh(x)
    stim_i = rt*1
    stim_i[ii,:] = stim
    for tt in range(window):    
        x = (1.0-dt)*x/tau + M0 @ (r*dt)/tau + stim_i[:,tt]
        r = np.tanh(x)
        rt[:,tt] = r
        xt[:,tt] = x
    
    pp.append(rt)
    xx.append(xt)
    
# %%
plt.figure()
#plt.plot(np.mean(pp[10],0))
plt.plot(pp[10][11,:],label='x')
plt.plot(xx[10][11,:],label='r')
plt.plot(stim,label='stim')
plt.legend(fontsize=40)

# %%
pp_con = np.zeros((N,N))
pp_con_D = pp_con*1
for ii in range(N):
    for jj in range(N):
        temp = pp[ii][jj,:]
#        measure = temp[int(window/2+ipt_du/2)] #np.max( temp[int(window/2+ipt_du/2):-1] )
        measure = temp[int(window/2+ipt_du/2)] - temp[int(window/2-ipt_du/2)]   #value for now
        pp_con[ii,jj] = measure
        
        temp = xx[ii][jj,:]
        measure = temp[int(window/2+ipt_du/2)] - temp[int(window/2-ipt_du/2)] - amp   #value for now
        pp_con_D[ii,jj] = measure

# %%
J_ = pp_con_D @ np.linalg.pinv(pp_con)
#np.fill_diagonal(J_, -np.diag(J_))
np.fill_diagonal(M0, np.zeros(N)*np.nan)
np.fill_diagonal(J_, np.zeros(N)*np.nan)
M0[sparse>p] = np.nan
J_[sparse>p] = np.nan
plt.figure()
plt.plot((M0), (J_)/N/2, 'k.')
plt.xlabel(r'$J_{true}$', fontsize=30)
plt.ylabel(r'$\hat{J}_{probe}$',fontsize=30)

# %%
plt.figure()
plt.plot(M0, pp_con,'k.')
plt.xlabel(r'$J_{true}$', fontsize=30)
plt.ylabel(r'$\hat{J}_{probe}$',fontsize=30)