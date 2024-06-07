# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 23:38:04 2022

@author: kevin
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# %%
###############################################################################
# %% Effective model
###############################################################################

# %%
#functions
def environment(xx,yy,M=50):
    """
    a simple Gaussian diffusion 2-D environement
    """
#    M = 20   #max concentration
    sig2 = 20  #width of Gaussian
    target = np.array([30,30])   #target position
    NaCl = M*np.exp(-((xx-target[0])**2+(yy-target[1])**2)/2/sig2**2) ### Gaussian bump
#    NaCl = max(0,M*((xx-target[0])**2+(yy-target[1])**2)**0.5) ### linear sharp distance
    return NaCl+np.random.randn()*0.1

def steering(vv,alpha_s,dcp,K):
    """
    slow continuous steering angle change
    """
    dth_s = alpha_s*dcp*np.abs(vv) + np.random.randn()*K
    return dth_s

def Pirouette(vv,alpha_p,lamb0):
    """
    Frequency of the random turning process
    """
    lambda_p = lamb0 + alpha_p*vv
    lambda_p = min(1,max(lambda_p,0))
    th = -0
#    lambda_p = 0.023/(1+np.exp(alpha_p*(vv-th))) + lamb0
    return lambda_p

def turn_angle(vv,alpha_g,gamma0):
    """
    Biased turning angle after Pirouette
    """
    gammaB = gamma0 + alpha_g*vv
    gammaA = np.max(1 - gammaB,0)
    sigma = np.pi/12
    dth_b = gammaA*(np.random.rand()*2*np.pi-np.pi) + gammaB*(sigma*np.random.randn()-np.pi) #opposite direction
    #f_theta = gammaA/(np.pi*2) + gammaB/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(th-np.pi)/(2*sigma**2))
    return dth_b

def dc_measure(dxy,xx,yy):
    """
    perpendicular concentration measure
    """
    perp_dir = np.array([-dxy[1], dxy[0]])  #perpdendicular direction
    perp_dir = perp_dir/np.linalg.norm(perp_dir)*1 #unit norm vector
    perp_dC = environment(xx+perp_dir[0], yy+perp_dir[1]) - environment(xx-perp_dir[0], yy-perp_dir[1])
    return perp_dC

def ang2dis(x,y,th):
    e1 = np.array([1,0])
    vec = np.array([x,y])
    theta = math.acos(np.clip(np.dot(vec,e1)/np.linalg.norm(vec)/np.linalg.norm(e1), -1, 1)) #current orienation relative to (1,0)
    v = vv + vs*np.random.randn()
    dd = np.array([v*np.sin(th), v*np.cos(th)])  #displacement
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,s), (-s, c)))  #rotation matrix, changing coordinates
    dxy = np.dot(R,dd)
    return dxy

#simple switch
def sigmoid(x,w,t):
    ss = 1/(1+np.exp(-w*x-t))
    return ss

# %% 
#with behavior
dt = 0.1
T = 300
lt = int(T/dt)
alpha = 4
beta = 1.5
gamma = .2
Ft,St = np.zeros(lt),np.zeros(lt) #fast and slow part of AWC
tau = 1
target = np.array([30,30])
Vs = np.zeros((2,lt))
Cs = np.zeros(lt)
ths = np.zeros(lt)
prt = np.zeros(lt)
XY = np.random.randn(2,lt)
proj = np.array([.5,.2])*1
lamb0 = 0.01  #0.05
gamma0 = 0.5
alpha_p, alpha_s, alpha_g = -.1, -.02, 0.0001
dxy = np.random.randn(2)
vv,vs = 0.55,0.05
K = np.pi/20 #12
J = np.array([[.1,-5.],[-5,.1]])*1
env_noise = 1
for tt in range(lt-1):
    ### sensor dynamics
    Ft[tt+1] = Ft[tt] + dt*(alpha*Cs[tt] - beta*Ft[tt])  # fast dynamics
    St[tt+1] = St[tt] + dt*gamma*(Ft[tt] - St[tt])  # slow dynamics
    Is = Ft[tt] - St[tt]
    ###neural dynamics
    Vs[:,tt+1] = Vs[:,tt] + dt/tau*( -Vs[:,tt] + proj*Is + J @ sigmoid(Vs[:,tt],1,0) ) \
    + np.random.randn(2)*np.sqrt(dt)*1
    ###behavior
    lambda_p = Pirouette(Vs[0,tt+1],alpha_p,lamb0)  #Pirouette #Cs[tt]
    if lambda_p>=np.random.rand():
        dth = turn_angle(Vs[0,tt+1],alpha_g,gamma0)  #bias
#        print('k')
    else:
        dcp = dc_measure(dxy,XY[0,tt],XY[1,tt])
        dth = steering(Vs[1,tt+1],alpha_s,dcp,K)  #weathervaning #Vs[1,tt+1]
    ths[tt+1] = ths[tt]+dth
    ###environment
    #dxy = np.squeeze(np.array([np.cos(dth), np.sin(dth)]))
    dxy = ang2dis(XY[0,tt],XY[1,tt],ths[tt+1])
    XY[:,tt+1] = XY[:,tt] + dxy*dt
    Cs[tt+1] = environment(XY[0,tt+1],XY[1,tt+1],50) + np.random.randn()*env_noise
    prt[tt+1] = dth
    
# 
plt.figure()
plt.plot(Cs)
plt.figure()
y, x = np.meshgrid(np.linspace(-10, 50, 60), np.linspace(-10, 50, 60))
plt.imshow(environment(x,y),origin='lower',extent = [-10,50,-10,50])
plt.plot(XY[0,:],XY[1,:],'blue')
plt.scatter(XY[0,0],XY[1,0], color='green', marker='o');plt.scatter(XY[0,-1],XY[1,-1], color='red', marker='o')
plt.figure()
plt.plot(Vs.T)

# %%
###############################################################################
# %% Effective model
###############################################################################
envs = np.array([20, 40, 60, 80])  # slope of the chemical gradient
params = np.array([[-.1, -.00, 0.0001],
                   [-.0, -.02, 0.0001],
                   [-.1, -.02, 0.0001],
                   [-.1, -.02, 0.0001]]) #BRW, WV, both, and switching...
#params = np.array([[-.1, -.02, 0.0001],
#                   [-.1, -.02, 0.0001]]) #BRW, WV, both, and switching...
repeats = 50  # number of trials for statistics  #50
eps = 5 # distance from the point source
CIs = np.zeros((len(envs), params.shape[0]))  # environments by parameters

for ee in range(len(envs)):  #environment loop
    M = envs[ee]
    for pp in range(params.shape[0]):  # parameter loop
        alpha_p, alpha_s, alpha_g = params[pp,:]
        
        ### run chemotaxis trials
        for rr in range(repeats):
            Ft,St = 0,0
            Vs = np.zeros(2)
            Cs = np.random.randn() #np.zeros(lt)
            ths = 0#np.zeros(lt)
            XY = np.random.randn(2)#,lt)
            dxy = np.random.randn(2)
            if pp==3:
                J = np.array([[.1,-5.],[-5,.1]])*2  #add interactions
            else:
                J = np.array([[.1,-5.],[-5,.1]])*0
            
            ### chemotaxis dynamics
            for tt in range(lt-1):
                ### sensor dynamics
                Ft = Ft + dt*(alpha*Cs - beta*Ft)  # fast dynamics
                St = St + dt*gamma*(Ft - St)  # slow dynamics
                Is = Ft - St
                ###neural dynamics
                Vs = Vs + dt/tau*( -Vs + proj*Is + J @ sigmoid(Vs,1,0) ) \
                + np.random.randn(2)*np.sqrt(dt)*1
                ###behavior
                lambda_p = Pirouette(Vs[0],alpha_p,lamb0)  #Pirouette #Cs[tt]
                if lambda_p>=np.random.rand():
                    dth = turn_angle(Vs[0],alpha_g,gamma0)  #bias
                else:
                    dcp = dc_measure(dxy,XY[0],XY[1])
                    dth = steering(Vs[1],alpha_s,dcp,K)  #weathervaning #Vs[1,tt+1]
                ths = ths + dth
                ###environment
                dxy = ang2dis(XY[0],XY[1],ths)
                XY = XY + dxy*dt
                Cs = environment(XY[0],XY[1], M) + np.random.randn()*env_noise
        
            ### record CI
            endpos = XY.copy()
            dist = np.linalg.norm(endpos - target)  # final distance
            if dist<eps:
                CIs[ee,pp] += 1/repeats
                        
        print(ee,pp)
        
# %%
plt.figure()
plt.plot(envs[:],CIs[:,:],'-o')
plt.xlabel('gradient')
plt.ylabel('CI')
lab = np.array(["BRW","WV","BRW+WV","HMM"])
plt.legend(labels=lab)
