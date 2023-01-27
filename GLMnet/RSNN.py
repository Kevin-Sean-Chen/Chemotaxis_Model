# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 00:36:54 2023

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import matplotlib     
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

#torch.autograd.set_detect_anomaly(True)
#%matplotlib qt5

# %%
class RSNN(nn.Module):# , torch.autograd.Function):
    
    def __init__(self, input_dim, net_dim, output_dim, tau, dt, spk_param, init_std=1.):
        """
        Initialize an RSNN
        parameters:...
        input_dim: int
        net_dim: int
        output_dim: int
        dt: float
        tau: float
        init_std: float, initialization variance for the connectivity matrix
        spk_param: three values for stochastic spiking process
        """
        # Setting some internal variables
        super(RSNN, self).__init__()  # pytorch administration line
        self.input_dim = input_dim
        self.N = net_dim
        self.output_dim = output_dim
        self.dt = dt
        self.tau = tau
        thr, temp, damp = spk_param
        self.thr, self.temp, self.damp = thr, temp, damp  # spiking threshold, temperature, and dampled factor
        global temp_
        global damp_
        temp_, damp_ = temp, damp  # for BPTT usage
        self.spike_op = self.Spike.apply  # snn-torch spike-opteration class
#        self.spike_op = Spike().apply  # if it is not a sub-class
        
        # Defining the parameters of the network
        self.J = nn.Parameter(torch.Tensor(net_dim, net_dim))     # connectivity matrix
        self.B = nn.Parameter(torch.Tensor(net_dim, input_dim))   # input weights
#        self.W = nn.Parameter(torch.Tensor(output_dim, net_dim))  # output matrix
        
#        self.B = torch.Tensor(net_dim,input_dim)  # I/O without training constraint
#        self.W = torch.Tensor(output_dim, net_dim)
        self.W = torch.eye(net_dim) #torch.ones(output_dim, net_dim)/output_dim #
        
        # Initializing the parameters to some random values
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_()
            self.J.normal_(std=init_std / np.sqrt(self.N))
#            self.W.normal_(std=1. / np.sqrt(self.N))
        
    def forward(self, inputs, initial_state=None):
        """
        Run the RSNN with input for one trial
        
        parameters:
        inp: torch tensor of shape (duration x input_dim)
        initial_state: None or torch tensor of shape (input_dim)
        
        returns:
        x_seq: sequence of voltages, torch tensor of shape ((duration+1) x net_size)
        output_seq: torch tensor of shape (duration x output_dim)
        """
#        # initialize arrays
#        T = inputs.shape[0]  # input time series
#        vt = torch.zeros(T+1 , self.N)  # voltage time series
#        zt = torch.zeros(T+1 , self.N)  # spiking time series
#        if initial_state is not None:
#            vt[0] = initial_state
#            zt[0] = self.spikefunction(vt[0])
#        output_seq = torch.zeros(T, self.output_dim)  # output time series
#        
#        # loop through time
#        for t in range(T):
#            vt[t+1,:] = (1 - self.dt/self.tau)*vt[t,:] + self.dt/self.tau*(self.J @ torch.sigmoid(vt[t,:]) + self.B @ inputs[t])
#            zt[t+1,:] = self.spikefunction(vt[t+1])
#            output_seq[t] = self.W @ self.NL(vt[t+1])
        
        # initialize arrays
        n_trials = inputs.shape[0]  # number of trials
        T = inputs.shape[1]  # input time series
        vt = torch.zeros((n_trials, T+1 , self.N))  # voltage time series
        zt = torch.zeros((n_trials, T+1 , self.N))  # spiking time series
        if initial_state is not None:
            vt[0] = initial_state
#            zt[0] = self.spikefunction(vt[0])
#            zt[0] = self.spike_op(self.spkNL(vt[0])*self.dt)  # Poisson
            zt[0] = self.spike_op(self.pre_spk(vt[0]))  # Bernoulli
        output_seq = torch.zeros((n_trials, T, self.output_dim))  # output time series
        
        # loop through time
        for t in range(T):
            ### ODE form
#            vt[:,t+1] = (1 - self.dt/self.tau)*vt[:,t] + self.dt/self.tau*(self.linear_map(zt[:,t]) @ self.J.T + inputs[:,t] @ self.B.T)
##            zt[:,t+1] = self.spikefunction(vt[:,t+1])
#            zt[:,t+1] = self.spike_op(self.pre_spk(vt[:,t+1]))
#            output_seq[:,t] = self.pre_spk(vt[:,t+1]) @ self.W.T
##            output_seq[:,t] = self.NL(vt[:,t+1]) @ self.W.T  
            
            ### GLM form
            vt[:,t+1] = (1 - self.dt/self.tau)*vt[:,t] + self.dt/self.tau*zt[:,t]
            
            # Poisson
#            lamb = self.spkNL(self.synNL(vt[:,t+1]) @ self.J.T + inputs[:,t] @ self.B.T)
#            zt[:,t+1] = self.spike_op(self.linear_map(lamb)*self.dt)  
            # Bernoulli
            lamb = self.linear_map(self.synNL(vt[:,t+1]) @ self.J.T + inputs[:,t] @ self.B.T)
            zt[:,t+1] = self.spike_op(self.pre_spk(lamb))
            
            output_seq[:,t] = (lamb) @ self.W.T
        
#        self.save_for_backward(vt)  # test with this
        
        return vt, zt, output_seq
    
#    @staticmethod
#    def backward(self, grad_output):
#        """
#        Custom backward pass for RNN gradients
#        """
#        v_scaled, = self.saved_tensors
#        dE_dz = grad_output*1
#        dz_dv_scaled = self.pseudo_derivative(v_scaled)
#        dE_dv_scaled = dE_dz * dz_dv_scaled
#        return dE_dv_scaled
    
    def pseudo_derivative(self, v_scaled):
        """
        Bellec's pseudo-derivative for binary spikes
        """
        abs_u = torch.abs(v_scaled)
        return self.damp * torch.clamp(1-abs_u, min=0.0)
    
    def synNL(self, x):
        """
        Synaptic nonliearity
        """
#        nl = torch.sigmoid(x)
        nl = torch.tanh(x)*3
        return nl
    
    def spkNL(self, x):
        """
        Spiking nonliearity
        """
        nl = torch.clamp(x, min=0)
#        nl = torch.sigmoid(x)
        return nl
    
    def linear_map(self,x):
        """
        Passing the same vector but allow gradient computation
        """
        n = x.shape[0]
        I = torch.eye(n)
        return I @ x
    
    def pre_spk(self, mem):
        """
        input raw memebrane mem and rescale, weight, and pass through NL
        this is a function used for the spiking class method
        """
        mem1 = (mem - self.thr)/self.thr
        mem2 = mem1 #self.spkNL(self.temp * mem1)
        return mem2
    
    def spikefunction(self, vt):
        """
        Stochastic spiking through Bernoulli process with rescaled/threshold voltage
        """
        new_v = (vt - self.thr)/self.thr
        new_z = torch.gt(torch.sigmoid(self.temp*new_v) , torch.rand(new_v.shape))
        return new_z
    
    @staticmethod
    class Spike(torch.autograd.Function):
        """
        Spiking opertation sub-class borrowed from snn-torch framework
        with this subclass, directly use Function.apply for spiking process 
        and the gradient step follows
        """
#        def __init__(self, temp, damp):
#            self.temp = temp
#            self.damp = damp
        @staticmethod
        def forward(ctx, v):
#            spk = (v > 0).float() # Heaviside on the forward pass
            spk = torch.gt(torch.sigmoid(temp_*v) , torch.rand(v.shape))
#            spk = torch.poisson(temp_*v) #(torch.sigmoid(temp_*v))
            ctx.save_for_backward(v)  # store the spike for use in the backward pass
#            self.save_for_backward(v)
            return spk
        @staticmethod
        def backward(ctx, grad_output):
            (spk,) = ctx.saved_tensors  # retrieve the spike
#            (spk,) = self.saved_tensors
#            grad = grad_output * spk # scale the gradient by the spike: 1/0
            grad = damp_ * torch.clamp(1-torch.abs(spk), min=0.0)
            return grad
        
def error_function(outputs, targets, masks):
    """
    parameters:
    outputs: torch tensor of shape (n_trials x duration x output_dim)
    targets: torch tensor of shape (n_trials x duration x output_dim)
    mask: torch tensor of shape (n_trials x duration x output_dim)
    
    returns: float
    """
    return torch.sum(masks * (targets - outputs)**2) / outputs.shape[0]

# %% test run
net_size = 60
dt = .1
tau = 1
spk_param = 0.4, 6, 0.5  # threshold, temperature, damp
###  input_dim, net_dim, output_dim, tau, dt, spk_param, init_std=1.
#my_net = RSNN(1, net_size, 1, tau, dt, spk_param, init_std=.1)
my_net = RSNN(1, net_size, net_size, tau, dt, spk_param, init_std=1)

# %% set simulation
# Let us run it with some constant input for a duration T=200 steps:
T = 200
n_trials = 100
inp = torch.zeros((n_trials, T, 1))  # the tensor containing inputs should have a shape (duration x input_dim), even if input_dim is 1.

v_seq, z_seq, output_seq = my_net.forward(inp, initial_state=torch.randn(net_size))  # this effectively runs the simulation

v_seq = v_seq.detach().squeeze().numpy()  # useful line for detaching the sequences from pytorch gradients (we will see that later)
output_seq = output_seq.detach().squeeze().numpy()
z_seq = z_seq.detach().squeeze().numpy()

# looking at the shapes of the obtained data
print(v_seq.shape)
print(output_seq.shape)

# %% test back-prop
n_epochs = 300
lr = 1e-2
batch_size = 32
n_trials = inputs.shape[0]
optimizer = torch.optim.Adam(my_net.parameters(), lr=lr)  # fancy gradient descent algorithm
losses = []

for epoch in range(n_epochs):
    optimizer.zero_grad()
    random_batch_idx = random.sample(range(n_trials), batch_size)
    batch = inputs[random_batch_idx]
    _, _, output = my_net.forward(batch)
    loss = error_function(output, targets[random_batch_idx], masks[random_batch_idx])
    loss.backward()  # with this function, pytorch computes the gradient of the loss with respect to all the parameters
    optimizer.step()  # here it applies a step of gradient descent
    
    losses.append(loss.item())
    print(f'Epoch {epoch}, loss={loss:.3f}')
    loss.detach_()  # 2 lines for pytorch administration
    output.detach_()

# %% test truncation
#rr = 3
#Jij = my_net.J.detach().numpy()
#uu,ss,vv = np.linalg.svd(Jij)
#Jhat = (uu[:,:rr] * ss[:rr]) @ vv[:rr,:]
#my_net.J = weight = nn.Parameter(torch.Tensor(Jhat))

# %%
#inputs_pos, _, _, _ = generate_trials(n_trials, coherences=[+1], T=T)
#inputs_neg, _, _, _ = generate_trials(n_trials, coherences=[-1], T=T)

#inputs_pos, _, _, _ = generate_trials2(n_trials, coherences=[.0], T=T)
#inputs_neg, _, _, _ = generate_trials2(n_trials, coherences=[1.], T=T)

inputs_pos, _, _, _ = generate_activity(n_trials, coherences=[+1], T=T)
inputs_neg, _, _, _ = generate_activity(n_trials, coherences=[-1], T=T)

# run network
v_pos, z_pos, output_pos = my_net.forward(inputs_pos*1)
v_neg, z_neg, output_neg = my_net.forward(inputs_neg*1)

# convert all tensors to numpy
v_pos = v_pos.detach().numpy().squeeze()
z_pos = z_pos.detach().numpy().squeeze()
v_neg = v_neg.detach().numpy().squeeze()
z_neg = z_neg.detach().numpy().squeeze()
output_pos = output_pos.detach().numpy().squeeze()
output_neg = output_neg.detach().numpy().squeeze()

plt.figure()
plt.plot(output_pos.T,'r')
plt.plot(output_neg.T,'b',alpha=0.1)

# %%
import random
def generate_trials(n_trials, coherences=[-2, -1, 1, 2], std=3., T=100):
    """
    Generate a set of trials for the noisy decision making task
    
    parameters:
    n_trials: int
    coherences: list of ints
    std: float, standard deviation of stimulus noise
    T: int, duration of trials
    
    returns (4-tuple):
    inputs: np array of shape (n_trials x T x input_dim)
    targets: np array of shape (n_trials x T x output_dim)
    masks: np array of shape (n_trials x T x 1)
    coherences: list of coherences chosen for each trial
    """
    inputs = std * torch.randn((n_trials, T, 1))
    targets = torch.zeros((n_trials, T, 1))
    mask = torch.zeros((n_trials, T, 1))
    mask[:, T-1] = 1  # set mask to one only at the end
    coh_trials = []
    
    for i in range(n_trials):
        coh = random.choice(coherences)  # choose a coherence
        inputs[i] += coh  # modify input
        targets[i, :] = 1 if coh > 0 else -1 # modify target
        coh_trials.append(coh)
        
    return inputs, targets, mask, coh_trials

def generate_trials2(n_trials, coherences=[0., 1.], T=T):
    """
    Generate deterministic input but stochastic output
    """
    inputs = torch.zeros((n_trials, T, 1))
    targets = torch.ones((n_trials, T, 1))
    mask = torch.zeros((n_trials, T, 1))
    coh_trials = []
    mask[:, T-1] = 1  # set mask to one only at the end
    
    for i in range(n_trials):
        coh = random.choice(coherences)
        inputs[i] += coh
        if coh > np.random.rand():
            targets[i] = -1#torch.rand(200,1)
        coh_trials.append(coh)
        
        
    return inputs, targets, mask, coh_trials

def spiking(x):
    spk = torch.bernoulli(x)
    return spk

def NL(x):
#    nl = torch.sigmoid(x)
    m = nn.ReLU()
    nl = m(x)
    return nl

def Asyn_Irreg(N, lt, freq):
    """
    Generate population spikes as target patterns
    """
    time = torch.arange(0,lt,1)
    latent = 1*torch.sin(time*freq*0.1)
    M = torch.ones(N)
    b = 0
    spk = (NL(M[:,None]*latent-b))
#    m = nn.ReLU()
    spk = torch.zeros(N,lt)
    rt = spk*1
    for tt in range(lt-1):
     spk[:,tt] = spiking(NL(M*latent[tt]-b))
#     spk[:,tt] = spiking(NL(J @ phi(rt[:,tt]) + 0)*dt)  # matched model control
     rt[:,tt+1] = rt[:,tt] + dt/tau*(-rt[:,tt] + spk[:,tt])
    return rt.T # m(latent)[:,None]#
    
def generate_activity(n_trials, coherences=[-1., 1.], T=T, std=0.2):
    """
    Generate target population activity
    """
    inputs = std * torch.randn((n_trials, T, 1)) #torch.zeros((n_trials, T, 1))
    targets = torch.ones((n_trials, T, net_size))
#    targets = torch.ones((n_trials, T, 1))
#    mask = torch.ones((n_trials, T, 1))
    mask = torch.ones((n_trials, T, net_size))
    mask[:,:,int(np.floor(net_size*2/3)):] = torch.zeros((n_trials, T, int(np.floor(net_size*1/3))))
    coh_trials = []
#    mask[:, T-1] = 1  # set mask to one only at the end
    
    for i in range(n_trials):
        coh = random.choice(coherences)
        inputs[i] += 0#coh
        if coh > np.random.rand():
            targets[i] = Asyn_Irreg(net_size, T, 1)#torch.rand(T,net_size)
        else:
            targets[i] = Asyn_Irreg(net_size, T, 1)
        coh_trials.append(coh)
        
    return inputs, targets, mask, coh_trials

#inputs, targets, masks, coh_trials = generate_trials(100,T=T)
#inputs, targets, masks, coh_trials = generate_trials2(100,T=T)
inputs, targets, masks, coh_trials = generate_activity(100,T=T)

### ideas ###
# trained network rank:
# current observation is that the strength of J effects the learned rank (but maybe not for rate RNN!??)
# build up simulator to analyze low-rank or truncated networks
# probablistic output:
# probablistic output requires an objective function different from direct MSE reconstruction
# (might still agree at large training set limit)
# but if we use other summary statistics it might speed up the process (?)
# this can then be compared to semi-hand-tunned method with probablistic latent
