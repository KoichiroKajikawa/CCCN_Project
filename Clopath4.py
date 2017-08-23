#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:56:56 2017

@author: koichirokajikawa
"""
import numpy as np
import matplotlib.pyplot as plt

# Activation function
def activation(x,method):
    if method=='exp':           # Exponential
        y=np.exp(x)
    elif method=='sigmoid0':     # Sigmoid 
        y=1/(1+np.exp(-(x)))
    elif method=='sigmoid':
        # Can put several parameters as follows.
        y=1/(1+np.exp(-(x-5)))
    elif method=='ReLU':        # ReLU
        y=np.maximum(0,x)        
    return y
    
def calc_clustering_index(W_mat,num_groups,groupsize):
    # calculate a scalar as a clustering measure for a given weight matrix. This assumes an equal number of neurons in each cluster.
    mask = np.zeros(W_mat.shape) #create a mask of zeros and ones to index neurons in vs. neurons out of groups
    for i in range(num_groups):
        mask[i*groupsize:i*groupsize+groupsize, i*groupsize:i*groupsize+groupsize] = 1;
    clustermean = np.mean(mask * W_mat) #get mean firing rate of all neurons within groups
    allmean  = np.mean(W_mat) #get mean firing rate of all neurons total
    return np.divide(clustermean, allmean)

## Parameters
# 1. Network structure
N=100       # number of neurons
C=5         # number of clusters
N_c=N//C    # number of neurons in a cluster
act_fn='sigmoid'
# 2. time constants: tau_r ≤ tau_t ≤ tau_w
[tau_r,tau_t,tau_w] = [5, 50, 500]
[w_max, w_min]=[1,0]
# 3. Coefficients 
#[w_c,s_c,ip_c]=[0.25,100,10]
#[w_c,s_c,ip_c]=[0.02,1.2,2]
[w_c,s_c,ip_c,th_c]=[0.2,1/N,10,1]

## Setup arrays
T_sec  = 5             # total time to simulate (sec)
T      = T_sec*10**3   # total time to simulate (msec)        
dt     = 0.125         # simulation time step (msec)
time   = np.arange(0, T+dt, dt)    # time array
r      = np.zeros((N,len(time)))
r[:,0] = np.random.rand(N)
W      = np.zeros((N,N,len(time)))
W[:,:,0]= np.random.rand(N,N)*w_c
#W[:,:,0]= np.ones((N,N))*0.5  # This leads to all 1 W. 
theta  = np.zeros((N,len(time)))
#theta[:,0]=np.random.rand(N)
#theta[:,0]=np.ones(N,)*np.random.rand(1)
theta[:,0]=activation(np.dot(W[:,:,0],r[:,0])*s_c,act_fn)**2
#theta[:,0]=np.ones(N,)

## Inputs
ip_pattern='simple2'
ip_time=20      # Duration of input ~ tau_r*10 (?)tau_theta
# Define input
if ip_pattern=='simple':
    # 1. Simple correlated input with negative inputs 
    ip_group=np.random.randint(0,C,int(T/ip_time))
    I=np.ones((N, int(T/dt)))*(-10)    # Work for 'sigmoid0'
    for k in range(len(ip_group)):
        I[N_c*ip_group[k]:N_c*(ip_group[k]+1),k*int(ip_time/dt):(k+1)*int(ip_time/dt)]=1*ip_c
elif ip_pattern=='simple2':
    # 1'. Simple correlated input w/o negative inputs
    ip_group=np.random.randint(0,C,int(T/ip_time))
    I=np.zeros((N, int(T/dt)))         # Work for 'sigmoid' with s_c=1/N
    for k in range(len(ip_group)):
        I[N_c*ip_group[k]:N_c*(ip_group[k]+1),k*int(ip_time/dt):(k+1)*int(ip_time/dt)]=1*ip_c
elif ip_pattern=='tuning':
    # 2. Tuning curve
    I = np.zeros((N, int(T/dt)))
    angle=np.random.rand(1,int(T/ip_time))*180
    angle_proj=np.empty((N,int(T/ip_time)))
    for k in range(C):
        angle_proj[k*N_c:(k+1)*N_c,:]=np.cos(angle-k*np.pi/C)*ip_c    
    for k in range(int(T/ip_time)):
        I[:,k*int(ip_time/dt):(k+1)*int(ip_time/dt)] = np.tile(angle_proj[:,k].reshape(-1,1),(1,int(ip_time/dt)))
elif ip_pattern=='sequence':
    # Sequential inputs
    seq_length=4
    seq=np.zeros((N_c,seq_length))
    for k in range(seq_length):
        seq[k*int(N_c/seq_length):(k+1)*int(N_c/seq_length),k]=1
    seq_block=np.tile(seq,(1,int(ip_time/dt/seq_length)))        
    ip_group=np.random.randint(0,C,int(T/ip_time))
    I=np.ones((N, int(T/dt)))*(-10)    
    for k in range(len(ip_group)):
        I[N_c*ip_group[k]:N_c*(ip_group[k]+1),k*int(ip_time/dt):(k+1)*int(ip_time/dt)]=seq_block*ip_c
elif ip_pattern=='sequence2':
    # Sequential inputs through whole network
    seq_duration=20     # msec
    seq_block=np.zeros((N,int(seq_duration/dt)))
#    seq_block=np.ones((N,int(seq_duration/dt)))*(-10)
    for k in range(C):
        seq_block[k*N_c:(k+1)*N_c,k*int(seq_duration/dt/C):(k+1)*int(seq_duration/dt/C)]=1*ip_c
    I=np.tile(seq_block,(1,int(T/seq_duration)))
            
# Learning
for t in range(len(time)-1):   
    # Postsynaptic activity 
    v=activation(np.dot(W[:,:,t],r[:,t])*s_c+I[:,t],act_fn)

    # rate update
    r[:,t+1]=r[:,t]+(-r[:,t]+v)*dt/tau_r

    # weight update
    r_post=r[:,t].reshape(-1,1)    
    r_pre=r[:,t].reshape(1,-1)
    
    W[:,:,t+1]=W[:,:,t]+(r_post*(r_post-theta[:,t].reshape(-1,1))*r_pre)*dt/tau_w
#    W[:,:,t+1]=np.minimum(W[:,:,t+1],w_max*np.ones((N,N)))
    W[:,:,t+1]=np.maximum(W[:,:,t+1],w_min*np.ones((N,N)))
     
    # Set diagonal entries as 0
    np.fill_diagonal(W[:,:,t],0)
    
    # theta update
    theta[:,t+1]=theta[:,t]+((r[:,t]**2)/th_c-theta[:,t])*dt/tau_t

CI1=np.zeros(len(time)//1000)
for t in range(len(CI1)):
    CI1[t]=calc_clustering_index(W[:,:,t*1000],C,N_c)
    
l=-1
plt.imshow(W[:,:,l])
plt.colorbar()
plt.show()

print(W[:,:,l].max())

plt.plot(W[0,15,::100])
plt.plot(W[0,25,::100])

plt.plot(theta[0,::100])
plt.plot(r[0,::100])

#plt.plot(time[125::1000],CI1)
#plt.plot(CI)

S_pre=1-np.mean(I[:,0])/np.max(I[:,0])
O=np.dot(W[:,:,-1],I[:,0])
S_post=1-np.mean(O)/np.max(O)

