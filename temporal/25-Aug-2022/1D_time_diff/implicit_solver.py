#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:18:18 2022

@author: pdavid
"""
from module import *
import numpy as np 

#%%
t_step=10
N_r=100 #spacial points
T=10000
N_t=int(T/t_step) #time points

L=10
Rv=L/100
h=(L-Rv)/(N_r-1)

r=np.linspace(h/2,L-h/2,N_r)


time=np.linspace(0,T,N_t)


Lap=get_1D_lap_operator(r, h, 1)

u=np.zeros(len(r))

q=0.1
B=np.zeros(len(r))
B[0]=-q/(2*np.pi*Rv)

#%%

M=np.identity(N_r)-t_step*Lap

u_imp=np.zeros(len(r))
for i in time:
    if i==0:
        prev=u_imp
    else:
        prev=u_imp[-1,:]
    u_imp=np.vstack((u_imp, np.linalg.solve(M, prev-B*t_step)))
    
plot_dynamic(r,u_imp)