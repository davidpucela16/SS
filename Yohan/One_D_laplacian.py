#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:59:02 2022

@author: pdavid

I'm trying to get the delta formulation to work'
"""

import numpy as np 
import matplotlib.pyplot as plt
class one_D_lapl():
    
    def __init__(self,L,N):
        h=L/N
        r=np.linspace(h/2, L-h/2,N)
        self.r=r
        self.h=h
        self.N=N
    def laplacian(self):
        r=self.r
        h=self.h
        N=self.N
        lap=np.zeros((N,N))
        for i in range(N):
            ri=r[i]
            
            if i!=0:
                lap[i,i-1]=(ri-h/2)/(ri*h**2)
                lap[i,i]-=1/h**2
            
            if i!=N-1:
                lap[i,i+1]=(ri+h/2)/(ri*h**2)
                lap[i,i]-=1/h**2
        self.lap=lap
        return(lap)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


L=10
N=1000

R=1
Lj=8.0001


q0=5

a=one_D_lapl(L,N)
lap=a.laplacian()


lap[0,:]=0
lap[-1,:]=0
lap[0,0]=1
lap[-1,-1]=1



q=np.zeros(N)
q[0]=1
q[find_nearest(a.r,R)]=q0/(4*a.h*R*np.pi)


plt.plot(a.r, np.linalg.solve(lap,q))
plt.show()

P=np.zeros(N)
idx=find_nearest(a.r,R)
P[:idx]=2
P[idx:]=2+q0/(2*np.pi)*np.log(R/a.r[idx:])
P[find_nearest(a.r,Lj):]=0




plt.plot(P)
