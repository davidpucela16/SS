#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 19:06:19 2021

@author: pdavid
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

hx=hy=0.05	 
dom=10	
x=np.arange(0,dom,hx)
y=np.arange(0,dom,hy)

init=np.array([2,5], dtype=float)
end=np.array([7,5], dtype=float)
tau=(end-init)/np.linalg.norm(end-init)

plt.plot(np.array([init[0], end[0]]), np.array([init[1], end[1]]))
plt.ylim([0,10])
plt.xlim([0,10])

def sing_term(init, end, x, Rv):
    tau=(end-init)/np.linalg.norm(end-init)
    L=np.linalg.norm(end-init)
    a=x-init
    b=x-end
    s=np.sum(a*tau)
    d=np.linalg.norm(a-tau*np.sum(a*tau))
    
    G_term=np.sqrt(L**2/4+Rv**2)
    
    if d<Rv:
        if s<0 or s>L:
            C_0=s-L if s>L else -s
            G=np.log((L+C_0)/C_0)
        else:
            G=np.log((G_term+L/2)/(G_term-L/2))
        
    else:
        rb=np.linalg.norm(b)
        ra=np.linalg.norm(a)
        G=np.log((rb+L-np.dot(a,tau))/(ra-np.dot(a,tau)))
    return(G)

def sing_term2(init, end, x, Rv):
    tau=(end-init)/np.linalg.norm(end-init)
    L=np.linalg.norm(end-init)
    a=x-init
    b=x-end
    s=np.sum(a*tau)
    d=np.linalg.norm(a-tau*np.sum(a*tau))
    
    
    rb=np.linalg.norm(b)
    ra=np.linalg.norm(a)
    G=np.log((rb+L-np.dot(a,tau))/(ra-np.dot(a,tau)))
    return(G)

st_jit=njit()(sing_term)
    
def Green(init, end, x, y, Rv):
    L=np.linalg.norm(init-end)
    tau=(end-init)/L
    print(tau)
    A=np.zeros([len(x), len(y)])
    
    c=0
    for i in x:
        k=0
        for j in y:
            p=np.array([i,j])
            A[k,c]=sing_term(init, end,p, Rv)/(np.pi*4)
            k+=1
        c+=1
    return(A)


def rec_field_points(x,y,z,R,init,tau,h):
    c=0
    A=np.zeros([len(x), len(y)])
    for i in x:
        k=0
        for j in y:
            p=np.array([i,j])
            for t in z:
                #if np.linalg.norm(t-p)>R and np.abs(np.dot((p-init),tau))<L: 
                if np.linalg.norm(t-p)>R:
                    A[k,c]+=(np.linalg.norm(t-p)*(np.pi*4))**-1
                else:
                    A[k,c]+=0
                
            k+=1
        c+=1
    return(A*h)
st_jit=njit()(rec_field_points)

class Greens_3D_comparison():
    def __init__(self, init, end, Rv, N):
        self.end=end
        self.init=init
        self.R=Rv
        L=np.linalg.norm(end-init)
        self.L=L
        self.h=L/N
        self.s=np.linspace(self.h/2,L-self.h/2,N)
        
        self.tau=(end-init)/L
        # z is the variable that contains the coordinates of every discrete vessel point
        self.z=np.vstack((init[0]+self.s*self.tau[0], init[1]+self.s*tau[1])).T
        
    def rec_field_points(self, x,y):
        to_ret=rec_field_points(x,y,self.z,self.R,self.init,self.tau,self.h)
        return(to_ret)
    
    def rec_field_line(self,x,y):
        init=self.init
        end=self.end        
        L=self.L
        tau=(end-init)/L
        A=np.zeros([len(x), len(y)])
        c=0
        for i in x:
            k=0
            for j in y:
                p=np.array([i,j])
                A[k,c]=sing_term2(init, end,p, Rv)/(np.pi*4)
                k+=1
            c+=1
        return(A)

#%%
Rv=0.001
a=Greens_3D_comparison(init, end, Rv, 100)


Green_log=a.rec_field_line(x,y)
Green_3D=a.rec_field_points(x,y)

#%%
plt.imshow(Green_3D, origin='lower')
plt.title("Numerical integration")
plt.colorbar()

#%%

plt.imshow(Green_log, origin='lower')
plt.title("Line source integral")
plt.colorbar()

#%%
plt.imshow(Green_log-Green_3D, origin='lower', vmax=0.05)
plt.colorbar()
#%%



X,Y=np.meshgrid(x,y)

ls=np.arange(0,1,0.08)
A=Green(init, end, x, y,0.1)
plt.figure()
plt.plot(np.array([init[0], end[0]]), np.array([init[1], end[1]]))
plt.contourf(X,Y,A, levels=ls)
plt.colorbar()

#%%
L=10
init=np.array([0,0])
end=np.array([0,L])
N=100
Rv=0.5
x=np.linspace(0,L,N)
s=np.array([np.zeros(N)+Rv,x]).T

h=L/N

SL=np.zeros(N)
c=0
for i in s:
    SL[c]=sing_term2(init, end, s[c], Rv)*2*np.pi*Rv*h
    c+=1