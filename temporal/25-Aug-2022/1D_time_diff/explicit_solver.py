from module import *
import numpy as np 

#%%
t_step=0.005
N_r=100 #spacial points
T=120000
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
B[0]=q/(2*np.pi*Rv)
#%% - Set up the explicit resolution of the problem 

u_exp=np.zeros(len(r))
c=0
for i in time[1:]:
    if c==0:
        prev=u_exp
    else:
        prev=u_exp[-1,:]
    if c%10000==0:
        u_exp=np.vstack((u_exp,prev+t_step*(np.dot(Lap,prev))+B))
    
    c+=1


plot_dynamic(r,u_exp)
