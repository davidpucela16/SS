#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 17:30:05 2021

@author: pdavid
"""


import os
os.chdir('/home/pdavid/Bureau/Code/SS/2D_cartesian/SS_code')
import numpy as np 
import reconst_and_test_module as post
import matplotlib.pyplot as plt
os.chdir('/home/pdavid/Bureau/Code/SS/2D_cartesian/SS_code/FV_subgrid')
# insert at 1, 0 is the script path (or '' in REPL)
from module_2D_FV import * 

#0-Set up the sources
validation=True
Rv=0.01
C0=1
K_eff=1/(np.pi*Rv**2)


#1-Set up the domain
D=1
L=6
h_ss=1

validation=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
y_ss=x_ss


#pos_s=np.array([[3.8,3.8],[3.4,3.4], [3.9, 3.6],[1.9,1.9],[1.75,1.75]])+np.array([1.85,1.85])


#pos_s=np.array([[2.5,2.5]])
#pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6]])-np.array([0.25,0.25])
#pos_s=np.array([[1.5,4.5]])-0.25

#pos_s=np.array([[1.5,1.5],[1.5,2.5],[2.5,1.5]])+1

pos_s=np.zeros([0,2], dtype=float)
for i in range(10):
    pos_s=np.concatenate((pos_s, np.array([np.random.random(2)*L/2])), axis=0)
    

n=full_ss(pos_s, 0.01, 0.05, K_eff, D, L)
phi_q=np.ones(len(pos_s))

phi_mat=n.solve_problem(phi_q)

plt.imshow(phi_mat, origin='lower', extent=[0,L, 0, L]); plt.colorbar()
plt.title("regular term")
plt.xlabel("x")
plt.ylabel("y")
plt.show()



i=coord_to_pos(n.x, n.y, pos_s[1])//len(n.x)
plt.plot(n.x,phi_mat[:,49])
plt.title("regular term along a line containing the source")
plt.xlabel("x")
plt.ylabel("concentration (regular)")
plt.show()



k=n.reconstruct(n.phi,phi_q)

# =============================================================================
# k_inf=n.reconstruct_inf(n.phi[n.s_blocks],phi_q)
# ny=n.s_blocks//len(n.x)
# plt.figure()
# plt.plot(n.x,k[ny][0], label="sol_split")
# plt.plot(n.x,k_inf[ny][0], label="Peaceman")
# plt.xlabel("x")
# plt.ylabel("concentration")
# plt.title("Peaceman vs sol_split")
# =============================================================================

plt.figure()
plt.imshow(k, origin="lower"); plt.colorbar()






#experimetn
sa=7

v_s=phi_mat.copy()
for i in range(len(n.x)):
    for j in range(len(n.y)):
        g=Green(pos_s[sa],np.array([n.x[i],n.y[j]]), Rv)
        v_s[j,i]+=g
pos=np.array([np.argmin(np.abs(pos_s[sa,1]-n.y)),np.argmin(np.abs(pos_s[sa,0]-n.x))])

v_neigh=0
for i in np.array([1,-1]):
    for j in np.array((1,-1)):
        v_neigh+=v_s[pos+np.array([j,i])]
v_s[pos]=v_neigh/4+n.phi_q[sa]*n.h**2

plt.imshow(v_s, origin="lower")


ratio=10
ls=np.array([len(n.y)//ratio+1, len(n.x)//ratio+1], dtype=int)
positions=np.arange(ratio//2, len(n.x), ratio)

lapl=np.dot(A_assembly_Dirich(ls[1], ls[0]),np.ndarray.flatten(v_s[positions,:][:,positions]))
plt.imshow(lapl.reshape(ls)[1:-1,1:-1], origin="lower")
plt.colorbar()