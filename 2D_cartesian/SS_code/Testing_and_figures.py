#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:13:06 2021

@author: pdavid

This is a script meant to test the original formulation and the neighbour formulation.
Therefore this script is situated one level above the two modules
"""


import os
os.chdir('/home/pdavid/Bureau/Code/SS/2D_cartesian/SS_code')
import numpy as np 
import reconst_and_test_module as post
import matplotlib.pyplot as plt
os.chdir('../non_neigh_orig')
import module_2D_coupling as orig 

#0-Set up the sources
validation=True
Rv=0.01
C0=1
K_eff=1/(np.pi*Rv**2)


#1-Set up the domain
D=1
L=5
h_ss=1

validation=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
y_ss=x_ss

#pos_s=np.array([[3.8,3.8],[3.4,3.4], [3.9, 3.6],[1.9,1.9],[1.75,1.75]])+np.array([1.85,1.85])


pos_s=np.array([[2.5,2.5]])
#pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6]])-np.array([0.25,0.25])
#pos_s=np.array([[1.5,4.5]])-0.25

# =============================================================================
# #5- Set up the coupling model 
# n_sources=5
# pos_s=np.empty((n_sources, 2)).astype(int)
# 
# #random generator of n_sources
# #NO BOUNDARY CELLS AND NO SEVERAL SOURCES PER CELL SO FAR
# for i in range(n_sources):
#     pos_s[i,:]=[random.uniform(h_ss,L-h_ss), h_ss+np.around((L-2*h_ss)/5)*i]
# phi_s=np.empty(5)
# =============================================================================
A=orig.A_assembly(len(x_ss), len(y_ss))*D/h_ss**2
#set dirichlet
B,A=orig.set_TPFA_Dirichlet(0,A, h_ss, orig.get_boundary_vector(len(x_ss), len(y_ss)), np.zeros(len(x_ss)*len(y_ss)),D)

t=orig.assemble_SS_2D_FD(pos_s, A, Rv, h_ss,x_ss,y_ss, K_eff, D)
t.pos_arrays()
t.assembly_sol_split_problem()

B_v=np.zeros(len(np.unique(t.s_blocks))*9)
B_q=np.ones(len(t.s_blocks))
B=np.concatenate((B,B_v,B_q))

phi=np.linalg.solve(t.M, B)

phi_FV, phi_v, phi_q=post.separate_unk(t, phi)
phi_mat=phi_FV.reshape(len(x_ss), len(y_ss))


Up=np.hstack((t.A, t.b, t.c))

grads=post.reconstruction_gradients_manual(phi, t,orig.get_boundary_vector(len(t.x), len(t.y))) 

o=post.reconst_microscopic(t, phi)    
rec=o.reconstruct(20,L)
plt.figure(figsize=(10,10), dpi=180)
plt.imshow(rec, extent=[0,L,0,L], origin='lower')
plt.colorbar()
plt.show()
    
if validation:
    sol, xlen2, ylen2,q_array,a, Y, s_b,x_v, y_v=orig.get_validation(16, t, pos_s, B_q, D, K_eff, Rv,L)

plt.figure(figsize=(10,10), dpi=180)
plt.imshow(sol.reshape(ylen2, xlen2), extent=[0,L,0,L], origin='lower'); plt.colorbar()
c=0
for i in pos_s:
    plt.scatter(i[0], i[1], marker='x', label="source={}".format(c))
    c+=1
plt.legend()
plt.show()



