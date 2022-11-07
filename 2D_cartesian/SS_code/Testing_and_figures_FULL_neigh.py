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
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Neighbours')
from module_2D_coupling_neigh import * 

#0-Set up the sources
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

#pos_s=np.array([[2.5,1.25],[2.5,1.75],[1.5,1.25],[1.5,1.75]])

pos_s=np.array([[1.5,1.5],[1.5,2.5],[2.5,1.5]])+1

#pos_s=np.array([[1.5,1.5],[2.5,1.5]])
#pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6],[2.5,2.5]])-np.array([0.25,0.25])
#pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6]])-np.array([0.25,0.25])
#pos_s=np.array([[2,2]])+0.25



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
A=A_assembly(len(x_ss), len(y_ss))*D/h_ss**2
#set dirichlet
B=np.zeros(len(x_ss)*len(y_ss))
B,A=set_TPFA_Dirichlet(0,A, h_ss, get_boundary_vector(len(x_ss), len(y_ss)), np.zeros(len(x_ss)*len(y_ss)),D)


# =============================================================================
# The new system is validated with the previous one!!
# to=orig.assemble_SS_2D_FD(pos_s, A.copy(), Rv, h_ss,x_ss,y_ss, K_eff, D)
# to.pos_arrays()
# to.assembly_sol_split_problem()
# 
# Up_orig=np.hstack((to.A, to.b, to.c))
# Mid_orig=np.hstack((to.d, to.e, to.f))
# Down_orig=np.hstack((to.g, to.H, to.I))
# =============================================================================

t=assemble_SS_2D_FD(pos_s, A, Rv, h_ss,x_ss,y_ss, K_eff, D)
t.pos_arrays()
t.assembly_sol_split_problem()

B_v=np.zeros(len(np.unique(t.s_blocks))*9)
B_q=np.ones(len(t.s_blocks))
B_q[0]=0.5
B=np.concatenate((B,B_v,B_q))

phi=np.linalg.solve(t.M, B)

phi_FV, phi_v, phi_q=post.separate_unk(t, phi)
phi_mat=phi_FV.reshape(len(x_ss), len(y_ss))



o=post.reconst_microscopic(t, phi)    
rec=o.reconstruct(20,L)
plt.imshow(rec, extent=[0,L,0,L],origin='lower')
plt.colorbar()
c=0
for i in pos_s:
    plt.scatter(i[0], i[1], marker='x', label="source={}".format(c))
    c+=1
plt.legend()
plt.show()
    
if validation:
    sol, xlen2, ylen2,q_array,a, Y, s_b,x_v, y_v=get_validation(20, t, pos_s, B_q, D, K_eff, Rv,L)
    plt.figure(figsize=(10,10), dpi=180)
    plt.imshow(sol.reshape(ylen2, xlen2), extent=[0,L,0,L], origin='lower'); plt.colorbar()
    c=0
    for i in pos_s:
        plt.scatter(i[0], i[1], marker='x', label="source={}".format(c))
        c+=1
    plt.legend()
    plt.show()


#Study of the Fluxes


flux=post.reconst_real_fluxes_posteriori(phi, t) 
f_s=flux[t.s_blocks]

fluxes_matrix=np.hstack((t.b, t.c))

row=np.argmin(np.abs(o.y-pos_s[0,1]))
plt.plot(rec[row,:])

row=np.argmin(np.abs(y_v-pos_s[0,1]))
plt.plot(sol.reshape(ylen2, xlen2)[row,:])