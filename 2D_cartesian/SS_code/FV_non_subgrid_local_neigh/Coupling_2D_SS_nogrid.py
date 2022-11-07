#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 19:33:13 2021

@author: pdavid
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:52:20 2021

@author: pdavid
"""

import numpy as np 
import matplotlib.pyplot as plt
from module_2D_coupling_FV_nogrid import * 
import reconst_and_test_module as post
import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg

#0-Set up the sources
#1-Set up the domain
D=1
L=6
h_ss=0.5
#Rv=np.exp(-2*np.pi)*h_ss
Rv=0.01
C0=1
K_eff=1/(np.pi*Rv**2)

validation=False
real_Dirich=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
y_ss=x_ss
directness=2
#pos_s=np.array([[x_ss[2], y_ss[2]],[x_ss[4], y_ss[4]]])
#pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6],[2,2]])-np.array([0.25,0.25])
#pos_s/=2
#pos_s=np.array([[1.25,1.25],[1.25,1.75], [1.75,1.75],[1.75,1.25]])
#pos_s=np.array([[4.3,4.3],[4.3,5.5], [3.5,4.5],[3.5,3.5]])
S=15
pos_s=np.array([[3.37906653, 2.25662776],
       [2.81452878, 2.38701789],
       [2.10068342, 3.15304824],
       [2.89109378, 2.90483852],
       [2.39540118, 2.81519292],
       [3.22824767, 3.57820218],
       [2.52451267, 3.50308419],
       [2.77560641, 2.82470712],
       [2.27383187, 3.56744184],
       [3.8928436 , 2.31976285],
       [3.61648574, 2.85538718],
       [3.49673019, 2.9824555 ],
       [2.66398838, 3.99574789],
       [2.98675591, 3.60749549],
       [2.68653765, 2.26706154]])

#pos_s=np.random.random((S,2))*2+2
#pos_s=np.append(pos_s, np.array([[4.5,3.5]]), axis=0)
#pos_s=np.array([[1.1,1.6],[1.7,1.1],[1.3,1.7],[2.5,1.5]])+1

#pos_s=np.array([[2.5,2.5],[2.5,3.5]])
#pos_s=np.array([[2.5,2.5]])

#pos_s=np.array([[1.25,1.25]])+2.25
#pos_s=np.array([[1.25,1.25],[1.25,1.75]])
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

if real_Dirich:
    A=A_assembly(len(x_ss), len(y_ss))*D/h_ss**2
    #set dirichlet
    B,A=set_TPFA_Dirichlet(0,A, h_ss, get_boundary_vector(len(x_ss), len(y_ss)), np.zeros(len(x_ss)*len(y_ss)),D)
else:
    A=A_assembly_Dirich(len(x_ss), len(y_ss))*D/h_ss**2
    B=np.zeros(A.shape[0])
    
    
t=assemble_SS_2D_FD(pos_s, A, Rv, h_ss,x_ss,y_ss, K_eff, D, directness)
t.pos_arrays()
t.initialize_matrices()


t.assembly_sol_split_problem()
B_v=np.zeros(len(t.uni_s_blocks))
B_q=np.ones(len(t.s_blocks))
B=np.concatenate((B,B_v,B_q))


phi=np.linalg.solve(t.M, B)
phi_FV, phi_v, phi_q=post.separate_unk(t, phi)
phi_mat=phi_FV.reshape(len(x_ss), len(y_ss))

plt.imshow(phi_mat, origin="lower"); plt.colorbar()
plt.show()
Up=np.hstack((t.A, t.B_matrix, t.C1_matrix+t.C2_matrix))



o=post.reconstruct_coupling(phi, 1, t, directness)
rec=o.reconstruction(20)
grads_inner=np.delete(o.grads_FV, t.boundary, axis=0)

plt.imshow(rec, extent=[0,L,0,L], origin='lower')
plt.colorbar()
plt.show()
    
ratio=5

sol, xlen2, ylen2,q_array,a, Y, s_b,x_v, y_v=get_validation(ratio, t, pos_s, B_q, D, K_eff, Rv,L)
plt.imshow(sol.reshape(ylen2, xlen2), origin='lower'); plt.colorbar()
plt.show()



a=full_ss(pos_s, Rv, h_ss/ratio, K_eff, D,L)
SS=a.solve_problem(B_q)
SS=a.reconstruct(a.v, a.phi_q)
plt.show()
plt.imshow(SS, origin="lower", extent=[0,L,0,L]); plt.colorbar()
plt.title("reconstruction Full_SS")
plt.show()
#reconstruct the full solution splitting
a.uni_s_blocks=np.array([])
a.s_blocks=np.array([])
a.FV_DoF=np.arange(len(a.v))

grads_v=post.reconstruction_gradients_manual(a.v, a, a.boundary)
v_rec=post.reconstruct_from_gradients(a.v, grads_v, 5, a.x, a.y, a.h)

# =============================================================================
# plt.plot(a.x, SS[:,3])
# plt.plot(x_ss, phi_mat[:,3])
# plt.show()
# =============================================================================


plt.figure()
plt.plot(phi_q, label="coupling", marker='*')
plt.plot(a.phi_q, label="Full_SS",marker='*')
plt.plot(q_array, label="fine FV",marker='*')
plt.legend()


#error plots
error_fv=(q_array-a.phi_q)/a.phi_q
error_coup=(phi_q-a.phi_q)/a.phi_q
plt.figure(figsize=(12,12))
plt.rcParams.update({'font.size': 24})

fig, axs=plt.subplots(1,2, figsize=(14,7))
fig.suptitle("Flux comparison", fontsize=24)
im=axs[0].plot(np.arange(S),phi_q, label="coupling")
im=axs[0].plot(np.arange(S),a.phi_q, label="Full_SS")
im=axs[0].plot(np.arange(S),q_array, label="fine FV ({}xfiner mesh)".format(ratio))
axs[0].legend()
axs[0].set_title("Absolute flux")
im=axs[1].plot(np.arange(S),np.abs(error_fv), label="FV({}xfiner mesh)".format(ratio))
im=axs[1].plot(np.arange(S),np.abs(error_coup), label="couping")
plt.title("relative error: \n finite volumes vs coupling model")
plt.legend()

L2=np.sqrt(np.sum(error_coup**2))