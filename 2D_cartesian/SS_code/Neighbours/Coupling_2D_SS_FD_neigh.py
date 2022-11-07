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
from module_2D_coupling_neigh import * 
import module_2D_coupling_orig as orig
import reconst_and_test_module_neigh as post
import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import pdb
#0-Set up the sources
Rv=0.01
C0=1
K_eff=1/(np.pi*Rv**2)


#1-Set up the domain
D=1
L=4
h_ss=1

validation=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
y_ss=x_ss

#pos_s=np.array([[2.5,1.25],[2.5,1.75],[1.5,1.25],[1.5,1.75]])

pos_s=np.array([[1.5,1.5],[2.5,1.5],[1.5,2.5],[2.5,2.5]])

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
B=np.concatenate((B,B_v,B_q))

phi=np.linalg.solve(t.M, B)

phi_FV, phi_v, phi_q=post.separate_unk(t, phi)
phi_mat=phi_FV.reshape(len(x_ss), len(y_ss))



grads=post.reconstruction_gradients_manual(phi, t,get_boundary_vector(len(t.x), len(t.y))) 

o=post.reconst_microscopic(t, phi)    
rec=o.reconstruct(20,L)
plt.imshow(rec, extent=[0,L,0,L],origin='lower')
plt.colorbar()
plt.show()
    
if validation:
    sol, xlen2, ylen2,q_array,a, Y, s_b,x_v, y_v=get_validation(20, t, pos_s, B_q, D, K_eff, Rv,L)
    plt.imshow(sol.reshape(ylen2, xlen2), origin="lower"); plt.colorbar()



