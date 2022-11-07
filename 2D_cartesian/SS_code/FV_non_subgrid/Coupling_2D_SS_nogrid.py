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
Rv=0.01
C0=1
K_eff=1/(np.pi*Rv**2)


#1-Set up the domain
D=1
L=3
h_ss=0.5

validation=False
real_Dirich=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
y_ss=x_ss

#pos_s=np.array([[x_ss[2], y_ss[2]],[x_ss[4], y_ss[4]]])
#pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6],[2,2]])-np.array([0.25,0.25])
#pos_s/=2
pos_s=np.array([[1.25,1.25],[1.25,1.75], [1.75,1.75],[1.75,1.25]])
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
    
    
t=assemble_SS_2D_FD(pos_s, A, Rv, h_ss,x_ss,y_ss, K_eff, D)
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


o=post.reconst_microscopic(t, phi)    
rec=o.reconstruct(10,L)
grads=o.grads


plt.imshow(rec, extent=[0,L,0,L], origin='lower')
plt.colorbar()
plt.show()
    

sol, xlen2, ylen2,q_array,a, Y, s_b,x_v, y_v=get_validation(10, t, pos_s, B_q, D, K_eff, Rv,L)
plt.imshow(sol.reshape(ylen2, xlen2), origin='lower'); plt.colorbar()
plt.show()

a=full_ss(pos_s, Rv, h_ss, K_eff, D,L)
SS=a.solve_problem(B_q)
SS=a.reconstruct(a.v, a.phi_q)
plt.show()
plt.imshow(SS, origin="lower"); plt.colorbar()
plt.show()


#reconstruct the full solution splitting
a.uni_s_blocks=np.array([])
a.s_blocks=np.array([])
a.FV_DoF=np.arange(len(a.v))
u=post.reconst_microscopic(a, a.v)    
rec=u.reconstruct(10,L)


plt.plot(a.x, SS[:,3])
plt.plot(x_ss, phi_mat[:,3])
plt.show()