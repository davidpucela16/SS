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
from module_2D_coupling import * 
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
L=6
h_ss=1

validation=False
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
y_ss=x_ss

n_sources=1
pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6],[1.5,4.5]])-np.array([0.25,0.25])
#pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6]])-np.array([0.25,0.25])
phi_sources=np.ones(pos_s.shape[0])


        

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
total_FV_cells=len(x_ss)*len(y_ss)
A=A_assembly(len(x_ss), len(y_ss))*D/h_ss**2
boundary=get_boundary_vector(len(x_ss), len(y_ss))



#set dirichlet
B,A=set_TPFA_Dirichlet(0,A, h_ss, get_boundary_vector(len(x_ss), len(y_ss)), np.zeros(len(x_ss)*len(y_ss)),D)

t=assemble_SS_2D_FD(pos_s, A, Rv, h_ss,x_ss,y_ss, K_eff, D)
t.pos_arrays()

t.assembly_sol_split_problem()

B_v=np.zeros(len(np.unique(t.s_blocks))*9)
B_q=np.ones(len(t.s_blocks))

B=np.concatenate((B,B_v,B_q))

phi=np.linalg.solve(t.M, B)

phi_FV=phi[:len(x_ss)*len(y_ss)]
phi_mat=phi_FV.reshape(len(x_ss), len(y_ss))

phi_FV, phi_v, phi_q=post.separate_unk(t, phi)

Up=np.hstack((t.A, t.b, t.c))

grads=post.reconstruction_gradients_manual(phi, t,get_boundary_vector(len(t.x), len(t.y))) 

o=post.reconst_microscopic(t, phi)    
rec=o.reconstruct(25,L)
plt.imshow(rec)
plt.colorbar



def get_validation(ratio, SS_ass_object, pos_s, phi_j, D, K_eff, Rv):
    t=SS_ass_object
    C_0=K_eff*np.pi*Rv**2
    h=t.h/ratio
    num=int(L//h)
    h=L/num
    x=np.linspace(h/2, L-h/2, num)
    y=x
    
    A=A_assembly(len(x), len(y))*D/h**2
    A_virgin=A_assembly_Dirich(len(x), len(y))*D/h**2
    boundary=get_boundary_vector(len(x_ss), len(y_ss))

    #set dirichlet
    B,A=set_TPFA_Dirichlet(0,A, h, get_boundary_vector(len(x), len(y)), np.zeros(len(x)*len(y)),D)
    #Set sources
    s_blocks=np.array([], dtype=int)
    c=0
    for i in pos_s:
        x_pos=np.argmin(np.abs(i[0]-x))
        y_pos=np.argmin(np.abs(i[1]-y))
        
        block=y_pos*len(x)+x_pos
        A[block, block]-=C_0/h**2
        B[block]-=C_0/h**2*phi_j[c]
        s_blocks=np.append(s_blocks, block)
        c+=1
    sol=np.linalg.solve(A,B)
    
    q_array=-np.dot(A_virgin[s_blocks],sol)*h**2/D
    
    return(sol, len(x), len(y),q_array, B, A, s_blocks)
    

sol, xlen2, ylen2,b,a, Y, s_b=get_validation(10, t, pos_s, B_q, D, K_eff, Rv)
plt.imshow(sol.reshape(ylen2, xlen2))
    