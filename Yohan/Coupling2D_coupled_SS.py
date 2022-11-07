#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:52:20 2021

@author: pdavid
"""

import numpy as np 
import matplotlib.pyplot as plt
from module_2D_coupling import * 
import random 
import scipy as sp
import scipy.sparse.linalg
#0-Set up the sources
Rv=0.01

C0=1
K_eff=1/(np.pi*Rv**2)


#1-Set up the domain
D=1
L=6
h_ref=0.05
h_ss=1

validation=False
x_ref=np.linspace(h_ref/2, L-h_ref/2,int(L//h_ref))
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
y_ref=x_ref
y_ss=x_ss

n_sources=1
#pos_s=np.array([[4.5,4.5],[2.5,2.5]])+0.25
pos_s=np.array([[3.5,3.5]])+0.25
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

t=assemble_SS_2D(pos_s, A, Rv, h_ss,x_ss,y_ss, K_eff, D)

# =============================================================================
# #get the tranmisibilities
# cell_s_1=pos_to_coords(x_ss, y_ss, t.coup_cells[0])
# cell_s_2=pos_to_coords(x_ss, y_ss, t.coup_cells[1])
# trans_1=get_trans(t.h,t.h, pos_s[0]-cell_s_1)
# trans_2=get_trans(t.h,t.h, pos_s[1]-cell_s_2)
# =============================================================================

i=0
pos_v_n=n_sources+4*t.cell_source_position[i]
pos_v=np.array([pos_v_n,pos_v_n+1,pos_v_n+2,pos_v_n+3 ]).astype(int)


t.assemble_full_system(np.ones(n_sources))

solution=np.linalg.solve(t.M, t.RHS)



plt.imshow(solution[:len(t.y)*len(t.x)].reshape(len(t.y), len(t.x)))
plt.colorbar()
q_value=solution[len(t.x)*len(t.y):len(t.x)*len(t.y)+n_sources]
def get_v_values(solution, coup_cells, total_FV_cells):
    print("only for one source per cell")
    n_sources=len(coup_cells)
    v_value=np.zeros((n_sources, 5))
    v_value[:,0]=solution[coup_cells]
    print(v_value[0,1:])
    for i in range(n_sources):
        v_value[i,1:]=solution[total_FV_cells+n_sources+4*i:total_FV_cells+n_sources+4*(i+1)]
    return(v_value)

v=get_v_values(solution, t.coup_cells, total_FV_cells)

R=reconstruction_microscopic_field(t, solution,10)
micro_solution=R.reconstruction()

if validation==True:
    #Solution reference
    A_ref=A_assembly(len(x_ref), len(y_ref))*D/h_ref**2
    RHS_ref=np.zeros(len(x_ref)*len(y_ref))
    b=get_boundary_vector(len(x_ref), len(y_ref))
    A_ref[b,:]=0
    A_ref[b,b]=1
    RHS_ref[b]=0
    
    ref_coup_cells=np.zeros(n_sources, dtype=int)
    for i in range(n_sources):
        ref_coup_cells[i]=coord_to_pos(x_ref,y_ref,pos_s[i])
    
    RHS_ref[ref_coup_cells]=-phi_sources
    
    A_ref_initial=A_ref[ref_coup_cells,:].copy()
    A_ref[ref_coup_cells,:]*=(h_ref**2)/C0
    A_ref[ref_coup_cells,ref_coup_cells]-=1
    
    sol_ref=np.linalg.solve(A_ref, RHS_ref).reshape(len(y_ref), len(x_ref))
    q_ref=C0*(phi_sources-np.ndarray.flatten(sol_ref)[ref_coup_cells])
    q_ref2=-A_ref_initial.dot(np.ndarray.flatten(sol_ref))*h_ref**2
    phi_ref3=A_ref[ref_coup_cells,:].dot(np.ndarray.flatten(sol_ref))
