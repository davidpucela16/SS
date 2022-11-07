#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:25:00 2021

@author: pdavid

Testing functions script
"""


import numpy as np 
import matplotlib.pyplot as plt
from module_2D_coupling_neigh import * 
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

pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6],[2.5,2.5]])-np.array([0.25,0.25])
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
A=A_assembly(len(x_ss), len(y_ss))*D/h_ss**2
#set dirichlet
B,A=set_TPFA_Dirichlet(0,A, h_ss, get_boundary_vector(len(x_ss), len(y_ss)), np.zeros(len(x_ss)*len(y_ss)),D)

t=assemble_SS_2D_FD(pos_s, A, Rv, h_ss,x_ss,y_ss, K_eff, D)
t.pos_arrays()





for i in range(8):
    a,b,c=get_v_neigh_norm_FVneigh(i, len(t.x))
    print("For DoF=", i)
    print("v_neigh", a)
    print("FV_neigh", b)
    print("normal",c)
    
    
    
