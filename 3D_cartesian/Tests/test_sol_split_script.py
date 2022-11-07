#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 19:53:23 2021

@author: pdavid
"""

import sys
sys.path.append('../')

import assembly_cartesian as ac
from post_processing import *
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg
import time
from sol_split_3D import *

L=np.array([10.25,10.25,10.5])
h=np.array([0.25,0.25,0.25])
hs=np.min(h)/5

R=h[0]
Lx,Ly,Lz=L
#Constant vessel along the x axis
init=np.array([0.5*Lx,0,0.7*Lz])
end=np.array([0.5*Lx,Ly,0.7*Lz])
L_vessel=np.linalg.norm(end-init)

Rv=0.01
K_eff=10/(np.pi*Rv**2)
D=1

#plt.plot(np.array([init[0], end[0]]), np.array([init[1], end[1]]))

#Initialize source object
s=ac.finite_source(hs, init, end, Rv)
s.get_s_coordinates() #Initializes the init, end, and center arrays 

#phi_vessel=np.cos(np.pi*s.s/(L_vessel*2)) 
phi_vessel=np.ones(len(s.s))

#CREATION OF THE SOL SPLIT OBJECT
t=assemble_sol_split(phi_vessel, L, h, s, K_eff, D) #in the init function there is already the creation 
                                                        #of the laplacian operator

#Assembly of the D matrix (single term on the vessel walls)
D=t.D_assembly()   
#Assembly of the C matrix (mainly a linear interpolator that calculates the value of the regular term
#on each of the vessel centerlines)
d=t.C_assembly(R)
C=sp.sparse.csc_matrix((d[0], (d[1], d[2])),shape=(len(phi_vessel), t.total_tissue))

     
RHS=np.zeros(t.total_tissue)

Dirichlet=np.zeros(6)

(RHS, ope, B)=set_TPFA_Dirichlet_SS(Dirichlet,t.lap.copy(),  h, t.op.boundary, RHS,t.op.x, t.op.y, t.op.z, s.init_array, s.end_array, Rv)

#Include the BOUNDARY CONDITIONS for the vessel 
RHS_total=np.concatenate((RHS, phi_vessel))

D=sp.sparse.diags(np.ones(len(phi_vessel)), 0)
C=sp.sparse.coo_matrix((len(phi_vessel), t.total_tissue))

M=sp.sparse.hstack((ope, B))
M=sp.sparse.vstack((M,sp.sparse.hstack((C,D))))

#sol=sp.sparse.linalg.spsolve(M,RHS_total)
sol_reg_term=sp.sparse.linalg.spsolve(M, RHS_total)
o_regterm=post_processing(sol_reg_term, t)
o_regterm.get_middle_plots(L, "reg term")

# =============================================================================
# def D_test(init_array, end_array, coords, Rv, phi_vessel):
#     #for a single cell 
#     a=0
#     c=0
#     for i in range(len(phi_vessel)):
#         a+=sing_term(init_array[i], end_array[i], coords, Rv)
#         c+=1
#     return(a)
# 
# c=0
# for i in range(len(phi_vessel)):
#     a=D_test(s.init_array, s.end_array, t.s_coords[i], Rv, phi_vessel )
# 
#     print((a+1)==np.sum(D[i,:]))
#     if not (a+1)==np.sum(D[i,:]):
#         print(a+1)
#         print(np.sum(D[i,:]))
#     c+=1
#     
#     
# data=np.array([])
# row=np.array([])
# col=np.array([])
# for i in range(len(t.phi_vessel)):
#     print(i)
#     a=linear_interpolation(t.s_coords[i], t.op.x, t.op.y, t.op.z, R)
#     data=np.concatenate([data, a[:,1]])
#     col=np.concatenate([col, a[:,0]])
#     row=np.concatenate([row, np.zeros(len(a[:,0]))+i])
# 
# =============================================================================



def get_inf_domain_solution(q, x, y, z, init_array, end_array, Rv):
    sol=np.zeros(len(x)*len(y)*len(z))
    for i in range(len(sol)):
        a=ac.position_to_coordinates(i,  x, y, z)
        sol[i]=np.dot(cell_single_term(init_array, end_array, a, Rv),q)
    return(sol)


sol_total=get_inf_domain_solution(phi_vessel, t.x, t.y, t.z, t.init_array, t.end_array, Rv)+sol_reg_term[:-len(phi_vessel)]
o_tot=post_processing(sol_total, t)
o_tot.get_middle_plots(L, "total solution")

pp=s.get_coupling_cells(t.x, t.y, t.z)
plt.figure()
plt.plot(sol_total[s.coup_cells])

get_25_profiles("x",np.array([len(t.x),len(t.y),len(t.z)]), s.coup_cells, sol_total)
# =============================================================================
# source.get_coupling_cells(t.op.x, t.op.y, t.op.z)
# source.get_s_coordinates()
# 
# op_virgin=ac.Lap_cartesian(1, L,h)
# op_virgin.assembly()
# lap_virgin=sp.sparse.csc_matrix((op_virgin.data, (op_virgin.row, op_virgin.col)), shape=(op_virgin.total, op_virgin.total))
# 
# 
# inf_dom=get_inf_domain_solution(np.ones(len(phi_vessel)), t.op.x, t.op.y, t.op.z, t.init_array, t.end_array, 0.01)
# 
# lap_inf_dom=np.dot(lap_virgin.toarray(), inf_dom)
# 
# 
# #Post-processing
# 
# o=post_processing(lap_inf_dom/8.8888, t.op)
# plt.figure()
# p=o.get_contour(0.5*L, "x","x_slice")
# plt.figure()
# p=o.get_contour(0.5*L, "y","y_slice")
# plt.figure()
# p=o.get_contour(0.5*L, "z","z_slice")
# 
# plt.figure()
# plt.plot(t.op.y, o.solution[o.get_profile(0.5*L, "x")])
# plt.xlabel("x")
# plt.ylabel("concentration")
# plt.title("profile")
# 
# fig = plt.figure()
# 
# ax = fig.add_subplot(111, projection='3d')
# 
# ax.plot(source.s_coords[:,0], source.s_coords[:,1], source.s_coords[:,2])
# 
# plt.show()
# 
# 
# 
# #I am gonna calculate the solution for the laplacian of the infinite domain for the constant fixed flux=1
# lap_sing_term=np.zeros(t.total_tissue)
# for i in source.coup_cells:
#     lap_sing_term[i]-=1/(5*h[0]**2)
# =============================================================================
