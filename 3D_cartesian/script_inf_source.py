#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:00:59 2021

@author: pdavid
"""

import assembly_cartesian as ac
from post_processing import *
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg
import time




L=np.array([6.25,6.25,2.25])
h=np.array([0.15,0.15,0.15])
hs=np.min(h)/5

R=h[0]
Lx,Ly,Lz=L
#Constant vessel along the x axis
init=np.array([0.2*Lx,0,0.5*Lz])
end=np.array([0.8*Lx,Ly,0.5*Lz])
L_vessel=np.linalg.norm(end-init)

Rv=0.01
K_eff=10/(np.pi*Rv**2)
D=1

#plt.plot(np.array([init[0], end[0]]), np.array([init[1], end[1]]))



op=ac.Lap_cartesian(1, L,h)
op.assembly()
lap=sp.sparse.csc_matrix((op.data, (op.row, op.col)), shape=(op.total, op.total))

        
RHS=np.zeros(op.total)

source=ac.finite_source(hs, init, end,0.02)
source.get_coupling_cells(op.x, op.y, op.z)
source.get_s_coordinates()

phi_vessel=np.cos(np.pi*source.s/(L_vessel*2)) 
#phi_vessel=np.ones(len(source.s))

def get_q_RHS(coupling_cells, q_value, h, hs, total):
    vol=h[0]*h[1]*h[2]
    RHS=np.zeros(total)
    c=0
    for i in coupling_cells:

        RHS[i]-=q_value[c]*(hs/vol)
        c+=1
    return(RHS)


RHS=get_q_RHS(source.coup_cells, phi_vessel, h, hs, op.total)


Dirichlet=np.zeros(6)
RHS, ope=ac.set_TPFA_Dirichlet(Dirichlet,lap.copy(),  h, op.boundary, RHS)



# =============================================================================
# #Set Dirichlet BC
# c=0
# d=["x", "y", "z"] #list containing the axis parallel to the boundaries that have Dirichlet BCs
# 
# for i in op.boundary.T:
#     code=i[1]
#     pos=i[0]
#     f="x" in d
#     w="y" in d
#     n="z" in d
#     if n and code<100: #everything except north and south
#         lap[pos,:]=0
#         lap[pos,pos]=1
#     if w and (code%100)//10 != 0 :
#         lap[pos,:]=0
#         lap[pos,pos]=1
#     if f and code%10:
#         lap[pos,:]=0
#         lap[pos,pos]=1
#     
# RHS[op.boundary[0,:]]=0
# 
# =============================================================================

#Solver
sol=sp.sparse.linalg.spsolve(ope,RHS)


        
class domain():
    def __init__(self, x, y, z):
        self.x=x
        self.y=y
        self.z=z
        self.lx=len(x)
        self.ly=len(y)
        self.lz=len(z)


#Post-processing

o=post_processing(sol, op)
plt.figure()
p=o.get_contour(0.5*L, "x","x_slice")
plt.figure()
p=o.get_contour(0.5*L, "y","y_slice")
plt.figure()
p=o.get_contour(0.5*L, "z","z_slice")

plt.figure()
plt.plot(op.y, sol[o.get_profile(0.5*L, "x")])
plt.xlabel("x")
plt.ylabel("concentration")
plt.title("profile")

plt.figure()
plt.plot(sol[source.coup_cells])


l=np.array([len(op.x), len(op.y), len(op.z)])
get_25_profiles("x", l, source.coup_cells, sol)




