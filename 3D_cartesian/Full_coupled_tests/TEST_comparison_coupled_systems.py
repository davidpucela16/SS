#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:19:53 2021

@author: pdavid
"""

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

L=np.array([6.25,6.25,2.25])
h=np.array([0.1,0.1,0.1])
hs=np.min(h)/5

R=h[0]
Lx,Ly,Lz=L
#Constant vessel along the x axis
init=np.array([0.2*Lx,0.1*Ly,0.3*Lz])
end=np.array([0.8*Lx,0.9*Ly,0.6*Lz])
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

#FV COUPLED SYSTEM
op_FV=ac.Lap_cartesian(1, L,h)
op_FV.assembly()
lap_FV=sp.sparse.csc_matrix((op_FV.data, (op_FV.row, op_FV.col)), shape=(op_FV.total, op_FV.total))
s.get_coupling_cells(op_FV.x, op_FV.y, op_FV.z)

Dirichlet=np.zeros(6)
FV_RHS, lap_FV=ac.set_TPFA_Dirichlet(Dirichlet,lap_FV.copy(),  h, op_FV.boundary, np.zeros(op_FV.total))
FV_RHS, lap_FV=ac.set_coupled_FV_system(lap_FV, phi_vessel, s.coup_cells, FV_RHS, K_eff*np.pi*Rv**2, hs,h[0]*h[1]*h[2])

sol_FV=sp.sparse.linalg.spsolve(lap_FV,FV_RHS)
get_25_profiles("x", np.array([len(op_FV.x),len(op_FV.y),len(op_FV.z)]), s.coup_cells, sol_FV,"FV h=0.1")



#CREATION OF THE SOL SPLIT OBJECT
h=1.25*h
t=assemble_sol_split(phi_vessel, L, h, s, K_eff, D) #in the init function there is already the creation 
                                                        #of the laplacian operator

#Assembly of the D matrix (single term on the vessel walls)
D=t.D_assembly()   
#Assembly of the C matrix (mainly a linear interpolator that calculates the value of the regular term
#on each of the vessel centerlines)
d=t.C_assembly(R)
C=sp.sparse.csc_matrix((d[0], (d[1], d[2])),shape=(len(phi_vessel), t.total_tissue))
#Initialize source object
source=ac.finite_source(hs, init, end, Rv)
source.get_s_coordinates() #Initializes the init, end, and center arrays 
source.get_coupling_cells(op_FV.x, op_FV.y, op_FV.z)

o_FV=post_processing(sol_FV, op_FV)
o_FV.get_middle_plots(L, "FV")
     
RHS=np.zeros(t.total_tissue)

Dirichlet=np.zeros(6)

(RHS, ope, B)=set_TPFA_Dirichlet_SS(Dirichlet,t.lap.copy(),  h, t.op.boundary, RHS,t.op.x, t.op.y, t.op.z, source.init_array, source.end_array, Rv)

#Include the BOUNDARY CONDITIONS for the vessel 
RHS_total=np.concatenate((RHS, phi_vessel))
M=sp.sparse.hstack((ope, B))
M=sp.sparse.vstack((M,sp.sparse.hstack((C,D))))

#sol=sp.sparse.linalg.spsolve(M,RHS_total)
sol_reg_term=sp.sparse.linalg.spsolve(M, RHS_total)
o_regterm=post_processing(sol_reg_term, t)
o_regterm.get_middle_plots(L, "reg term")

def get_inf_domain_solution(q, x, y, z, init_array, end_array, Rv):
    sol=np.zeros(len(x)*len(y)*len(z))
    for i in range(len(sol)):
        a=ac.position_to_coordinates(i,  x, y, z)
        sol[i]=np.dot(cell_single_term(init_array, end_array, a, Rv),q)
    return(sol)


sol_total=get_inf_domain_solution(phi_vessel, t.x, t.y, t.z, t.init_array, t.end_array, Rv)+sol_reg_term[:-len(phi_vessel)]
o_tot=post_processing(sol_total, t)
o_tot.get_middle_plots(L, "total solution")

pp=source.get_coupling_cells(t.x, t.y, t.z)
plt.figure()
plt.plot(sol_total[source.coup_cells])

get_25_profiles("x",np.array([len(t.x),len(t.y),len(t.z)]), source.coup_cells, sol_total, "Sol_split")

o=post_processing(sol_FV, op_FV)
o.get_middle_plots(L,"FV")

title="ratio=1.25"
get_25_profiles("x",np.array([len(t.x),len(t.y),len(t.z)]), source.coup_cells, sol_total, title + "Sol_split")

get_profiles_comparison_25_plots("x", np.array([len(op_FV.x),len(op_FV.y),len(op_FV.z)]), sol_FV,np.array([len(t.x),len(t.y),len(t.z)]), sol_total, title, op_FV.x, t.x)

get_contours_comparison(o_tot, o_FV, title)
get_contours_comparison(o_tot, o_regterm, "sol split vs reg term")