#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:46:22 2021

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
import sol_split_3D as SS

plt.rcParams.update({'font.size': 25})

L=np.array([6.25,6.25,2.25])
h=np.array([0.25,0.25,0.25])
ratio=5
hs=np.min(h)/ratio

R=h[0]
Lx,Ly,Lz=L
#Constant vessel along the x axis
init=np.array([0.2*Lx,0,0.5*Lz])
end=np.array([0.8*Lx,Ly,0.5*Lz])
L_vessel=np.linalg.norm(end-init)

Rv=0.01
K_eff=10/(np.pi*Rv**2)
D=1
validation=True


#plt.plot(np.array([init[0], end[0]]), np.array([init[1], end[1]]))

#Initialize source object
source=ac.finite_source(hs, init, end, Rv)
source.get_s_coordinates() #Initializes the init, end, and center arrays 


q_vessel=np.cos(np.pi*source.s/(L_vessel*2)) 
#q_vessel=np.ones(len(source.s))


def G_1D(s_coord, x_coord, Rv):
    d=np.linalg.norm(s_coord-x_coord)
    if d<Rv:
        print("too close")
        g=1/(Rv*np.pi*4)
    else:
        g=1/(d*np.pi*4)
    return(g)

def cell_single_point_term(s_coord,  x_coord, Rv):
    """Returns the kernel to multiply the q_array (of the vessel) that will 
    provide the singular term at the x_coord"""
    x=x_coord
    array=np.zeros(s_coord.shape[0])
    for i in np.arange(s_coord.shape[0]): #through the whole vessel       
        #"Integrates" through the network to get the contribution from each 
        
        array[i]=G_1D(s_coord[i], x, Rv)
    return(array)

def get_inf_dom_point_solution(q, x, y, z, s_coord, Rv, hs):
    sol=np.zeros(len(x)*len(y)*len(z))
    for i in range(len(sol)):
        a=ac.position_to_coordinates(i,  x, y, z)
        sol[i]=np.dot(cell_single_point_term(s_coord, a, Rv),q)*hs
    return(sol)


def get_inf_domain_solution(q, x, y, z, init_array, end_array, Rv):
    sol=np.zeros(len(x)*len(y)*len(z))
    for i in range(len(sol)):
        a=ac.position_to_coordinates(i,  x, y, z)
        sol[i]=np.dot(SS.cell_single_term(init_array, end_array, a, Rv),q)
    return(sol)



#Solution with the 2D G-function
cart_op=ac.Lap_cartesian(1, L,h)
cart_op.assembly()
cart_lap=sp.sparse.csc_matrix((cart_op.data, (cart_op.row, cart_op.col)), shape=(cart_op.total, cart_op.total))
sol_2D=get_inf_domain_solution(q_vessel, cart_op.x, cart_op.y, cart_op.z, source.init_array, source.end_array, Rv)
laplacian_sing_term_2D=np.dot(cart_lap.toarray(), sol_2D)
o=post_processing(sol_2D, cart_op)
o.get_middle_plots(L, "Green function")


#Sing term solution for the 3D G-function:
sol_1D=get_inf_dom_point_solution(q_vessel, cart_op.x, cart_op.y, cart_op.z, source.s_coords, Rv, hs)
laplacian_sing_term_1D=np.dot(cart_lap.toarray(), sol_1D)
o_1D=post_processing(sol_1D, cart_op)
o_1D.get_middle_plots(L, "1D Green function")





#Validation with FV
FV_op=ac.Lap_cartesian(1, L,h)
FV_op.assembly()
FV_lap=sp.sparse.csc_matrix((FV_op.data, (FV_op.row, FV_op.col)), shape=(FV_op.total, FV_op.total))

source.get_coupling_cells(FV_op.x, FV_op.y, FV_op.z)
        
RHS=np.zeros(FV_op.total)



def get_q_RHS(coupling_cells, q_value, h, hs, total):
    vol=h[0]*h[1]*h[2]
    RHS=np.zeros(total)
    c=0
    for i in coupling_cells:

        RHS[i]-=q_value[c]*(hs/vol)
        c+=1
    return(RHS)

RHS=get_q_RHS(source.coup_cells, q_vessel, h, hs, FV_op.total)

Dirichlet=np.zeros(6)
RHS_BC, FV_op=ac.set_TPFA_Dirichlet(Dirichlet,FV_lap.copy(),  h, FV_op.boundary, RHS)

if validation:
    FV_sol=sp.sparse.linalg.spsolve(FV_op,RHS_BC)
o_FV=post_processing(FV_sol, cart_op)
o_FV.get_middle_plots(L, "Validation")
    
# =============================================================================
# #Validation for the inf_dom_sol
# lap_sing_term_validation=np.zeros(cart_op.total)
# c=0
# for i in source.coup_cells:
#     lap_sing_term_validation[i]-=q_vessel[c]/(ratio*h[0]**2)
#     c+=1
# o_val=post_processing(lap_sing_term_validation, cart_op)
# o_val.get_middle_plots(L, "validation")
# 
# 
# =============================================================================

plt.figure()
plt.plot(np.dot(cart_lap.toarray(),FV_sol)[source.coup_cells], label="validation")
plt.plot(laplacian_sing_term_2D[source.coup_cells], label="2D")
plt.plot(laplacian_sing_term_1D[source.coup_cells], label="1D")
plt.xlabel("tissue cell along the vessel")
plt.ylabel("laplacian value")
plt.legend(fontsize=10)


#get the cells where the cell center lies too close to a vessel:

counter=np.array([])
ppp=np.where(laplacian_sing_term_1D<-50)[0]
for i in range(cart_op.total):
    coord=ac.position_to_coordinates(i, cart_op.x, cart_op.y, cart_op.z)
    for j in source.s_coords:
        d=np.linalg.norm(coord-j)
        if d<10*Rv:
            counter=np.append(counter, i)
        
        



