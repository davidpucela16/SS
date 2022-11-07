#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:32:04 2021

@author: pdavid
"""

import sys
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg
import time
sys.path.insert(1, '/home/pdavid/Bureau/Code/Solution_splitting')

from assembly import assemble_system_sparse_cylindrical
from bifurcation_massTransfer import flow 
from bifurcation_massTransfer import massTransport #Although for just one vessel it may be better to 
#straight forwardly code a new function

z_Points=50
r_Points=50
L=R=10
h_z=L/z_Points
h_r=R/r_Points

z_faces=np.linspace(0,L, z_Points+1)
r_faces=np.linspace(0, R, r_Points+1)

z_cells=z_faces[:-1]+h_z/2
r_cells=r_faces[:-1]+h_r/2

#We are gonna start with a fixed concentration profile with the shape of a cosine:
z_vessel_faces=np.linspace(0,L, 10*z_Points+1)
factor=10 #relationship between the discretization size longitudinal to the vessel in the network and in the tissue
z_vessel=z_vessel_faces[:-1]+h_z/factor
phi_v=np.cos(np.pi*z_vessel/(2*L))

ass=assemble_system_sparse_cylindrical(1, 1, z_cells, r_cells, h_z, h_r)
A=ass.assembly()

shapes=ass.compute_matrix_shapes(z_vessel)

AA=sp.sparse.csc_matrix((A[0], (A[1], A[2])), shape=(shapes[0]))

A=ass.assembly_full_system_FixedVesselConcentration(1, phi_v, z_vessel)
    
phi_tissue=np.zeros(ass.lentis)
phi=np.concatenate([phi_tissue, phi_v])
sol=sp.sparse.linalg.spsolve(A, phi)

sol_tissue=sol[:r_Points*z_Points].reshape(r_Points, z_Points)
plt.contourf(sol_tissue)
plt.colorbar()

north_boundary=np.arange(z_Points*(r_Points-1), z_Points*r_Points-1)



# =============================================================================
#     
# 
# def matrix_B_assembly(K_eff, ID_network, lentissue, d, factor):
#     c=0
#     row_B=np.array([], dtype=int)
#     col_B=np.array([], dtype=int)
#     data_B=np.array([])
#     for i in range(len(ID_network)):
#         row_B=np.append(row_B, i)
#         col_B=np.append(col_B, c)
#         data_B=np.append(data_B, -0.25*d**2*K_eff/factor)
#         c+=1 
#     return(np.vstack([data_B, row_B, col_B]))
# 
# def compute_matrix_shapes(z, r, network):
#     """kinda useless function to compute the shape of the matrices"""
#     lentis=len(z)*len(r)
#     array_to_return=[[lentis, lentis],[lentis, len(network)],[len(network), lentis],[len(network), len(network)]]
#     #list of tuples in the order a, B, C, D
#     return(np.array(array_to_return))
# =============================================================================



    
    