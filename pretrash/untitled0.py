#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:51:14 2021

Code for testing the splitting method and the discretized green function

@author: pdavid
"""

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg
import time



zPoints=1000
rPoints=1000

class assemble_system_sparse_cylindrical():
    def __init__(self,K_eff, D_eff, z_vector, r_vector, h_z, h_r):
        self.K=K_eff
        self.D=D_eff
        self.h_z=h_z
        self.h_r=h_r
        self.z=z_vector
        self.r=r_vector
        
    def get_stensil(self, n_col, n_row,  boundaries):
        
        data=np.array([])
        col=np.array([], dtype=int)
        row=np.array([], dtype=int)
        
        D=self.D
        rho_n=(self.r[n_col]+self.r[n_col+1])/2
        rho_s=(self.r[n_col]+self.r[n_col-1])/2
        rho_i=self.r[n_col]
        
        north, south, east, west=boundaries
        coeff_north, coeff_south, coeff_east, coeff_west=0,0,0,0
        if north:
            coeff_north=rho_n*D/(rho_i*self.h_r**2)
            row=np.append(row, n_row)
            col=np.append(col, n_col+1)
            data=np.append(data, coeff_north)
        if south:
            coeff_south=rho_s*D/(rho_i*self.h_r**2)
            row=np.append(row, n_row)
            col=np.append(col, n_col-1)
            data=np.append(data, coeff_south)
        if east:
            coeff_east=D/self.h_z**2
            row=np.append(row, n_row+1)
            col=np.append(col, n_col)
            data=np.append(data, coeff_east)
        if west:
            coeff_west=D/self.h_z**2
            row=np.append(row, n_row-1)
            col=np.append(col, n_col)
            data=np.append(data, coeff_west)
            

        #center
        row=np.append(row, n_row)
        col=np.append(col, n_col)
        data=np.append(data, -(coeff_north+coeff_south-coeff_east-coeff_west))        
        return(np.vstack([data, row, col]))
        
        
        
    def assembly(self):
        #Arrays that compose the matrix 
        data=np.array([])
        col=np.array([], dtype=int)
        row=np.array([], dtype=int)
        
        for i in range(len(self.z)): #i represents the longitudinal position
            for j in range(len(self.r)): #j represents the radial position
                #boundaries=north, south, east, west 
                if i==0: #south
                    array=self.get_stensil(j,i,(True, False, True, True))
                elif i==len(self.z)-1: #north
                    array=self.get_stensil(j,i,(False, True, True, True))
                elif j==0: #west
                    array=self.get_stensil(j,i,(True, True, True, False))
                elif j==len(self.r)-1: #east
                    array=self.get_stensil(j,i,(True, True, False, True))
                else: 
                    array=self.get_stensil(j,i,(True, True, True, True))
                
                data=np.concatenate([data, array[0]])
                row=np.concatenate([row, array[1]])
                col=np.concatenate([col, array[2]])
        
        self.A=sp.sparse.csc_matrix((data, (row, col)), shape=(len(self.z)*len(self.r), len(self.z)*len(self.r)))
             

            
            
            
        
        
        
        
        
