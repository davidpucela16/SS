#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:51:14 2021

Code for testing the splitting method and the discretized green function

@author: pdavid
"""

##################################

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg
import time



class assemble_system_sparse_cylindrical():
    def __init__(self,K_eff, D_eff, z_vector, r_vector, h_z, h_r):
        self.K=K_eff
        self.D=D_eff
        self.h_z=h_z
        self.h_r=h_r
        self.z=z_vector
        self.r=r_vector
        self.lenz=len(z_vector)
        self.lenr=len(r_vector)
        self.lentis=self.lenz*self.lenr
        
    def get_stensil(self, n_row, n_col,  boundaries):
        """n_col -> radial postion
           n_row -> longitudinal postion"""
        pos=n_col*self.lenz+n_row
        data=np.array([])
        col=np.array([], dtype=int)
        row=np.array([], dtype=int)
        
        D=self.D
        rho_i=self.r[n_col]
        
        north, south, east, west=boundaries
        coeff_north, coeff_south, coeff_east, coeff_west=0,0,0,0
        if north:
            rho_n=(self.r[n_col]+self.r[n_col+1])/2
            coeff_north=rho_n*D/(rho_i*self.h_r**2)
            row=np.append(row, pos)
            col=np.append(col, pos+self.lenz)
            data=np.append(data, coeff_north)
        if south:
            rho_s=(self.r[n_col]+self.r[n_col-1])/2
            coeff_south=rho_s*D/(rho_i*self.h_r**2)
            row=np.append(row, pos)
            col=np.append(col, pos-self.lenz)
            data=np.append(data, coeff_south)
        if east:
            coeff_east=D/self.h_z**2
            row=np.append(row, pos)
            col=np.append(col, pos+1)
            data=np.append(data, coeff_east)
        if west:
            coeff_west=D/self.h_z**2
            row=np.append(row, pos)
            col=np.append(col, pos-1)
            data=np.append(data, coeff_west)
            

        #center
        row=np.append(row, pos)
        col=np.append(col, pos)
        data=np.append(data, -(coeff_north+coeff_south+coeff_east+coeff_west))        
        return(np.vstack([data, row, col]))
        
        
    
    def assembly(self):
        """Data returned in a vector of shape=(3, number of elements), where the first position 
        is data, second is row and third is col"""
        #Arrays that compose the matrix 
        data=np.array([])
        col=np.array([], dtype=int)
        row=np.array([], dtype=int)
        
        for i in range(len(self.z)): #i represents the longitudinal position
            for j in range(len(self.r)): #j represents the radial position

                #boundaries=north, south, east, west 
                north, south, east, west=True, True, True, True
                if j==0: #south
                #j should never be zero since we are gonna adjust this line to couple it with the vessel
                    south=False
                    data=np.append(data,-self.K)
                    row=np.append(row, i)
                    col=np.append(col, i)
                if j==len(self.r)-1: #north
                    north=False
                if i==0: #west
                    west=False
                if i==len(self.z)-1: #east
                    east=False
                boundaries=north, south, east, west
                array=self.get_stensil(i,j,boundaries)
                data=np.concatenate([data, array[0]])
                row=np.concatenate([row, array[1]])
                col=np.concatenate([col, array[2]])
        
        self.A=sp.sparse.csc_matrix((data, (row, col)), shape=(len(self.z)*len(self.r), len(self.z)*len(self.r)))
        
        self.data_array=np.vstack([data, row, col])
        return(self.data_array)
    


    def matrix_B_assembly(self,  d, factor, z_vessel):
        K_eff=self.K
        lentissue=len(self.z)*len(self.r)
        ID_network=np.array([])
        for i in z_vessel:
            ID_network=np.append(ID_network, np.argmin((i-self.z)**2))
        c=0
        row_B=np.array([], dtype=int)
        col_B=np.array([], dtype=int)
        data_B=np.array([])
        for i in ID_network:
            row_B=np.append(row_B, i)
            col_B=np.append(col_B, c)
            data_B=np.append(data_B, -0.25*d**2*K_eff/factor)
            c+=1 
        return(np.vstack([data_B, row_B, col_B]))
    
    def compute_matrix_shapes(self, z_vessel):
        """kinda useless function to compute the shape of the matrices"""
        network=z_vessel
        lentis=len(self.z)*len(self.r)
        array_to_return=[[lentis, lentis],[lentis, len(network)],[len(network), lentis],[len(network), len(network)]] 
        #list of tuples in the order a, B, C, D
        return(np.array(array_to_return))
    
    def assembly_full_system_FixedVesselConcentration(self,Rv, phi_v, z_vessel):
        """This function will compute the full problem (BC included) and RHS included for 
        a system where the concentration in the 1D line vessel is given by a function. Things to 
        keep in mind about this function:
            -> it is a method of a cylindrical class, therefore it should be kept in mind the 
               problem must be axylsymmetric in nature.
            -> The vessel is a line
            -> The first row of the tissue matrix is considered to be at r=0. Therefore it is consider
               to be exactly at the level of the line vessel. It is treated as a no volume FV cell in
               order to apply the BCs there
            -> The (steady state) system will be the following shape:
               0=A*phi= [a  B]* {phi_tissue} 
                        [C  D]  {phi_vessel} """
        shapes=self.compute_matrix_shapes(z_vessel)
        a=self.assembly() #get the tissue matrix 
        self.a=a
 

        b=np.array([np.zeros(len(z_vessel))+self.K,np.arange(len(z_vessel)),np.arange(len(z_vessel))])
        

        
        #The lower portion of the A matrix is simple, since the pressure is fixed on this particular 
        #problem (Not coded the intravascular transport equation yet)
        row_D=np.arange(len(z_vessel))
        col_CD=np.arange(len(z_vessel))+self.lentis
        data_D=np.zeros(len(z_vessel))+1
        cd=sp.sparse.csc_matrix((data_D, (row_D, col_CD)), shape=(shapes[2][0],shapes[2][1]+shapes[3][1]))
        
        A=sp.sparse.csc_matrix((a[0], (a[1], a[2])), shape=(shapes[0]))
        B=sp.sparse.csc_matrix((b[0], (b[1], b[2])), shape=(shapes[1]))

        A=sp.sparse.hstack([A,B])
        self.A=A
        self.cd=cd
        A=sp.sparse.csc_matrix(sp.sparse.vstack([A,cd]))
        
        
        return(A)

            
            
            
        
        
        
        
        
