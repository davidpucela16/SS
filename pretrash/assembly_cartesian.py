#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:54:56 2021

@author: pdavid
"""

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg
import time



class Lap_cartesian():
    def __init__(self,K_eff, D_eff,L, h):
        """In the init function, the main calculation done (besides the initialization of the variables)
        is the calculation of the boundary nodes and its codes through the function of encode_boundary. Therefore, 
        the calling of this function already will take some time since the function encode_boundary has to iterate
        through each cell of the domain"""
        
        self.data=np.array([])
        self.row=np.array([])
        self.col=np.array([])
        
        self.K=K_eff
        self.D=D_eff
        self.Lx, self.Ly, self.Lz=L
        self.hx, self.hy, self.hz=h
        points=np.array(L/h-1, dtype=int)
        self.xPoints, self.zPoints, self.yPoints=points
        self.total=self.xPoints*self.yPoints*self.zPoints
        
        
        self.f_step=self.xPoints
        s_step=self.xPoints*self.yPoints
        self.s_step=s_step
        
        self.boundary=self.encode_boundary(self.total).astype(int)
    
    def get_arrays(self):
        self.x=np.arange(self.hx/2, self.Lx, self.hx)
        self.y=np.arange(self.hy/2, self.Ly, self.hy)
        self.z=np.arange(self.hz/2, self.Lz, self.hz)
        return()
    def get_coeffs(self, sides):
        """From the sides that are given, the function calculates the stensil. It will return a 0 if the 
        cell is a boundary
        
        This function does not admit non-homogeneous diffusion coefficient nor varying cell size (h)"""
        north, south, east, west, forw, back=sides
        coeff_cent, coeff_north, coeff_south, coeff_east, coeff_west, coeff_forw, coeff_back=np.zeros(7)
        D=self.D
        if north:
            coeff_north=D/self.hz**2
            coeff_cent-=D/self.hz**2
        if south:
            coeff_south=D/self.hz**2
            coeff_cent-=D/self.hz**2
        if east:
            coeff_east=D/self.hy**2
            coeff_cent-=D/self.hy**2
        if west:
            coeff_west=D/self.hy**2
            coeff_cent-=D/self.hy**2
        if forw:
            coeff_forw=D/self.hx**2
            coeff_cent-=D/self.hx**2
        if back:
            coeff_back=D/self.hx**2
            coeff_cent-=D/self.hx**2
        return(np.array([coeff_cent,coeff_north, coeff_south, coeff_east, coeff_west, coeff_forw, coeff_back ]))
        

    def decode_boundary(self, code):
        """It returns the array of sides (In the established order north, south, east ,west, front, back)
        returning a True if the given cell has a neighbour in that side and a False if it doesn't (therefore
        the cell constitutes the boundary on that side). 
        The only argument that is needed to feed the function is the code that will be decoded, it must have all 
        the information of the boundary "status" of the given cell. 
        For more information on the code, check the function encode_boundary"""
        north, south, east, west, front, back=True, True, True, True, True, True
        if code//100==1:
            north=False
        if code//100==2:
            south=False
        if (code%100)//10==3:
            east=False
        if (code%100)//10==4:
            west=False
        if code%10==5:
            front=False
        if code%10==6:
            back=False
        return([north, south, east, west, front, back])
    
    def encode_boundary(self,total):
        """This function encodes in an array the position of the boundaries (pos_arr), and the code that 
        identifies which type of boundary (superuseful for the corners, since one cell can represent one, two or 
                                           even three boundaries at the same time)
        
        For the boundary encoding we use the following values:
            north: +100
            south: +200
            east: +30
            west: +40
            front: +5
            back: +6
        
        Since this function iterates through each of the cells of the domain and computes the boundary information 
        regarding the length of x, y and z, this will only work if the domain is a cube, and obviously, if the f_step, 
        and s_step are properly calculated and representative of the distance between neighbours"""
    
        pos_arr=np.array([])
        code=np.array([])
        
        for i in range(total):
            c=0
            if i<self.s_step:
                #south
                c+=200
            if i>=self.total-self.s_step:
                #north
                c+=100
            if (i%self.s_step)<self.f_step:
                #west
                c+=40
            if (i%self.s_step)>=(self.s_step-self.f_step):
                #east
                c+=30
            if i%self.f_step==0:
                #back
                c+=6
            if (i%self.f_step)==self.f_step-1:
                #front
                c+=5
            
            if c!=0:
                pos_arr=np.append(pos_arr, i)
                code=np.append(code, c)
            
        return(np.vstack([pos_arr, code]))
                
            
        
    def add_toarrays(self, sides, i):
        """This function uses the position of the current cell and the boundary information included
        in the array sides to extend the matrix vectors called data (which contains the actual coefficients 
                                                                     of the matrix), and the vectors row 
        and col which contain the position of each coeff within the matrix.
        Notice how it modifies the objects arrays as well as returns the values of the stensil. The function 
        here was designed with the idea that there is no need to return something as long as the modifications
        of the objects arrays throught the class' methods are properly organized."""
        pos_arrays=[True]+sides
        coe=self.get_coeffs(sides)
        r=np.zeros(7)+i
        c=np.array([i, i+self.s_step, i-self.s_step, i+self.f_step, i-self.f_step, i+1,i-1])
        row=r[pos_arrays]
        col=c[pos_arrays]
        data=coe[pos_arrays]
        self.data=np.concatenate([self.data, coe])
        self.col=np.concatenate([self.col, col])
        self.row=np.concatenate([self.row, row])
        if np.max(col)>self.total-1:
            print(i)
            print(sides)
        return(np.vstack([data, row, col]))
    
    def set_Dirichlet_BC(self, boundaries):
        
        boundaries=np.unique(boundaries)
        data=np.array(np.zeros(len(boundaries))+1)
        row=np.array(boundaries)
        col=np.array(boundaries)
        self.row=np.concatenate([self.row, row])
        self.col=np.concatenate([self.col , col])
        self.data=np.concatenate([self.data, data])
            
        return(np.vstack([data, row, col]))
    
    def assembly(self):
        """This function performs the assembly of the laplacian operator for a 3D cartesian domain.
        The boundary nodes are taken as flux BCs, so if need be to set Dirichlet, it would be very easy
        setting the boundary rows (which have been already calculated in self.boundary[0]), to zero and inserting
        the Dirichlet BC"""
        f_step=self.xPoints
        s_step=self.xPoints*self.yPoints
        total=s_step*self.zPoints
        data=np.array([])
        row=np.array([])
        col=np.array([])
        boundaries=self.boundary[0]
        codes=self.boundary[1]
        for i in range(self.total):
            self.i=i
            if i in boundaries:
                position_in_array=np.where(boundaries==i)[0]
                sides=self.decode_boundary(codes[position_in_array])
                a=self.add_toarrays(sides, i)
                data=np.concatenate([data, a[0]])
                row=np.concatenate([row, a[1]])
                col=np.concatenate([col, a[2]])

            else:
                #since inner matrix:
                sides=[True, True, True, True, True, True]
                a=self.add_toarrays(sides,i)
                data=np.concatenate([data, a[0]])
                row=np.concatenate([row, a[1]])
                col=np.concatenate([col, a[2]])   
                
        self.row=row
        self.col=col
        self.data=data
        return(np.vstack([data, row, col]))
        

    

            
        
        
        
        
        
