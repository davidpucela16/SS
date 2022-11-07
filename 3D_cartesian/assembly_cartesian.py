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
from numba import njit
from sol_split_3D import * #This is necessary just for the interpolation functions

@njit
def array_to_tuple(i, x, y ,z):
    """from the position in the 1D array of the domain (x,y,z) 
    OUTPUTS the position within each axis as an array of int16"""
    zPos=i//(len(x)*len(y))
    yPos=(i%(len(x)*len(y)))//len(x)
    xPos=i%len(x)
    
    return(np.array([xPos, yPos, zPos]).astype(np.int16))

@njit
def tuple_to_array(pos_tuple, x,y,z):
    xPos, yPos, zPos=pos_tuple
    return(zPos*(len(x)*len(y))+yPos*len(x)+xPos)
    

@njit
def position_to_coordinates(i,  x, y, z):
    xPos, yPos, zPos=array_to_tuple(i, x,y,z)    
    return(np.array([x[xPos], y[yPos], z[zPos]],dtype=np.float64))
    
@njit
def coordinates_to_position(coords, x, y, z):
    xPos=np.argmin(np.abs(x-coords[0]))
    yPos=np.argmin(np.abs(y-coords[1]))
    zPos=np.argmin(np.abs(z-coords[2]))
    
    Points=np.array([len(x), len(y), len(z)])
    
    f_step=Points[0]
    s_step=Points[0]*Points[1]
    
    return(int(xPos+yPos*f_step+zPos*s_step))

#@njit
def get_r_s(X,p0,lamb):
    """it returns the point s of the array, and the vector r.
    The vector r represents the shortest vector that goes from the vessel described by 
    p0 and lamb to the point in space X
    
    Careful with the return array:
        First three positions are the vector r
        Last position is the point s"""
    
    s=np.sum(lamb*(X-p0)) #scalar product (projection of X onto vessel)
    r=X-p0-lamb*s
    return(np.append(r,s))

def get_coeffs(sides,D,h):
       """From the sides that are given, the function calculates the stensil. It will return a 0 if the 
       cell is a boundary
       
       This function does not admit non-homogeneous diffusion coefficient nor varying cell size (h)"""
       hx,hy,hz=h
       north, south, east, west, forw, back=sides
       coeff_cent, coeff_north, coeff_south, coeff_east, coeff_west, coeff_forw, coeff_back=np.zeros(7)
       if north:
           coeff_north=D/hz**2
           coeff_cent-=D/hz**2
       if south:
           coeff_south=D/hz**2
           coeff_cent-=D/hz**2
       if east:
           coeff_east=D/hy**2
           coeff_cent-=D/hy**2
       if west:
           coeff_west=D/hy**2
           coeff_cent-=D/hy**2
       if forw:
           coeff_forw=D/hx**2
           coeff_cent-=D/hx**2
       if back:
           coeff_back=D/hx**2
           coeff_cent-=D/hx**2
       return(np.array([coeff_cent,coeff_north, coeff_south, coeff_east, coeff_west, coeff_forw, coeff_back ]))

def decode_boundary(code):
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

class Lap_cartesian():
    def __init__(self, D,L, h):
        """In the init function, the main calculation done (besides the initialization of the variables)
        is the calculation of the boundary nodes and its codes through the function of encode_boundary. Therefore, 
        the calling of this function already will take some time since the function encode_boundary has to iterate
        through each cell of the domain"""
        
        self.data=np.array([])
        self.row=np.array([])
        self.col=np.array([])
        
        self.D=D
        self.Lx, self.Ly, self.Lz=L
        self.h=h
        self.hx, self.hy, self.hz=h
        points=np.array(L/h, dtype=int)
        self.xPoints, self.yPoints, self.zPoints=points
        self.total=self.xPoints*self.yPoints*self.zPoints
        
        
        self.f_step=self.xPoints
        self.s_step=self.xPoints*self.yPoints
        
        self.boundary=self.encode_boundary(self.total).astype(int)
    
        self.x=np.linspace(0+self.hx/2,self.Lx-self.hx/2,self.xPoints)
        self.y=np.linspace(0+self.hy/2,self.Ly-self.hy/2,self.yPoints)
        self.z=np.linspace(0+self.hz/2,self.Lz-self.hz/2,self.zPoints)

    
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
        coe=get_coeffs(sides,self.D,self.h)
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
    
    
    def assembly(self):
        """This function performs the assembly of the laplacian operator for a 3D cartesian domain.
        The boundary nodes are taken as flux BCs, so if need be to set Dirichlet, it would be very easy
        setting the boundary rows (which have been already calculated in self.boundary[0]), to zero and inserting
        the Dirichlet BC"""

        data=np.array([])
        row=np.array([])
        col=np.array([])
        boundaries=self.boundary[0]
        codes=self.boundary[1]
        for i in range(self.total):
            self.i=i
            if i in boundaries:
                position_in_array=np.where(boundaries==i)[0]
                sides=decode_boundary(codes[position_in_array])
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
        

    
class finite_source():
    """This is meant to create the source object. In the class there will be stored geometrical information 
    regarding the vessel and the tissue
    
    THIS CLASS WILL HAVE TO BE MODIFIED TO INCLUDE MULTIPLE VESSELS INSIDE OF A BLOCK
    """
    def __init__(self, hs, init, end, constant_Rv):
        """So far the constant_Rv is a constant. It does not allow for diameter variations within the network.
        It will have to be modified to include a diameter per vessel, at least."""
        self.init=init
        self.end=end
        
        self.Ls=np.sqrt(np.sum((end-init)**2))
        self.lamb=(end-init)/self.Ls
        self.s=np.arange(hs/2, self.Ls, hs)
        self.hs=hs
        
        self.Rv=constant_Rv
        
    def get_coupling_cells(self, tissue_x, tissue_y, tissue_z):
        
        """This funtion attempts to link each vessel cell with the containing tissue cell
        It will have to be called from the script.
        
        OUTPUT -> array of tissue cells that correspond to each of the network cells. 
        OUTPUT.shape=the length of the network (one coupled tissue cell per network cell)"""
        x, y, z=tissue_x, tissue_y, tissue_z
        array=np.array([], dtype=int) #array containing the tissue cell of each of the vessel cell. It will be therefore, 
                           #the same length as s since it relates each of the s cells.

        for i in self.s:
            pos_s=self.init+self.lamb*i #cartesian position of the center of the cell
            array=np.append(array, coordinates_to_position(pos_s,x,y,z,))
        self.coup_cells=array      
        return(array)
    
    def get_s_coordinates(self):
        """This function will have to be called from the Script.
        It initializes the arrays of coordinates for the center init and end of each network cell
        
        OUTPUT: the array of centers of network FV cells
        """
        self.s_coords=np.outer(self.s,self.lamb)+np.outer(np.ones(len(self.s)),self.init)
    
        self.init_array=np.outer(self.s-self.hs/2, self.lamb)+np.outer(np.ones(len(self.s)),self.init)
        self.end_array=np.outer(self.s+self.hs/2, self.lamb)+np.outer(np.ones(len(self.s)),self.init)
        
        return(self.s_coords)   
            
class linear_coup_BC():
    def __init__(self,Lap_object, source_object):
        self.h=Lap_object.h
        self.boundary_array_ID=Lap_object.boundary[0]
        self.boundary_array_code=Lap_object.boundary[1]
    
    def set_TPFA_Dirichlet(self,Dirichlet,operator,  h, boundary_array_code, RHS):
        """The 6 boundaries in order north, south, east, west, front, back
        
        boundary_array_code contains the array from the Lap_cartesian class; therefore contains 
        cell's ID and their codes
        
        This function will admit 6 values for the Dirichlet condition so far. One for boundary;
        Therefore Dirichlet.shape=6
        
        Make sure to input the operator as a copy or it will be modified for some reason"""
        
        hx, hy, hz=h
        c=0
        for i in self.boundary_array_ID:
            code=self.boundary_array_code[c]
            
            [north, south, east, west, front, back]=decode_boundary(code)
            C_n=0 if north else (hz/2)**-2
            C_s=0 if south else (hz/2)**-2
            C_e=0 if east else (hy/2)**-2
            C_w=0 if west else (hy/2)**-2
            C_f=0 if front else (hx/2)**-2
            C_b=0 if back else (hx/2)**-2
            
            C=np.array([C_n, C_s, C_e, C_w, C_f, C_b])
            
            operator[i,i]-=np.sum(C)
            RHS[i]-=np.dot(C,Dirichlet)
    
            c+=1
        return(RHS, operator)
        
def set_TPFA_Dirichlet(Dirichlet,operator,  h, boundary_array_code, RHS):
    """The 6 boundaries in order north, south, east, west, front, back
    
    boundary_array_code contains the array from the Lap_cartesian class; therefore contains 
    cell's ID and their codes
    
    This function will admit 6 values for the Dirichlet condition so far. One for boundary;
    Therefore Dirichlet.shape=6
    
    Make sure to input the operator as a copy or it will be modified for some reason"""
    hx, hy, hz=h
    c=0
    for i in boundary_array_code[0,:]:
        code=boundary_array_code[1,c]
        
        
        [north, south, east, west, front, back]=decode_boundary(code)
        C_n=0 if north else (hz/2)**-2
        C_s=0 if south else (hz/2)**-2
        C_e=0 if east else (hy/2)**-2
        C_w=0 if west else (hy/2)**-2
        C_f=0 if front else (hx/2)**-2
        C_b=0 if back else (hx/2)**-2
        
        C=np.array([C_n, C_s, C_e, C_w, C_f, C_b])
        
        operator[i,i]-=np.sum(C)
        RHS[i]-=np.dot(C,Dirichlet)

        c+=1
    return(RHS, operator)
    
    
def set_coupled_FV_system(lap_operator, phi_vessel, coup_cells, RHS, C_0, hs,V_cell):
    """It changes the lap_operator and the RHS to include the contribution from phi_vessel
    
    DOES NOT INCLUDE A COUPLING MODEL"""
    f=C_0*hs/V_cell
    for i in range(len(phi_vessel)):
        j=coup_cells[i]
        lap_operator[j,j]-=f
        RHS[j]-=f*phi_vessel[i]
    return(RHS, lap_operator)
    