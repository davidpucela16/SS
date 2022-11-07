#!/usr/bin/eanv python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:59:09 2021

@author: pdavid
"""
import numpy as np
from numba import njit
import assembly_cartesian as ac
import numba
import scipy as sp
from scipy import sparse



@njit
def get_neighbours(x,y,z, coordinate_list,R):
    c=coordinate_list
    pos_x=np.where((x<(c[0]+R)) & (x>(c[0]-R)))[0]
    pos_y=np.where((y<(c[1]+R)) & (y>(c[1]-R)))[0]
    pos_z=np.where((z<(c[2]+R)) & (z>(c[2]-R)))[0]
    print(pos_x, pos_y, pos_z)
    
    #the array of positions to interpolate between:
    p=np.zeros((0))
    
    for i in pos_x:
        for j in pos_y:
            for k in pos_z:
                p=np.append(p, int(ac.tuple_to_array(np.array([i,j,k]), x,y,z)))

    return(p.astype(np.int32))
    
def get_coeffs(neigh, dist, tolerance):
    if np.any(dist<tolerance):
        print("kakota")
        pos=np.where(dist<tolerance)[0][0]
        a=np.array([pos, 1])
        
    else:
        alpha=1/np.sum(1/dist)
        coeffs=alpha/dist
        a=np.vstack((neigh.astype(np.float64), coeffs)).T
    return(a)

@njit
def get_dist_array(x_coords, x,y,z, R):
    #x_coords in np.array form 
    neigh=get_neighbours(x, y, z, x_coords, R).astype(np.int32)
    print(neigh)
    N=np.shape(neigh)[0]
    array=np.empty((2,N), dtype=np.float64)
    dist=np.empty((N), dtype=np.float64)
    for i in np.arange(N):
        dist[i]=np.float64(np.linalg.norm(x_coords-ac.position_to_coordinates(neigh[i],x,y,z)))
    return(np.vstack((dist, neigh.astype(np.float64))))

def linear_interpolation(x_coords, x, y, z, R):
    d=get_dist_array(x_coords, x,y,z, R)

    return(get_coeffs(d[1].astype(np.int32), d[0], 0.00001))

    
    
@njit
def useless():

    a=np.where(np.arange(10)>4)[0][0]
    return(a)

def sing_term(init, end, x, Rv):
    tau=(end-init)/np.linalg.norm(end-init)
    L=np.linalg.norm(end-init)
    a=x-init
    b=x-end
    s=np.sum(a*tau)
    d=np.linalg.norm(a-tau*np.sum(a*tau))
    G_term=np.sqrt(L**2/4+Rv**2)
    
    if d<Rv:
        if s<0 or s>L:
            C_0=s-L if s>L else -s
            G=np.log((L+C_0)/C_0)
        else:
            G=np.log((G_term+L/2)/(G_term-L/2))
        
    else:
        rb=np.linalg.norm(b)
        ra=np.linalg.norm(a)
        G=np.log((rb+L-np.dot(a,tau))/(ra-np.dot(a,tau)))
    return(G/(4*np.pi))
    
st_jit=njit()(sing_term)

    
@njit
def cell_single_term(init_array, end_array, x_coord, Rv):
    """Returns the kernel to multiply the q_array (of the vessel) that will 
    provide the singular term at the x_coord"""
    x=x_coord
    array=np.zeros(init_array.shape[0])
    for i in np.arange(init_array.shape[0]):
        #"Integrates" through the network to get the contribution from each 
        array[i]=st_jit(init_array[i], end_array[i], x, Rv)
    return(array)
    
class assemble_sol_split(): 
    """This class is meant to assemble the problem as a solution splitted problem 
    
    It is used in place of the assembly_cartesian.Lap_assembly class that would build a standard 
    FV problem
    
    In the sol split problem, there is a coupling between the source and the FV problem
    due to the dependency everywhere from the singular term"""
    def __init__(self, phi_vessel, L,h, finite_source_object,K_eff, D):
        """init function here onlin initializes the variables and the 
        ASSEMBLY_CARTESIAN.LAP_CARTESIAN TO CREATE THE LAPLACIAN OPERATOR
        """
        self.phi_vessel=phi_vessel
        self.K_eff=K_eff
        self.s_coords=finite_source_object.get_s_coordinates()
        self.init_array=finite_source_object.init_array
        self.end_array=finite_source_object.end_array

        self.Rv=finite_source_object.Rv   
        
        self.op=ac.Lap_cartesian(D, L,h)
        self.op.assembly()
        self.lap=sp.sparse.csc_matrix((self.op.data, (self.op.row, self.op.col)), shape=(self.op.total, self.op.total))
        self.total_tissue=self.op.total
        
        self.x=self.op.x
        self.y=self.op.y
        self.z=self.op.z
        
        self.data=np.array([])
        self.row=np.array([])
        self.col=np.array([])
        
    def C_assembly_test(self):
        C_0=self.K_eff*np.pi*self.Rv**2
        C=np.zeros([len(self.phi_vessel), len(self.phi_vessel)])
        for i in range(len(self.phi_vessel)):
            
            C[i,i]=st_jit(self.init_array[i], self.end_array[i], self.s_coords[i], self.Rv)
            for j in np.delete(np.arange(len(self.phi_vessel)), i):
                C[i,j]=sing_term(self.init_array[j], self.end_array[j], self.s_coords[i], self.Rv)
        return(C)
        
    def D_assembly(self):
        C_0=self.K_eff*np.pi*self.Rv**2
        D=np.zeros([len(self.phi_vessel), len(self.phi_vessel)])
        for i in range(len(self.phi_vessel)):
            
            D[i,i]=1/C_0+st_jit(self.init_array[i], self.end_array[i], self.s_coords[i], self.Rv)
            for j in np.delete(np.arange(len(self.phi_vessel)), i):
                #Maybe it is worth to do this loop in a different njit function to accelerate and to mimic 
                #a bit the creation of B-BCs
                D[i,j]=sing_term(self.init_array[j], self.end_array[j], self.s_coords[i], self.Rv)
        self.D_sing_term=D
        return(D)
                
    def C_assembly(self, Ra):
        """Assembles the C matrix which is mainly a an interpolation withing the closest cells given by 
        the function linear_interpolation.
        
        The interpolated values are meant to represent the values of the regular term at the level of 
        the vessel wall"""
        data=np.array([])
        row=np.array([])
        col=np.array([])
        for i in range(len(self.phi_vessel)):
            a=linear_interpolation(self.s_coords[i], self.op.x, self.op.y, self.op.z, Ra)
            data=np.concatenate([data, a[:,1]])
            col=np.concatenate([col, a[:,0]])
            row=np.concatenate([row, np.zeros(len(a[:,0]))+i])
        self.C_initial=sp.sparse.csc_matrix((data, (row, col)), shape=(len(self.phi_vessel), self.total_tissue))
        return(np.vstack((data, row, col)))
    
    def B_Dirichlet_assembly(self):
        B=np.zeros([self.lap.shape[0], len(self.s_coords)])
        
        for i in self.op.boundary[0]:
            b_coord=ac.position_to_coordinates(i,  self.op.x, self.op.y, self.op.z)
            B[i,:]=cell_single_term(self.init_array, self.end_array,b_coord, self.Rv)
        return(B)
            


def set_TPFA_Dirichlet_SS(Dirichlet,operator,  h, boundary_array_code, RHS, x, y, z, init_array, end_array, Rv):
    """The 6 boundaries in order north, south, east, west, front, back
    
    boundary_array_code contains the array from the Lap_cartesian class; therefore contains 
    cell's ID and their codes
    
    This function will admit 6 values for the Dirichlet condition so far. One for boundary;
    Therefore Dirichlet.shape=6"""
    hx, hy, hz=h
    c=0
    B=np.zeros((operator.shape[0], init_array.shape[0]))
    for i in boundary_array_code[0,:]:
        code=boundary_array_code[1,c]
        
        
        [north, south, east, west, front, back]=ac.decode_boundary(code)
        C_n=0 if north else (hz**-2)/2
        C_s=0 if south else (hz**-2)/2
        C_e=0 if east else (hy**-2)/2
        C_w=0 if west else (hy**-2)/2
        C_f=0 if front else (hx**-2)/2
        C_b=0 if back else (hx**-2)/2
        
        C=np.array([C_n, C_s, C_e, C_w, C_f, C_b])
        
        operator[i,i]-=np.sum(C)
        RHS[i]-=np.dot(C,Dirichlet)
        
        
        b_coord=ac.position_to_coordinates(i,  x, y, z)
        B[i,:]=-np.sum(C)*cell_single_term(init_array, end_array,b_coord, Rv)
        c+=1
        
    return(RHS, operator, B)
    

    
    
    