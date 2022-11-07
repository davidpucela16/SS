#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 17:15:05 2021

@author: pdavid
"""

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg
import time
from numba import njit
plt.rcParams["figure.figsize"] = (8,8)

from cylindrical_problem_assembly import *

def get_sol_coupled(Lap_operator,phi_vessel, Rv, inc_r, inc_s, in_boundary, K_effpiRvsquare, RHS_without_coupling):
    """modifies the lap_operator and the RHS in order to accomodate the incomming flux
    THE OUT BOUNDARY CONDITION IS NOT SET HERE, IT MUST COME FROM THE LAP_OPERATOR AND RHS"""
    
    RHS=RHS_without_coupling
    
    K_eff=K_effpiRvsquare/(np.pi*Rv**2)
    
    factor=K_effpiRvsquare/(np.pi*(2*Rv+inc_r)*inc_r)
    #print("q_factor{}".format(K_eff*np.pi*Rv**2))
    
    #i=inn_boundary
    i=np.arange(len(phi_vessel))
    RHS[i]=-factor*phi_vessel
    
    #WATCH OUT FOR THIS LINE!!!!!
    Lap_operator[in_boundary, in_boundary]-=factor
    
    sol=np.linalg.solve(Lap_operator, RHS)
    lenr=len(RHS)/len(phi_vessel)
    return(sol.reshape(int(lenr), len(phi_vessel)))


def get_validation_q_imposed_outerDirichlet(q_vessel, R_max, Rv,r_points,s_points, L, D):
    """The Dirichlet north are set in the next function"""
    
    obj=simplified_assembly_Laplacian_cylindrical(R_max, Rv,r_points,s_points, L, D)
    hs, hr=obj.inc_s, obj.inc_r
    r=obj.r
    s=obj.s
    Lap_operator=obj.assembly()
    
    #Set Dirichlet on the outer boundary
    ext_boundary=np.concatenate([[obj.corners[2]], obj.outer_boundary, [obj.corners[3]]])
    Lap_operator[ext_boundary,:]=0
    Lap_operator[ext_boundary,ext_boundary]=1
    
    RHS=np.zeros(Lap_operator.shape[0])
    factor=1/(np.pi*(2*Rv+hr)*hr)
    print(factor)
    i=np.arange(len(q_vessel))
    RHS[i]=-factor*q_vessel

    sol=sp.sparse.linalg.spsolve(Lap_operator, RHS)
    lenr=len(RHS)/len(q_vessel)
    
    return(sol.reshape(int(lenr), len(q_vessel)))

class get_coupled_problem_matrix():
    """This class has the functions necessary to assemble the full problem matrix without an intravascular equation"""
    def __init__(self,R_max, Rv, r_points, s_points, L,D, fine_s_points, fine_r_points):
        #CREATE GHOSTS
        hr=(R_max-Rv)/r_points
        R_max_ghosts=R_max+hr
        
        
        #There needs to be a row of ghost cells to more accurately impose the Dirichlet BC north
        k=simplified_assembly_Laplacian_cylindrical(R_max_ghosts, Rv,r_points+1,s_points, L, D)
        self.c_hs, self.c_hr=k.inc_s, k.inc_r
        self.c_r=k.r
        self.c_s=k.s
        self.Rv=Rv
        self.R_max=R_max
        
        self.a=k.assembly()
        
        self.f_hr=(R_max-Rv)/(fine_r_points)
        self.f_hs=L/(fine_s_points)
        self.f_r=np.linspace(Rv+self.f_hr/2, R_max+hr-self.f_hr/2, fine_r_points)
        self.f_s=np.linspace(self.f_hs/2,L-self.f_hs/2,fine_s_points)
        
        self.east_boundary=np.concatenate([[k.corners[1]], k.east_boundary, [k.corners[3]]])
        self.west_boundary=np.concatenate([[k.corners[0]], k.west_boundary, [k.corners[2]]])
        self.north_boundary=np.concatenate([[k.corners[2]], k.outer_boundary, [k.corners[3]]])
        self.inner_boundary=np.concatenate([[k.corners[0]],k.inner_boundary,[k.corners[1]]])
        
        self.total=k.total
        
        self.Gr=self.intGrdr(self.c_r,self.c_hr)
        
    def intGrdr(self,r,h_r):
        Rv=self.Rv
        G_array=np.array([])
        for i in r:
            a=oneD_integral(self.f_r, self.f_r*Green(self.f_r,Rv),[i-h_r/2,i+h_r/2])
            G_array=np.append(G_array,a)
        return(G_array/(r*h_r))
    
    def fine_to_coarse(self, coarse_r, coarse_s, fine_r, fine_s):
        #array that defines to which cell in the coarse arrays each of the fine positions belongs to
        position_r=np.array([]).astype(int)
        position_s=np.array([]).astype(int)
                
        for i in fine_r:
            position_r=np.append(position_r, np.argmin((i-coarse_r)**2))
        for j in fine_s:
            position_s=np.append(position_s, np.argmin((j-coarse_s)**2))
            
        return([position_r, position_s])
    

    def c_and_d_assemble(self,K_eff):
        Rv=self.Rv
        in_kernel=interp_kernel_oneD(self.c_s, self.c_hs, self.f_s, self.f_hs)
        in_kernel_data=in_kernel[0]
        in_kernel_row=in_kernel[1]
        in_kernel_col=in_kernel[2]
        
        self.c=sp.sparse.csc_matrix((in_kernel_data, (in_kernel_row, in_kernel_col)),shape=(len(self.f_s),self.total))
        
        data_d=np.ones(len(self.f_s))/(K_eff*np.pi*Rv**2)
        row_d=np.arange(len(self.f_s))
        col_d=row_d
        
        self.d=sp.sparse.csc_matrix((data_d, (row_d, col_d)), shape=(len(self.f_s),len(self.f_s)))
            


    
    def get_east_west_BC(self):
        """This function adds the boundary conditions for the left hand side of the problem"""
        #a*b*c:
        #a -> a row with two columns. Each column has a value equal to int G(r)rdr 
        #       for each of the east (for col 1), and west (for col 2) boundary cells
        #b -> boolean array that selects ony the last and the first value of an array
        #c -> the kernel that gets the first derivative of a one D function
        Gr= self.Gr
        row=np.concatenate([self.west_boundary, self.east_boundary])
        col=np.concatenate([np.zeros(len(self.west_boundary)),np.zeros(len(self.east_boundary))+1])
        data=np.concatenate([Gr,-Gr])
        
        t=self.total 
        t_network=len(self.f_s)
        
        hs_n=self.f_hs
        der_kernel=sp.sparse.csc_matrix(([-1,1,-1,1],([0,0,1,1],[0,1,t_network-2,t_network-1])), shape=(2,len(self.f_s)))/hs_n
        
        kernel_a_BC=sp.sparse.csc_matrix((data, (row, col)), shape=(t,2))*der_kernel/self.c_hs
        
        self.kernel_a_BC=kernel_a_BC
        return(kernel_a_BC)
        
    def regular_creation(self):
        """Due to the splitting method, there will appear a matter creation term for the 
        #regular term.
        [position kernel]*[2 order derivative kernel]*h_nerwork/h_s_coarse
        """
        
        position_s=self.fine_to_coarse(self.c_r, self.c_s, self.f_r, self.f_s)[1]
        
        data=np.array([])
        col=np.array([])
        row=np.array([])
        c=0
        for i in self.Gr:
            #we assemble the matrix each row at a time. Therefore all elements assembled each 
            #iteration have the same G(r)
            data=np.concatenate([data,np.ones(len(position_s))*i*self.f_hs/self.c_hs])
            col=np.concatenate([col,np.arange(len(position_s))]) #each 
            row=np.concatenate([row,position_s+c*len(self.c_s)])
            c+=1

        #This is the "position kernel"
        kernel=sp.sparse.csc_matrix((data, (row,col)),shape=(self.total, len(self.f_s)))
        self.kernel_creation=kernel
        sec_der_kernel=get_sec_der(self.f_s, self.f_hs)
        return(kernel*sec_der_kernel)
    
    def assemble_full_problem(self,K_eff):
        Rv=self.Rv
        R_max=self.R_max
        self.c_and_d_assemble(K_eff)
        c=self.c
        d=self.d
        
        b=self.get_east_west_BC()+self.regular_creation()
        #set_Dirichlet_north
        
        eq=self.fine_to_coarse(self.c_r, self.c_s, self.f_r, self.f_s)[1]
        G=Green(self.R_max,Rv)
        for i in self.north_boundary:
            pos_s=i%len(self.c_s)
            pos_vessel=np.where(eq==pos_s)[0]
            b[i,pos_vessel]=b[i,pos_vessel].toarray()-2*G*self.R_max*self.f_hs/(self.c_r[-1]*self.c_hr**2*self.c_hs)
            
            
        print(np.around(b.toarray(), decimals=1))
            
        a=self.a
        a[self.north_boundary, self.north_boundary]-=2*R_max/(self.c_r[-1]*self.c_hr**2)
                    
        M=sp.sparse.hstack([a,b])
        M=sp.sparse.vstack([M,sp.sparse.hstack([c,d])])
    
        return(M)
    
    def assemble_problem_Dirichlet_north_ghost(self,K_eff):
        Rv=self.Rv
        self.c_and_d_assemble(K_eff)
        c=self.c
        d=self.d
        
        b=self.get_east_west_BC()+self.regular_creation()
        #set_Dirichlet_north
        
        eq=self.fine_to_coarse(self.c_r, self.c_s, self.f_r, self.f_s)[1]
        G=Green(self.R_max,Rv)
        b[self.north_boundary,:]=0
        for i in self.north_boundary:
            pos_s=i%len(self.c_s)
            pos_vessel=np.where(eq==pos_s)[0]
            b[i,pos_vessel]=2*G/len(pos_vessel)

            
        print(np.around(b.toarray(), decimals=1))
            
        a=self.a
        a[self.north_boundary,:]=0
        a[self.north_boundary, self.north_boundary]=1
        a[self.north_boundary, self.north_boundary-len(self.c_s)]=1
                    
        M=sp.sparse.hstack([a,b])
        M=sp.sparse.vstack([M,sp.sparse.hstack([c,d])])
    
        return(M)

def full_2D_linear_interpolation(f_r,f_s, phi_solution, c_r, c_s, c_hs, c_hr):
    interp=np.empty((len(f_r), len(f_s)))
    c=0
    for i in f_r:
        d=0
        for j in f_s:
            value=linear_interpolation(i,j, phi_solution, c_r, c_s, c_hs, c_hr)
            interp[c,d]=value
            d+=1
        c+=1
    return(interp)

def linear_interpolation(f_r,f_s, v_solution, c_r, c_s, c_hs, c_hr):
    """linear interpolation for the 2D solution, 
    with nearest neighbour interpolation for the boundaries"""
    n_s=np.where(np.abs(c_s-f_s)<c_hs)[0]
    n_r=np.where(np.abs(c_r-f_r)<c_hr)[0]
    
    total_distance=0
    dist=np.array([])
    c=0
    d=0
    for i in n_s:
        for j in n_r:
            dist=np.append(dist,np.sqrt((f_r-c_r[j])**2+(f_s-c_s[i])**2) )
            c+=1
    tot_dist=np.sum(dist)
    tot_inv=np.sum(1/dist)
    if len(dist)==1:

        value=v_solution[n_r,n_s]
    else:
        value=0
        d=c=0
        for i in n_s:
            for j in n_r:
                weight=1/(dist[c]*tot_inv)
                value+=weight*v_solution[j,i]
                c+=1
    return(value)

#nearest neighbour interpolation 2D
def NN_interpolation(twoDfield_to_interpolate, x_coarse, y_coarse, x_fine, y_fine):
    if len(twoDfield_to_interpolate.shape)>1:
        phi=np.ndarray.flatten(twoDfield_to_interpolate)
    else:
        phi=twoDfield_to_interpolate
    interpolation=np.empty([len(y_fine),len(x_fine)])
    for i in range(len(x_fine)):
        for j in range(len(y_fine)):
            fx, fy=x_fine[i], y_fine[j]
            interpolation[j,i]=phi[look_up_nearest(fx,fy,x_coarse, y_coarse)]
    
    return(interpolation)

@njit
def look_up_nearest(fx, fy,cx,cy):
    dx=cx-fx
    dy=cy-fy
    position=np.argmin(dy**2)*len(cx)+np.argmin(dx**2)
    return(position)


def analyt_intGrdr(rk, hr, Rv):
    r=np.array([rk-hr/2,rk+hr/2])
    factor=r**2*(np.log(r/Rv)-0.5)
    integral=-(factor[1]-factor[0])/(4*np.pi*D)
    return(integral)

def oneD_integral(x, function, xlim):
    h=(np.max(x)-np.min(x))/(len(x)-1)
    indices_x=(x>=xlim[0]) & (x<xlim[1])
    integral=np.sum(function[indices_x])*h
    return(integral)

def test_integral_GRDR(r, hr, Rv):
    analyt=np.array([])
    num=np.array([])
    for i in r:
        analyt=np.append(analyt,analyt_intGrdr(i, hr, Rv))
        num=np.append(num, oneD_integral(k.f_r,k.f_r*Green(k.f_r, Rv), [i-hr/2,i+hr/2]))
    G_array=np.array([])
    for i in r:
        a=oneD_integral(k.f_r, k.f_r*Green(k.f_r,Rv),[i-hr/2,i+hr/2])
        G_array=np.append(G_array,a)
    print(num)
    print(analyt)
    return(G_array)

def interp_kernel_oneD(coarse,h_coarse, fine, h_fine):
    data=np.array([])
    row=np.array([])
    col=np.array([])
    c=0
    for i in fine:
        ind=np.argwhere((coarse-i)**2<h_coarse**2)
        ind=ind.reshape(len(ind))
        if len(ind)==1:
            row=np.append(row,c)
            col=np.append(col,ind)
            data=np.append(data,1)
            
        elif len(ind)==2:
        
            d1=np.abs(coarse[ind[0]]-i)
            d2=np.abs(coarse[ind[1]]-i)
            
            row=np.append(row,[c,c])
            col=np.append(col,ind)
            
            k=1/np.sum(1/np.array([d1,d2]))
            
            w1=k/d1
            w2=k/d2
            data=np.append(data,[w1,w2])
    
        c+=1
    return(np.vstack([data,row, col]))




