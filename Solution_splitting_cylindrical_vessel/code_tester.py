#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:56:44 2021

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
from scipy import interpolate

from numba import njit


plt.rcParams["figure.figsize"] = (12,12)
plt.rcParams.update({'font.size': 26})
from cylindrical_problem_assembly import *
from cylindrical_coupling_assembly import *

# =============================================================================
# class get_coupled_problem_matrix():
#     def __init__(self,R_max, Rv, r_points, s_points, L,D, fine_s_points, fine_r_points):
#         #CREATE GHOSTS
#         hr=(R_max-Rv)/r_points
#         R_max_ghosts=R_max+hr
#         
#         
#         #There needs to be a row of ghost cells to more accurately impose the Dirichlet BC north
#         k=simplified_assembly_Laplacian_cylindrical(R_max_ghosts, Rv,r_points+1,s_points, L, D)
#         self.c_hs, self.c_hr=k.inc_s, k.inc_r
#         self.c_r=k.r
#         self.c_s=k.s
#         self.Rv=Rv
#         self.R_max=R_max
#         
#         self.a=k.assembly()
#         
#         self.f_hr=(R_max-Rv)/(fine_r_points)
#         self.f_hs=L/(fine_s_points)
#         self.f_r=np.linspace(Rv+self.f_hr/2, R_max+hr-self.f_hr/2, fine_r_points)
#         self.f_s=np.linspace(self.f_hs/2,L-self.f_hs/2,fine_s_points)
#         
#         self.east_boundary=np.concatenate([[k.corners[1]], k.east_boundary, [k.corners[3]]])
#         self.west_boundary=np.concatenate([[k.corners[0]], k.west_boundary, [k.corners[2]]])
#         self.north_boundary=np.concatenate([[k.corners[2]], k.outer_boundary, [k.corners[3]]])
#         self.inner_boundary=np.concatenate([[k.corners[0]],k.inner_boundary,[k.corners[1]]])
#         
#         self.total=k.total
#         
#         self.Gr=self.intGrdr(self.c_r,self.c_hr)
#         
#     def intGrdr(self,r,h_r):
#         Rv=self.Rv
#         G_array=np.array([])
#         for i in r:
#             a=oneD_integral(self.f_r, self.f_r*Green(self.f_r,Rv),[i-h_r/2,i+h_r/2])
#             G_array=np.append(G_array,a)
#         return(G_array/(r*h_r))
#     
#     def fine_to_coarse(self, coarse_r, coarse_s, fine_r, fine_s):
#         #array that defines to which cell in the coarse arrays each of the fine positions belongs to
#         position_r=np.array([]).astype(int)
#         position_s=np.array([]).astype(int)
#                 
#         for i in fine_r:
#             position_r=np.append(position_r, np.argmin((i-coarse_r)**2))
#         for j in fine_s:
#             position_s=np.append(position_s, np.argmin((j-coarse_s)**2))
#             
#         return([position_r, position_s])
#     
# 
#     def c_and_d_assemble(self,K_eff):
#         self.Rv=Rv
#         in_kernel=interp_kernel_oneD(self.c_s, self.c_hs, self.f_s, self.f_hs)
#         in_kernel_data=in_kernel[0]*K_eff*np.pi*Rv**2
#         in_kernel_row=in_kernel[1]
#         in_kernel_col=in_kernel[2]
#         
#         self.c=sp.sparse.csc_matrix((in_kernel_data, (in_kernel_row, in_kernel_col)),shape=(len(self.f_s),self.total))
#         
#         data_d=np.ones(len(self.f_s))/(K_eff*np.pi*Rv**2)
#         row_d=np.arange(len(self.f_s))
#         col_d=row_d
#         
#         self.d=sp.sparse.csc_matrix((data_d, (row_d, col_d)), shape=(len(self.f_s),len(self.f_s)))
#     
# 
#     
#     def get_east_west_BC(self):
#         """This function adds the boundary conditions for the left hand side of the problem"""
#         #a*b*c:
#         #a -> a row with two columns. Each column has a value equal to int G(r)rdr 
#         #       for each of the east (for col 1), and west (for col 2) boundary cells
#         #b -> boolean array that selects ony the last and the first value of an array
#         #c -> the kernel that gets the first derivative of a one D function
#         Gr= self.Gr
#         row=np.concatenate([self.west_boundary, self.east_boundary])
#         col=np.concatenate([np.zeros(len(self.west_boundary)),np.zeros(len(self.east_boundary))+1])
#         data=np.concatenate([Gr,-Gr])
#         
#         t=self.total 
#         t_network=len(self.f_s)
#         
#         hs_n=self.f_hs
#         der_kernel=sp.sparse.csc_matrix(([-1,1,-1,1],([0,0,1,1],[0,1,t_network-2,t_network-1])), shape=(2,len(self.f_s)))/hs_n
#         
#         kernel_a_BC=sp.sparse.csc_matrix((data, (row, col)), shape=(t,2))*der_kernel/self.c_hs
#         
#         self.kernel_a_BC=kernel_a_BC
#         return(kernel_a_BC)
#         
#     def regular_creation(self):
#         """Due to the splitting method, there will appear a matter creation term for the 
#         #regular term.
#         [position kernel]*[2 order derivative kernel]*h_nerwork/h_s_coarse
#         """
#         
#         position_s=self.fine_to_coarse(self.c_r, self.c_s, self.f_r, self.f_s)[1]
#         
#         data=np.array([])
#         col=np.array([])
#         row=np.array([])
#         c=0
#         for i in self.Gr:
#             #we assemble the matrix each row at a time. Therefore all elements assembled each 
#             #iteration have the same G(r)
#             data=np.concatenate([data,np.ones(len(position_s))*i*self.f_hs/self.c_hs])
#             col=np.concatenate([col,np.arange(len(position_s))]) #each 
#             row=np.concatenate([row,position_s+c*len(self.c_s)])
#             c+=1
# 
#         #This is the "position kernel"
#         kernel=sp.sparse.csc_matrix((data, (row,col)),shape=(self.total, len(self.f_s)))
#         self.kernel_creation=kernel
#         sec_der_kernel=get_sec_der(self.f_s, self.f_hs)
#         return(kernel*sec_der_kernel)
#     
#     def assemble_full_problem(self,K_eff):
#         Rv=self.Rv
#         k.c_and_d_assemble(K_eff)
#         c=k.c
#         d=k.d
#         
#         b=self.get_east_west_BC()+self.regular_creation()
#         #set_Dirichlet_north
#         
#         eq=self.fine_to_coarse(self.c_r, self.c_s, self.f_r, self.f_s)[1]
#         G=Green(self.R_max,Rv)
#         for i in self.north_boundary:
#             pos_s=i%len(self.c_s)
#             pos_vessel=np.where(eq==pos_s)[0]
#             b[i,pos_vessel]=b[i,pos_vessel].toarray()-2*G*self.R_max*self.f_hs/(self.c_r[-1]*self.c_hr**2*self.c_hs)
#             
#             print("disc ratio",self.f_hs/self.c_hs*G)
#             print("pos_vessel", pos_vessel[0])
#             
#         print(np.around(b.toarray(), decimals=1))
#             
#         a=self.a
#         a[self.north_boundary, self.north_boundary]-=2*R_max/(self.c_r[-1]*self.c_hr**2)
#                     
#         M=sp.sparse.hstack([a,b])
#         M=sp.sparse.vstack([M,sp.sparse.hstack([c,d])])
#     
#         return(M)
#     
#     def assemble_problem_Dirichlet_north_ghost(self,K_eff):
#         Rv=self.Rv
#         k.c_and_d_assemble(K_eff)
#         c=k.c
#         d=k.d
#         
#         b=self.get_east_west_BC()+self.regular_creation()
#         #set_Dirichlet_north
#         
#         eq=self.fine_to_coarse(self.c_r, self.c_s, self.f_r, self.f_s)[1]
#         G=Green(self.R_max,Rv)
#         b[self.north_boundary,:]=0
#         for i in self.north_boundary:
#             pos_s=i%len(self.c_s)
#             pos_vessel=np.where(eq==pos_s)[0]
#             b[i,pos_vessel]=2*G/len(pos_vessel)
# 
#             print("pos_vessel", pos_vessel[0])
#             
#         print(np.around(b.toarray(), decimals=1))
#             
#         a=self.a
#         a[self.north_boundary,:]=0
#         a[self.north_boundary, self.north_boundary]=1
#         a[self.north_boundary, self.north_boundary-len(self.c_s)]=1
#                     
#         M=sp.sparse.hstack([a,b])
#         M=sp.sparse.vstack([M,sp.sparse.hstack([c,d])])
#     
#         return(M)
# 
# def interp_kernel_oneD(coarse,h_coarse, fine, h_fine):
#     data=np.array([])
#     row=np.array([])
#     col=np.array([])
#     c=0
#     for i in fine:
#         ind=np.argwhere((coarse-i)**2<h_coarse**2)
#         ind=ind.reshape(len(ind))
#         if len(ind)==1:
#             row=np.append(row,c)
#             col=np.append(col,ind)
#             data=np.append(data,1)
#             
#         elif len(ind)==2:
#         
#             d1=np.abs(coarse[ind[0]]-i)
#             d2=np.abs(coarse[ind[1]]-i)
#             
#             row=np.append(row,[c,c])
#             col=np.append(col,ind)
#             
#             k=1/np.sum(1/np.array([d1,d2]))
#             
#             w1=k/d1
#             w2=k/d2
#             data=np.append(data,[w1,w2])
#     
#         c+=1
#     return(np.vstack([data,row, col]))
# 
# 
# def interpolate(f_r,f_s, v_solution, c_r, c_s, c_hs, c_hr):
#     n_s=np.where(np.abs(c_s-f_s)<c_hs)[0]
#     n_r=np.where(np.abs(c_r-f_r)<c_hr)[0]
#     
#     total_distance=0
#     dist=np.array([])
#     c=0
#     d=0
#     for i in n_s:
#         for j in n_r:
#             dist=np.append(dist,np.sqrt((f_r-c_r[j])**2+(f_s-c_s[i])**2) )
#             c+=1
#     tot_dist=np.sum(dist)
#     tot_inv=np.sum(1/dist)
#     print("n_s",n_s)
#     print("n_r",n_r)
#     print("tot_dist", dist)
#     if len(dist)==1:
#         print("corner")
# 
#         value=v_solution[n_r,n_s]
#     else:
#         value=0
#         d=c=0
#         for i in n_s:
#             for j in n_r:
#                 weight=1/(dist[c]*tot_inv)
#                 print("weight",weight)
#                 value+=weight*v_solution[j,i]
#                 c+=1
#     return(value)
# 
# #nearest neighbour interpolation 2D
# def NN_interpolation(twoDfield_to_interpolate, x_coarse, y_coarse, x_fine, y_fine):
#     if len(twoDfield_to_interpolate.shape)>1:
#         phi=np.ndarray.flatten(twoDfield_to_interpolate)
#     else:
#         phi=twoDfield_to_interpolate
#     interpolation=np.empty([len(y_fine),len(x_fine)])
#     for i in range(len(x_fine)):
#         for j in range(len(y_fine)):
#             fx, fy=x_fine[i], y_fine[j]
#             interpolation[j,i]=phi[look_up_nearest(fx,fy,x_coarse, y_coarse)]
#     
#     return(interpolation)
# 
# @njit
# def look_up_nearest(fx, fy,cx,cy):
#     dx=cx-fx
#     dy=cy-fy
#     position=np.argmin(dy**2)*len(cx)+np.argmin(dx**2)
#     return(position)
# 
# 
# def analyt_intGrdr(rk, hr, Rv):
#     r=np.array([rk-hr/2,rk+hr/2])
#     factor=r**2*(np.log(r/Rv)-0.5)
#     integral=-(factor[1]-factor[0])/(4*np.pi*D)
#     return(integral)
# 
# def oneD_integral(x, function, xlim):
#     h=(np.max(x)-np.min(x))/(len(x)-1)
#     indices_x=(x>=xlim[0]) & (x<xlim[1])
#     integral=np.sum(function[indices_x])*h
#     return(integral)
# 
# def test_integral_GRDR(r, hr, Rv):
#     analyt=np.array([])
#     num=np.array([])
#     for i in r:
#         analyt=np.append(analyt,analyt_intGrdr(i, hr, Rv))
#         num=np.append(num, oneD_integral(k.f_r,k.f_r*Green(k.f_r, Rv), [i-hr/2,i+hr/2]))
#     G_array=np.array([])
#     for i in r:
#         a=oneD_integral(k.f_r, k.f_r*Green(k.f_r,Rv),[i-hr/2,i+hr/2])
#         G_array=np.append(G_array,a)
#     print(num)
#     print(analyt)
#     return(G_array)
# =============================================================================
        
        
R_max=1
L=1
Rv=0.05
fine_s_points=200
fine_r_points=200
D=1
K_eff=1/(Rv**2*np.pi)
coarse_s_points=5
coarse_r_points=5

k=get_coupled_problem_matrix(R_max, Rv, coarse_r_points, coarse_s_points, L,D, fine_s_points, fine_r_points)
M=k.assemble_full_problem(K_eff)
M_ghosts=k.assemble_problem_Dirichlet_north_ghost(K_eff)

phi_vessel=np.cos(k.f_s/(L*0.7))

#Flux BC
s=M.shape
IC=np.zeros(s[0])
IC[-len(phi_vessel):]=phi_vessel
v=sp.sparse.linalg.spsolve(M,IC)
q=v[-len(phi_vessel):]
plt.plot(q)
phi_r=v[:k.total].reshape(len(k.c_r),len(k.c_s))
sol_sing=np.outer(Green(k.f_r,Rv), q)
interp=np.zeros([len(k.f_r), len(k.f_s)])
d=c=0
for i in k.f_r:
    d=0
    for j in k.f_s:
        t=interpolate(i, j, phi_r, k.c_r, k.c_s, k.c_hs, k.c_hr )
        
        interp[c,d]=t
        d+=1
    c+=1

#Ghosts
v_ghosts=sp.sparse.linalg.spsolve(M_ghosts,IC)
q_ghosts=v_ghosts[-len(phi_vessel):]
plt.plot(q_ghosts)
plt.title("q_ghosts")
v_ghosts=v_ghosts[:M_ghosts.shape[0]-len(phi_vessel)].reshape(coarse_r_points+1,coarse_s_points)
interp2=np.zeros([len(k.f_r), len(k.f_s)])
d=c=0
for i in k.f_r:
    d=0
    for j in k.f_s:
        t=interpolate(i, j, phi_r, k.c_r, k.c_s, k.c_hs, k.c_hr )
        
        interp2[c,d]=t
        d+=1
    c+=1

#tests of ghost
# =============================================================================
# pp=a[k.north_boundary,:]*np.ndarray.flatten(v_ghosts)
# plt.plot(k.c_s,pp); plt.plot(k.f_s, -2*Green(R_max, Rv)*q_ghosts)
# 
# =============================================================================
sol_final=sol_sing+interp
                
plt.figure()
plt.contourf(sol_final)
plt.colorbar()

# =============================================================================
# sol_NN=sol_sing+NN
# =============================================================================



#Comparison
k=simplified_assembly_Laplacian_cylindrical(R_max, Rv,coarse_r_points,coarse_s_points, L, D)

#Set Dirichlet on the outer boundary
ext_boundary=np.concatenate([[k.corners[2]], k.outer_boundary, [k.corners[3]]])
Lap_operator=k.assembly().toarray()
Lap_operator[ext_boundary,:]=0
Lap_operator[ext_boundary,ext_boundary]=1

h=len(phi_vessel)/len(k.s)
uu=np.linspace(h/2,len(phi_vessel)-h/2,len(k.s)).astype(int)
phi_coarse=phi_vessel[uu]

inner_boundary=np.concatenate([[k.corners[0]],k.inner_boundary,[k.corners[1]]])

pp=get_sol_coupled(Lap_operator,phi_coarse, Rv, k.inc_r, k.inc_s, inner_boundary, K_eff*np.pi*Rv**2, np.zeros([len(k.r)*len(k.s)]))


plt.figure()
plt.plot(k.s, phi_r[0], label="solution splitting")
plt.plot(f_s, sol_reference_fine_coupled[0,:], label="numerical reference")
plt.xlabel("vessel wall")
plt.ylabel("concentration")
plt.title("comparison final")
plt.legend(fontsize=22)
plt.savefig("Vessel wall estimation performance")


coarse_r_points=45
coarse_s_points=45
#Get coarse comparison
obj_coarse=simplified_assembly_Laplacian_cylindrical(R_max, Rv,coarse_r_points,coarse_s_points, L, D)
    
c_hs, c_hr=obj_coarse.inc_s, obj_coarse.inc_r
c_r=obj_coarse.r
c_s=obj_coarse.s
f_factor=1/(np.pi*(2*Rv+c_hr)*c_hr)
ff_factor=1/(2*np.pi*c_r[0]*c_hr)

Lap_operator=obj_coarse.assembly().toarray()

#Set Dirichlet on the outer boundary
ext_boundary=np.concatenate([[obj_coarse.corners[2]], obj_coarse.outer_boundary, [obj_coarse.corners[3]]])
Lap_operator[ext_boundary,:]=0
Lap_operator[ext_boundary,ext_boundary]=1



rhs_real_coarse_coupled=np.zeros(len(obj_coarse.s)*len(obj_coarse.r))
in_boundary=np.concatenate([[obj_coarse.corners[0]], obj_coarse.inner_boundary, [obj_coarse.corners[1]]])
east_boundary=np.concatenate([[obj_coarse.corners[1]], obj_coarse.east_boundary, [obj_coarse.corners[3]]])


#Up to here we have the laplacian operator "Lap_operator", with the Dirichlet BC set on hte outer boundary 
#and with phi_vessel defined
phi_coarse=np.cos(obj_coarse.s/(L*0.7))
#NOW THE FUNCTION THAT SOLVES THE PROBLEM GETS INVOKED
sol_reference_coarse_coupled=get_sol_coupled(Lap_operator, phi_coarse, Rv, c_hr, c_hs, in_boundary, K_eff*np.pi*Rv**2, rhs_real_coarse_coupled)

plt.contourf(sol_reference_coarse_coupled); plt.colorbar()

pp=interp_kernel_oneD(k.s, k.inc_s, f_s, f_hs)
NN_1D_interp_kernel=sp.sparse.csc_matrix((pp[0], (pp[1], pp[2])), shape=(len(f_s), len(k.s)))
ii=NN_1D_interp_kernel.dot(v_ghosts[0])

plt.figure()
plt.plot(f_s,ii, label="solution splitting")
plt.plot(f_s, sol_reference_fine_coupled[0,:], label="numerical reference")
plt.plot(c_s, sol_reference_coarse_coupled[0], label="coarse reference scheme")
plt.xlabel("vessel wall")
plt.ylabel("concentration")
plt.title("comparison final")
plt.legend(fontsize=22)
plt.savefig("Vessel wall estimation performance")


err_splitting=(ii-sol_reference_fine_coupled[0,:])**2/sol_reference_fine_coupled[0,:]**2
pp=interp_kernel_oneD(c_s, c_hs, f_s, f_hs)
kernel=sp.sparse.csc_matrix((pp[0], (pp[1], pp[2])), shape=(len(f_s), len(c_s)))
num_interp=kernel.dot(sol_reference_coarse_coupled[0])
err_numerical=(num_interp-sol_reference_fine_coupled[0,:])**2/sol_reference_fine_coupled[0,:]**2

