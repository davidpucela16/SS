#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:05:05 2021

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

plt.rcParams["figure.figsize"] = (8,8)

from cylindrical_problem_assembly import *

from cylindrical_coupling_assembly import *
    
R_max=1
L=1
fine_s_points=80
fine_r_points=80
Rv=0.05
D=1
K_eff=2/(np.pi*Rv**2)
inner_modification=False
west_mode=False
my_west_mode=True
sour="sine"


#Let's create the Laplacian operator with the proper boundary conditions (Dirichlet north and 
#no flux east and west)
obj_fine=simplified_assembly_Laplacian_cylindrical(R_max, Rv,fine_r_points,fine_s_points, L, D)

if sour=="cosine":
    #Set an analytical source profile
    q_vessel=np.cos(obj_fine.s/(L*0.7))
    q_first=-(1/(L*0.7))*np.sin(obj_fine.s/(L*0.7))
    q_sec=-np.cos(obj_fine.s/(L*0.7))*(1/(L*0.7))**2
if sour=="constant":
    q_vessel=np.zeros(len(obj_fine.s))+1
    q_first=q_vessel-1
    q_sec=q_vessel-1
if sour=="sine":
    q_vessel=np.sin(obj_fine.s/(L*0.7))
    
f_hs, f_hr=obj_fine.inc_s, obj_fine.inc_r
f_r=obj_fine.r
f_s=obj_fine.s
f_factor=1/(np.pi*(2*Rv+f_hr)*f_hr)
ff_factor=1/(2*np.pi*f_r[0]*f_hr)

Lap_virgin=obj_fine.assembly().toarray()
Lap_operator=obj_fine.assembly().toarray()

#Set Dirichlet on the outer boundary
ext_boundary=np.concatenate([[obj_fine.corners[2]], obj_fine.outer_boundary, [obj_fine.corners[3]]])
Lap_operator[ext_boundary,:]=0
Lap_operator[ext_boundary,ext_boundary]=1

phi_vessel=q_vessel

rhs_real_fine_coupled=np.zeros(len(obj_fine.s)*len(obj_fine.r))
in_boundary=np.concatenate([[obj_fine.corners[0]], obj_fine.inner_boundary, [obj_fine.corners[1]]])
east_boundary=np.concatenate([[obj_fine.corners[1]], obj_fine.east_boundary, [obj_fine.corners[3]]])


#Up to here we have the laplacian operator "Lap_operator", with the Dirichlet BC set on hte outer boundary 
#and with phi_vessel defined

#NOW THE FUNCTION THAT SOLVES THE PROBLEM GETS INVOKED
SRFC=get_sol_coupled(Lap_operator, phi_vessel, Rv, f_hr, f_hs, in_boundary, K_eff*np.pi*Rv**2, rhs_real_fine_coupled)

q_coupled_wall=K_eff*np.pi*Rv**2*(phi_vessel-SRFC[0,:]) 
q_coupled_theta=-Lap_virgin[in_boundary].dot(np.ndarray.flatten(SRFC))/ff_factor

SRFQ=get_validation_q_imposed_outerDirichlet(q_coupled_theta, R_max, Rv, fine_r_points, fine_s_points,L,D)


compare_full_solutions(SRFC, SRFQ, "numerical", "validation q imposed",[0,np.max(SRFQ)], "Coupled numerical comparison")

#The reference solution is obtained for this concentration profile: SRFC

new_operator=obj_fine.assembly().toarray()
in_flux=-new_operator[in_boundary].dot(np.ndarray.flatten(SRFC))/ff_factor

plt.plot(in_flux)

sol_reference_fine=get_validation_q_imposed_outerDirichlet(in_flux, R_max, Rv, fine_r_points, fine_s_points,L,D)

compare_full_solutions(SRFC, sol_reference_fine, "numerical", "validation q imposed",[0,np.max(sol_reference_fine)], "Coupled numerical comparison")

plt.plot(SRFC[0,:])



#FROM NOW ON THE COARSE GRID IS INTRODUCED 


def coord_to_pos(s,coord):
    """give the coordinates and return fine position"""
    pos=np.array([]).astype(int)
    for i in coord:
        pos=np.append(pos,np.argmin((s-i)**2))
    return(pos)





coarse_s_points=4
coarse_r_points=4

k=get_coupled_problem_matrix(R_max, Rv, coarse_r_points, coarse_s_points, L,D, fine_s_points, fine_r_points)
M=k.assemble_full_problem(K_eff)

c_s=k.c_s
c_hs=k.c_hs
c_r=k.c_r
c_hr=k.c_hr

M_ghosts=k.assemble_problem_Dirichlet_north_ghost(K_eff)

#Flux BC
s=M.shape
IC=np.zeros(s[0])
IC[-len(phi_vessel):]=phi_vessel
v=sp.sparse.linalg.spsolve(M,IC)
q=v[-len(phi_vessel):]
plt.plot(q)
phi_r=v[:k.total].reshape(len(k.c_r),len(k.c_s))
sing=np.outer(Green(k.f_r,Rv), q)
interp=np.zeros([len(k.f_r), len(k.f_s)])
d=c=0
for i in k.f_r:
    d=0
    for j in k.f_s:
        t=linear_interpolation(i, j, phi_r, k.c_r, k.c_s, k.c_hs, k.c_hr )
        
        interp[c,d]=t
        d+=1
    c+=1

#Ghosts
v_ghosts=sp.sparse.linalg.spsolve(M_ghosts,IC)
VghostsSC1C=v_ghosts[:M_ghosts.shape[0]-len(phi_vessel)].reshape(coarse_r_points+1,coarse_s_points)

pp=interp_kernel_oneD(c_s,c_hs, f_s, f_hs)
kernel=sp.sparse.csc_matrix((pp[0], (pp[1], pp[2])), shape=(len(f_s), len(c_s)))

ii=kernel.dot(VghostsSC1C[0])
q_ghosts_1=v_ghosts[(coarse_r_points+1)*(coarse_s_points):]
q_ghosts=K_eff*np.pi*Rv**2*(phi_vessel-kernel.dot(VghostsSC1C[0]))
plt.plot(q_ghosts)
plt.title("q_ghosts")

interp2=np.zeros([len(k.f_r), len(k.f_s)])
d=c=0
for i in k.f_r:
    d=0
    for j in k.f_s:
        t=linear_interpolation(i, j, VghostsSC1C, k.c_r, k.c_s, k.c_hs, k.c_hr )
        
        interp2[c,d]=t
        d+=1
    c+=1

#tests of ghost
# =============================================================================
# pp=a[k.north_boundary,:]*np.ndarray.flatten(v_ghosts)
# plt.plot(k.c_s,pp); plt.plot(k.f_s, -2*Green(R_max, Rv)*q_ghosts)
# 
# =============================================================================
sol_final=sing+interp
SSC1C=sol_final
                
plt.figure()
plt.contourf(SSC1C)
plt.colorbar()
plt.title("SS final contour")


ss=get_averaged_solution(c_r, c_s, f_r, f_s, SSC1C, c_hr, c_hs)
ss_ref=get_averaged_solution(c_r, c_s, f_r, f_s, SRFC, c_hr, c_hs)

compare_full_solutions(ss, ss_ref, "numerical", "validation q imposed",[-0.01,np.max(ss_ref)], "Coupled numerical comparison")

h=len(phi_vessel)/len(c_s)
uu=np.linspace(h/2,len(phi_vessel)-h/2,len(c_s)).astype(int)
phi_coarse=phi_vessel[uu]
avg_Green=Green(c_r[:-1], Rv)
q_split_coarse=K_eff*Rv**2*np.pi*(phi_coarse-VghostsSC1C[0,:])
avg_sol_split_ghost=np.outer(avg_Green,q_split_coarse)+VghostsSC1C[:-1,:]
plt.contourf(avg_sol_split_ghost); plt.colorbar()

# =============================================================================
# sol_NN=sing+NN
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
plt.plot(f_s, SRFC[0,:], label="numerical reference")
plt.xlabel("vessel wall")
plt.ylabel("concentration")
plt.title("comparison final")
plt.legend(fontsize=22)
plt.savefig("Vessel wall estimation performance")


c2_r_points=45
c2_s_points=45
#Get coarse comparison
obj_coarse=simplified_assembly_Laplacian_cylindrical(R_max, Rv,c2_r_points,c2_s_points, L, D)
    
c2_hs, c2_hr=obj_coarse.inc_s, obj_coarse.inc_r
c2_r=obj_coarse.r
c2_s=obj_coarse.s
f_factor=1/(np.pi*(2*Rv+c2_hr)*c2_hr)
ff_factor=1/(2*np.pi*c2_r[0]*c2_hr)

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
h=len(phi_vessel)/c2_s_points
uu=np.linspace(h/2,len(phi_vessel)-h/2,c2_s_points).astype(int)
phi_coarse=phi_vessel[uu]
#NOW THE FUNCTION THAT SOLVES THE PROBLEM GETS INVOKED
sol_reference_coarse_coupled=get_sol_coupled(Lap_operator, phi_coarse, Rv, c2_hr, c2_hs, in_boundary, K_eff*np.pi*Rv**2, rhs_real_coarse_coupled)

plt.contourf(sol_reference_coarse_coupled); plt.colorbar()


plt.figure()
plt.plot(f_s,ii, label="solution splitting")
plt.plot(f_s, SRFC[0,:], label="numerical reference")
plt.plot(c2_s, sol_reference_coarse_coupled[0], label="coarse reference scheme")
plt.xlabel("vessel wall")
plt.ylabel("concentration")
plt.title("comparison final")
plt.legend(fontsize=22)
plt.savefig("Vessel wall estimation performance")


err_splitting=(ii-SRFC[0,:])**2/SRFC[0,:]**2
pp=interp_kernel_oneD(c2_s, c2_hs, f_s, f_hs)
kernel=sp.sparse.csc_matrix((pp[0], (pp[1], pp[2])), shape=(len(f_s), len(c2_s)))
num_interp=kernel.dot(sol_reference_coarse_coupled[0])
err_numerical=(num_interp-SRFC[0,:])**2/SRFC[0,:]**2

plt.figure()
plt.plot(err_splitting, label="error split")
plt.plot(err_numerical, label="error numerical reference")
plt.legend()


#Flux estimation!
q_reference_coarse=K_eff*np.pi*Rv**2*(phi_coarse-sol_reference_coarse_coupled[0])

plt.figure()
plt.plot(f_s, q_coupled_theta, label="q_reference")
plt.plot(c2_s, q_reference_coarse, label="numerical")
plt.plot(f_s, q_ghosts,label="solution splitting")
plt.legend()



