#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:59:56 2021

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
from cylindrical_coupling_assembly import full_2D_linear_interpolation
#-------------------------------------------------------
#CYLINDRICAL COORDINATES 
#-------------------------------------------------------

R_max=1
L=1
s_points=100
r_points=100
Rv=0.05
D=1
K_eff=5
inner_modification=False
west_mode=False
my_west_mode=True
sour="cosine"

 

#Let's create the Laplacian operator with the proper boundary conditions (Dirichlet north and 
#no flux east and west)
k=simplified_assembly_Laplacian_cylindrical(R_max, Rv,r_points,s_points, L, D)
hs, hr=k.inc_s, k.inc_r
Lap_operator=k.assembly().toarray()

#Set Dirichlet on the outer boundary
ext_boundary=np.concatenate([[k.corners[2]], k.outer_boundary, [k.corners[3]]])
Lap_operator[k.outer_boundary,:]=0
Lap_operator[k.outer_boundary,k.outer_boundary]=1

if sour=="cosine":
    #Set an analytical source profile
    q_vessel=np.cos(k.s/(L*0.7))
    q_first=-(1/(L*0.7))*np.sin(k.s/(L*0.7))
    q_sec=-np.cos(k.s/(L*0.7))*(1/(L*0.7))**2
if sour=="constant":
    q_vessel=np.zeros(len(k.s))+1
    q_first=q_vessel-1
    q_sec=q_vessel-1

der1=get_first_der(q_vessel, hs).dot(q_vessel)
der2=get_sec_der(q_vessel, hs).dot(q_vessel)


#For starters, we are gonna compute the real value of the regular term by substracting the singular
#term from the real solution. 
#The real solution is computed numerically with the usual problem
GRFQ=np.outer(Green(k.r, Rv), q_vessel)
SRFQ=get_reference_q(Lap_operator,q_vessel, Rv, k.inc_r,k.inc_s, ext_boundary)
VRFQ=SRFQ-GRFQ
RHS_real_fine=Lap_operator.dot(np.ndarray.flatten(VRFQ))
rhs_real_rhs_fine=RHS_real_fine.reshape(len(k.r), len(k.s)) #this is the real array of the RHS of the regular term
    #It is used for validation

RHS_split_fine=get_RHS(q_vessel,D, Rv, k.s, k.r, k.inc_s, k.inc_r,  k.east_boundary, k.west_boundary, k.outer_boundary, k.corners, q_first, q_sec)
rhs_split_fine=RHS_split_fine.reshape(len(k.r), len(k.s)) 

# =============================================================================
# This next line takes place because I am to lazy to calculate the real value of the 
# inner Neumann boundary condition. In a normal problem it would be zero, however, due to the 
# definition of r that I have chosen the domain does not include the source, therefore, it needs 
# to have a flux boundary condition that I am obtaining from the retroengineered r_rhs array
# =============================================================================


if inner_modification:
    print("\ninner mode ON\n")
    rhs_split_fine[0,1:-1]=rhs_real_fine[0,1:-1]
    RHS_split_fine=np.ndarray.flatten(rhs)

if west_mode:
    print("\nwest mode on\n")
    rhs_split_fine[:,-1]=rhs_real_fine[:,-1]
    RHS_split_fine=np.ndarray.flatten(rhs_split_fine)
    
if my_west_mode:
    print("\n My west mode on \n")
    rhs_split_fine[1:-1,-1]*=-(1/hs)
    RHS_split_fine=np.ndarray.flatten(rhs_split_fine)

VSFQ=sp.sparse.linalg.spsolve(Lap_operator, RHS_split_fine).reshape(len(k.r), len(k.s))
SSFQ=VSFQ+GRFQ



N=3

    
plt.figure()
plot_profile(1, k.s, k.r, SSFQ, SRFQ, "Comparison radial profile")
plt.figure()
plot_profile(1, k.s, k.r, VRFQ, VSFQ, "Comparison radial profile v")
plt.ylim([0,1])
plt.ylabel("v")



plt.figure()
for i in range(N):
    l=len(k.s)//N
    pos=l*i
    plt.plot(k.r, SSFQ[:,pos])
    #plt.plot(k.r, analyt)
    plt.plot(k.r, SRFQ[:,pos])
    # =============================================================================
    # plt.plot(k.r, VRFQ[:,10])
    # plt.plot(k.r, sol[:,10])
    # =============================================================================
plt.legend(["sol_0","real_0","sol_1/3","real_1/3","sol_2/3","real_2/3"])
plt.title("Comparison of the radial profiles of the split solution and \n the real one for different positions along the s-axis")
plt.xlabel("r")
plt.ylabel("$\phi$")
#plt.legend(fontsize=12) # using a size in points


compare_full_solutions(SSFQ, SRFQ, "sol_split", "SRFQ", [0,np.max(SRFQ)], "title")

RHS_sing_fine=Lap_operator.dot(np.ndarray.flatten(GRFQ))
rhs_sing_fine=RHS_sing_fine.reshape(len(k.r), len(k.s))



#Coarse_solution
coarse_r_points=10
coarse_s_points=10


#Let's create the Laplacian operator with the proper boundary conditions (Dirichlet north and 
#no flux east and west)
coarse=simplified_assembly_Laplacian_cylindrical(R_max, Rv,coarse_r_points,coarse_s_points, L, D)
coarse_Lap=coarse.assembly().toarray()

c_r=coarse.r
c_s=coarse.s
c_hs=coarse.inc_s
c_hr=coarse.inc_r

obj_sol_split_coarse=sol_split_q_imposed(coarse, k, q_vessel, q_sec, q_first, coarse_Lap)

f_s=obj_sol_split_coarse.fine_s
f_r=obj_sol_split_coarse.fine_r
f_hs=obj_sol_split_coarse.fine_hs
f_hr=obj_sol_split_coarse.fine_hr

validation_RHS=np.zeros([len(coarse.r), len(coarse.s)])
for i in range(len(coarse.r)):
    for j in range(len(coarse.s)):
        validation_RHS[i,j]=-analyt_int_2D(L,D,coarse.r[i], coarse.s[j], coarse.inc_r, coarse.inc_s, coarse.Rv)

total_no_ghost=coarse_r_points*coarse_s_points


obj_sol_split_coarse.seg_north_with_ghost() #Invokes the function that creates the ghosts and sets the Dirichlet_out
#This function implies the creation of a new laplacian taking into account the ghosts cells

RHS_split_coarse=obj_sol_split_coarse.get_east_west_RHS() #Needed to satisfy the side Neumann BC
RHS_split_coarse+=np.ndarray.flatten(obj_sol_split_coarse.get_RHS_v()) #This is needed for the creation term

RHSghost_split_coarse=obj_sol_split_coarse.RHS_ghost
RHSghost_split_coarse[:total_no_ghost]=RHS_split_coarse

vsc1q=np.linalg.solve(obj_sol_split_coarse.ghost_coarse_Lap,RHSghost_split_coarse)
vghostsc1q=vsc1q[:total_no_ghost]

VghostSC1Q=vghostsc1q.reshape(coarse_r_points, coarse_s_points)
VghostSC1Q_interp=full_2D_linear_interpolation(f_r,f_s, VghostSC1Q, c_r, c_s, c_hs, c_hr)

SSC1Q_interp=VghostSC1Q_interp+GRFQ

#Comparison of the coarse solution without coupling
kk=simplified_assembly_Laplacian_cylindrical(R_max, Rv,len(obj_sol_split_coarse.coarse_r),len(obj_sol_split_coarse.coarse_s), L, D)
hs, hr=kk.inc_s, kk.inc_r
Lap_operator=kk.assembly().toarray()
ext_boundary=np.concatenate([[kk.corners[2]], kk.outer_boundary, [kk.corners[3]]])
k_q_vessel=np.cos(kk.s/(L*0.7))
SRCQ=get_reference_q(Lap_operator,k_q_vessel, Rv, kk.inc_r,kk.inc_s, ext_boundary)


plt.figure()
plt.plot(k.r, SSC1Q_interp[:,40], label="sol_split_interp")
plt.plot(k.r,SRFQ[:,40],label="SRFQ")
plt.plot(obj_sol_split_coarse.coarse_r,SRCQ[:,4],label="sol_real_coarse")
plt.legend()

def add_coarse_with_singular(coarse_r, coarse_s, coarse_v, fine_r, fine_s, q_vessel, Rv):
    sing=np.outer(Green(fine_r, Rv), q_vessel)
    #array that defines to which cell in the coarse arrays each of the fine positions belongs to
    position_r=np.array([]).astype(int)
    position_s=np.array([]).astype(int)
    
    coarse_v=coarse_v.reshape(len(coarse_r),len(coarse_s))
    
    for i in fine_r:
        position_r=np.append(position_r, np.argmin((i-coarse_r)**2))
    for j in fine_s:
        position_s=np.append(position_s, np.argmin((j-coarse_s)**2))
        
        
    solution=np.zeros([len(fine_r),len(q_vessel)])
    for i in range(len(fine_r)):
        for j in range(len(fine_s)):
            solution[i,j]=sing[i,j]+coarse_v[position_r[i], position_s[j]]
    return(solution)


b=add_coarse_with_singular(coarse.r, coarse.s, VghostSC1Q, k.r, k.s, q_vessel, Rv)


def get_averaged_solution(c_r,c_s, f_r, f_s, fine_solution, h_r, h_s):
    avg=np.zeros([len(c_r), len(c_s)])
    c=0
    for i in c_r:
       d=0
       pos_r=np.where((f_r-i)**2<h_r**2)[0]
       for j in c_s:
           pos_s=np.argmin((f_s-j)**2<h_s**2)[0]
           avg[c,d]=np.sum(fine_solution[pos_r,:][:,pos_s])/(len(pos_s)*len(pos_r))
           d+=1
       c+=1
    return(avg)
    
           