#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:35:22 2021

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
plt.rcParams["figure.figsize"] = (12,12)
plt.rcParams.update({'font.size': 20})
from cylindrical_problem_assembly import *

from cylindrical_coupling_assembly import *


R_max=1
L=1
fine_s_points=150
fine_r_points=150
Rv=0.05
D=1
K_eff=2/(np.pi*Rv**2)
inner_modification=False
west_mode=False
my_west_mode=True
sour="cosine" #shape of the source profile



#Let's create the Laplacian operator with the proper boundary conditions (Dirichlet north and 
#no flux east and west)
obj_fine=simplified_assembly_Laplacian_cylindrical(R_max, Rv,fine_r_points,fine_s_points, L, D)

if sour=="cosine":
    #Set an analytical source profile
    phi_vessel=np.cos(obj_fine.s/(L*0.7))
if sour=="constant":
    phi_vessel=np.zeros(len(obj_fine.s))+1
if sour=="sine":
    phi_vessel=np.sin(obj_fine.s/(L*0.7))

#kernel=get_first_der(phi_vessel,obj_fine.inc_s) #already in sparse matrix format
#phi_first=kernel.dot(phi_vessel)

plt.plot(obj_fine.s,phi_vessel)
plt.xlabel("s")
plt.ylabel(" $ \langle \phi \langle $ ")
plt.title("Fixed averaged concentration profile along the vessel centerline")


f_hs, f_hr=obj_fine.inc_s, obj_fine.inc_r
f_r=obj_fine.r
f_s=obj_fine.s

#factor pour prendre en compte la difference des volumes entre une cellule de vaisseau et une
#cellule du tissue a coté
f_factor=1/(np.pi*(2*Rv+f_hr)*f_hr)
ff_factor=1/(2*np.pi*f_r[0]*f_hr)

Lap_virgin=obj_fine.assembly().toarray()
Lap_operator=obj_fine.assembly().toarray()

#Set Dirichlet on the outer boundary
ext_boundary=np.concatenate([[obj_fine.corners[2]], obj_fine.outer_boundary, [obj_fine.corners[3]]])
Lap_operator[ext_boundary,:]=0
Lap_operator[ext_boundary,ext_boundary]=1

in_boundary=np.concatenate([[obj_fine.corners[0]], obj_fine.inner_boundary, [obj_fine.corners[1]]])
east_boundary=np.concatenate([[obj_fine.corners[1]], obj_fine.east_boundary, [obj_fine.corners[3]]])

#NOW THE FUNCTION THAT SOLVES THE PROBLEM GETS INVOKED
rhs_real_fine_coupled=np.zeros(len(obj_fine.s)*len(obj_fine.r))
#Solution Reference Fine Coupled
SRFC=get_sol_coupled(Lap_operator, phi_vessel, Rv, f_hr, f_hs, in_boundary, K_eff*np.pi*Rv**2, rhs_real_fine_coupled)


q_coupled_wall=K_eff*np.pi*Rv**2*(phi_vessel-SRFC[0,:]) #influx 
#The reference solution is obtained for this concentration profile: SRFC
new_operator=obj_fine.assembly().toarray()
in_flux=-new_operator[in_boundary].dot(np.ndarray.flatten(SRFC))/ff_factor
plt.plot(in_flux)
plt.title("coupling flux entering tissue")
plt.xlabel("s")
plt.ylabel("$kg m^{-1} s^{-1}$")

#Solution Reference Fine Q_imposed
SRFQ=get_validation_q_imposed_outerDirichlet(in_flux, R_max, Rv, fine_r_points, fine_s_points,L,D)
sol_reference_fine=get_validation_q_imposed_outerDirichlet(in_flux, R_max, Rv, fine_r_points, fine_s_points,L,D)

compare_full_solutions(SRFC, sol_reference_fine, "numerical", "validation q imposed",[0,np.max(sol_reference_fine)], "Coupled numerical comparison, just for validation")


plt.plot(SRFC[0,:])
plt.title("concentration in tissue along the vessel wall - Validation", fontsize=20)
plt.show()

coarse_s_points=5
coarse_r_points=5

k=get_coupled_problem_matrix(R_max, Rv, coarse_r_points, coarse_s_points, L,D, fine_s_points, fine_r_points)
M=k.assemble_full_problem(K_eff)

c_s=k.c_s
c_hs=k.c_hs
c_r=k.c_r
c_hr=k.c_hr

#The ghosts are used as one way to set the Dirichlet BCs
M_ghosts=k.assemble_problem_Dirichlet_north_ghost(K_eff)

#Flux BC
s=M.shape
IC=np.zeros(s[0])
IC[-len(phi_vessel):]=phi_vessel
v=sp.sparse.linalg.spsolve(M,IC)
q=v[-len(phi_vessel):]
#plt.plot(q)
phi_r=v[:k.total].reshape(len(k.c_r),len(k.c_s))
sing=np.outer(Green(k.f_r,Rv), q)

#The following snippet is to interpolate linearly (and very simply, without considering 
#the jacobian of the cylindrical system) the regular term.
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


pp=interp_kernel_oneD(c_s,c_hs, f_s, f_hs) #interpolation (not computation step) for the regular 
                                        #term along the vessel wall
kernel=sp.sparse.csc_matrix((pp[0], (pp[1], pp[2])), shape=(len(f_s), len(c_s)))

ii=kernel.dot(VghostsSC1C[0])
q_ghosts_1=v_ghosts[(coarse_r_points+1)*(coarse_s_points):]
q_ghosts=K_eff*np.pi*Rv**2*(phi_vessel-kernel.dot(VghostsSC1C[0]))
plt.plot(f_s, q_ghosts)
plt.xlabel("s")
plt.ylabel("$dfrac{kg}{ms}$")

plt.title("flux estimation solution splitting")

interp2=np.zeros([len(f_r), len(f_s)])
d=c=0
for i in f_r:
    d=0
    for j in f_s:
        t=linear_interpolation(i, j, VghostsSC1C, c_r, c_s, c_hs, c_hr )
        
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
#Solution Split COARSEST Coupled


plt.plot(f_s, v[-len(phi_vessel):])
plt.xlabel("s")
plt.ylabel("$dfrac{kg}{ms}$")

plt.title("flux estimation solution splitting")

h=len(phi_vessel)/len(c_s)
uu=np.linspace(h/2,len(phi_vessel)-h/2,len(c_s)).astype(int)
phi_coarse=phi_vessel[uu]
avg_Green=Green(c_r[:-1], Rv) #Not really an average!!!!!!!
q_split_coarse=K_eff*Rv**2*np.pi*(phi_coarse-VghostsSC1C[0,:])
avg_sol_split=np.outer(avg_Green,q_split_coarse)+VghostsSC1C[:-1,:]
plt.contourf(avg_sol_split); plt.colorbar(); plt.title("SS locally averaged")

ss=get_averaged_solution(c_r, c_s, f_r, f_s, SSC1C, c_hr, c_hs)
ss_ref=get_averaged_solution(c_r, c_s, f_r, f_s, SRFC, c_hr, c_hs)

compare_full_solutions(avg_sol_split, ss_ref, "Sol Split", "validation_avg",[-0.01,np.max(ss_ref)*1.2], "averaged comparison")

compare_full_solutions(avg_sol_split, SRFC, "Sol Split", "Validation",[-0.01,np.max(SRFC)*1.2], "Non averaged comparison")


sh=avg_sol_split.shape
hr=SRFC.shape[0]/sh[0]
hs=SRFC.shape[1]/sh[1]
uur=np.linspace(hr/2,SRFC.shape[0]-hr/2,sh[0]).astype(int)
uus=np.linspace(hs/2,SRFC.shape[1]-hs/2,sh[1]).astype(int)
SRFC_coarse_s=SRFC[uur,:]
SRFC_coarse_r=SRFC[:,uus]

fig3 = plt.figure(figsize=(16, 16))
ax1 = fig3.add_subplot(2, 1, 1)
ax2 = fig3.add_subplot(2, 1, 2)

for i in range(sh[0]):
    ax1.plot(f_s, SRFC_coarse_s[i,:], label="Validation pos {}/10 Rmax".format(2*i+1))
    ax1.plot(c_s, avg_sol_split[i,:], '*', label="Sol Split pos {}/10 Rmax".format(2*i+1))
    ax1.set_title("axial comparison")
    ax1.set_xlabel("s")
    ax1.set_ylabel("$\phi$")
    ax1.legend()

for j in range(sh[1]):
    ax2.plot(f_s, SRFC_coarse_r[:,j], label="Validation pos {}/10 L".format(2*j+1))
    ax2.plot(c_s, avg_sol_split[:,j], '*',label="Sol Split pos {}/10 L".format(2*j+1))
    ax2.set_title("radial comparison")
    ax2.set_xlabel("r")
    ax2.set_ylabel("$\phi$")
    ax2.legend()
    
    
x=np.linspace(0,1,10)

plt.plot(x,np.zeros(len(x))+1/10, '#1f77b4');
plt.plot(x,np.zeros(len(x))+3/10, 'g');
plt.plot(x,np.zeros(len(x))+5/10, '#9467bd');
plt.plot(x,np.zeros(len(x))+7/10, '#e377c2');
plt.plot(x,np.zeros(len(x))+9/10, 'y');
plt.ylim((0,1))
plt.xlabel('s')
plt.ylabel('r')
plt.title("positions axial comparison")

plt.plot(np.zeros(len(x))+1/10,x, '#1f77b4');
plt.plot(np.zeros(len(x))+3/10, x,'g');
plt.plot(np.zeros(len(x))+5/10, x,'#9467bd');
plt.plot(np.zeros(len(x))+7/10, x,'#e377c2');
plt.plot(np.zeros(len(x))+9/10, x,'y');
plt.xlim((0,1))
plt.xlabel('s')
plt.ylabel('r')
plt.title("positions radial comparison")

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
plt.plot(k.s, phi_r[0], '*', label="Solution Splitting")
plt.plot(f_s, SRFC[0,:], label="validation")
plt.xlabel("vessel wall")
plt.ylabel("concentration")
plt.title("Concentration along the vessel wall")
plt.legend(fontsize=22)
plt.savefig("Vessel wall estimation performance")


c2_r_points=49
c2_s_points=49



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
SRCC=sol_reference_coarse_coupled
plt.contourf(sol_reference_coarse_coupled); plt.colorbar(); plt.title("{} x {} numerical".format(c2_s_points,c2_r_points))


#Flux estimation!
q_reference_coarse=K_eff*np.pi*Rv**2*(phi_coarse-sol_reference_coarse_coupled[0])

plt.figure()
plt.plot(f_s, in_flux, label="q_reference")
plt.plot(c2_s, q_reference_coarse, '*',label="{} x {} numerical".format(c2_s_points+1,1+c2_r_points))
plt.plot(f_s, q_ghosts,'+',label="solution splitting")
plt.xlabel("s")
plt.ylabel("$\dfrac{kg}{m s}$")
plt.title("In flux along the vessel wall for the three solutions")
plt.legend()

pp=interp_kernel_oneD(c2_s, c2_hs, f_s, f_hs)
kernel=sp.sparse.csc_matrix((pp[0], (pp[1], pp[2])), shape=(len(f_s), len(c2_s)))
num_interp=kernel.dot(q_reference_coarse)
err_numerical=np.abs(num_interp-in_flux)/in_flux

err_splitting=np.abs(q_ghosts-in_flux)/in_flux

fig1 = plt.figure(figsize=(10,10))
ax1 = fig1.add_subplot(2, 1, 1)
ax2 = fig1.add_subplot(2, 1, 2)

ax1.plot(err_splitting, label="error split")
ax1.plot(err_numerical, label="error numerical reference")
ax1.set_title("relative error")
ax1.set_ylim(0, 0.3)
ax1.legend()

ax2.plot(err_splitting*in_flux, label="error split")
ax2.plot(err_numerical*in_flux, label="error numerical reference")
ax2.set_title("absolute error")
ax2.legend()

sh=avg_sol_split.shape
hr=SRFC.shape[0]/sh[0]
hs=SRFC.shape[1]/sh[1]
uur=np.linspace(hr/2,SRFC.shape[0]-hr/2,sh[0]).astype(int)
uus=np.linspace(hs/2,SRFC.shape[1]-hs/2,sh[1]).astype(int)
SRFC_coarse_s=SRFC[uur,:]
SRFC_coarse_r=SRFC[:,uus]

shc=avg_sol_split.shape
hrc=SRCC.shape[0]/shc[0]
hsc=SRCC.shape[1]/shc[1]
uurc=np.linspace(hrc/2,SRCC.shape[0]-hrc/2,shc[0]).astype(int)
uusc=np.linspace(hsc/2,SRCC.shape[1]-hsc/2,shc[1]).astype(int)
SRCC_coarse_s=SRCC[uurc,:]
SRCC_coarse_r=SRCC[:,uusc]

fig3 = plt.figure(figsize=(20,20))
ax1 = fig3.add_subplot(2, 1, 1)
ax2 = fig3.add_subplot(2, 1, 2)

for i in range(sh[0]):
    ax1.plot(f_s, SRFC_coarse_s[i,:], label="reference pos {}/10 Rmax".format(2*i+1))
    ax1.plot(c2_s, SRCC_coarse_s[i,:], '+' ,label="FV coarse pos {}/10 Rmax".format(2*i+1))
    ax1.plot(c_s, avg_sol_split[i,:], '*', label="sol split pos {}/10 Rmax".format(2*i+1))
    ax1.set_title("axial comparison")
    ax1.set_xlabel("s")
    ax1.set_ylabel("$\phi$")
    ax1.legend()

for j in range(sh[1]):
    ax2.plot(f_r, SRFC_coarse_r[:,j], label="reference pos {}/10 L".format(2*j+1))
    ax2.plot(c2_r, SRCC_coarse_r[:,j],'+', label="FV pos {}/10 L".format(2*j+1))
    ax2.plot(c_r[:-1], avg_sol_split[:,j], '*',label="sol split pos {}/10 L".format(2*j+1))
    ax2.set_title("radial comparison")
    ax2.set_xlabel("r")
    ax2.set_ylabel("$\phi$")
    ax2.legend()

fig3.suptitle("Comparison of the 3 solutions along the domain for the concentration field")



fig=plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(f_s, in_flux, 'b',label="Valildation")
ax1.plot(c2_s, q_reference_coarse, 'r+',label="{} x {} numerical".format(c2_s_points+1,1+c2_r_points))
ax1.plot(f_s, q_ghosts,'g',label="Sol Split")
ax1.set_xlabel("s")
ax1.set_ylabel("$\dfrac{kg}{m s}$")
ax1.set_title("In flux along the vessel wall for the three solutions")
ax1.legend(fontsize=22)

ax2.plot(f_s,ii, 'g',label="Sol Split")
ax2.plot(f_s, SRFC[0,:], 'b',label="Validation")
ax2.plot(c2_s, sol_reference_coarse_coupled[0],'r+', label="{} x {} numerical".format(c2_s_points+1,c2_r_points+1))
ax2.set_xlabel("vessel wall")
ax2.set_ylabel("concentration")
ax2.set_title("Comparison vessel wall concentration \n along the vessel axial coordinate")
ax2.legend(fontsize=22)


err_splitting=np.abs(ii-SRFC[0,:])/SRFC[0,:]
pp=interp_kernel_oneD(c2_s, c2_hs, f_s, f_hs)
kernel=sp.sparse.csc_matrix((pp[0], (pp[1], pp[2])), shape=(len(f_s), len(c2_s)))
num_interp=kernel.dot(sol_reference_coarse_coupled[0])
err_numerical=np.abs(num_interp-SRFC[0,:])/SRFC[0,:]


fig2 = plt.figure(figsize=(10,10))
ax2 = fig2.add_subplot(2, 1, 1)
ax3 = fig2.add_subplot(2, 1, 2)

ax2.plot(err_splitting, label="Sol Split")
ax2.plot(err_numerical, label="{} x {} numerical".format(c2_s_points+1,c2_r_points+1))
ax2.set_title("relative error")
ax2.set_ylim([0,0.3])
ax2.legend()

ax3.plot(np.abs(ii-SRFC[0,:]), label="Sol Split")
ax3.plot(np.abs(num_interp-SRFC[0,:]), label="{} x {} numerical".format(c2_s_points+1,c2_r_points+1))
ax3.set_title("absolute error")
ax3.legend()


plt.figure()
plt.imshow(sol_final)
plt.title("concentration contour for SS")
plt.colorbar()

len(np.where(sol_final<0)[0])



plt.imshow(VghostsSC1C); plt.colorbar(); plt.title("regular")

plt.figure()
plt.imshow(sing)
plt.title("singular")
plt.colorbar()

plt.contourf(VghostsSC1C); plt.colorbar(); plt.title("regular")

plt.figure()
plt.contourf(sol_final)
plt.title("discontinous concentration contour for SS")
plt.colorbar()


