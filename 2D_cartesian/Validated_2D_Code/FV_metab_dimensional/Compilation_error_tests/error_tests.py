#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This tests if made for the synthetic network
"""

directory='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Validated_2D_Code/FV_metab_dimensional'
directory2='/home/pdavid/Bureau/SS_malpighi/2D_cartesian/Validated_2D_Code/FV_metab_dimensional'
import os
os.chdir(directory2)

import numpy as np 
import matplotlib.pyplot as plt
from module_2D_coupling_FV_nogrid import * 
import reconst_and_test_module as post
import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab

import pandas 
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6,6 ),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

#0-Set up the sources
#1-Set up the domain
D=1
L=10
cells=20
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
#Rv=np.exp(-2*np.pi)*h_ss

alpha=50
diff_radii=True
sources=True #If both are source 


x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=1
print("directness=", directness)

d=3  #array of the separations!!
dist=d*L/alpha
q_array_source=np.zeros((0,2))

p1=np.array([L*0.5,L*0.5])
both_sources=True

#%% TO DELETE LATER
q_MyCode=np.zeros((0,2))
L2_array=np.array([])
i=dist
pos_s=np.array([[0.5*L-i/2, 0.5*L],[0.5*L+i/2, 0.5*L]])
S=len(pos_s)

K0=1
Rv=L/alpha+np.zeros(S)
K_eff=alpha*K0/(np.pi*L*Rv)
C_v_array=np.ones(S)
chunk='sources'
#%%


r=post.reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, directness)
r.solve_linear_prob(np.zeros(4), C_v_array)
phi_FV=r.phi_FV #values on the FV cells
phi_q=r.phi_q #values of the flux

directory_files='/home/pdavid/Bureau/SS_malpighi/2D_cartesian/Validated_2D_Code/axy_error/'

file=directory_files + 'alpha{}_d{}_'.format(alpha,int(d)) + chunk+ '_table.txt'
df=pandas.read_fwf(file, skiprows=5)
q_COMSOL=np.array(df.columns, dtype=float)


file=directory_files + 'alpha{}_d{}_'.format(alpha,int(d))+ chunk+ '_2D.txt'
df=pandas.read_fwf(file)
ref_data=np.array(df).T #reference 2D data from COMSOL

r.set_up_manual_reconstruction_space(ref_data[0], ref_data[1])
r.reconstruction_manual()
r.reconstruction_boundaries(np.zeros(4))
phi_MyCode=r.u+r.DL+r.SL
  
file_1D=directory_files + 'alpha{}_d{}_'.format(alpha,int(d))+ chunk+ '_1D.txt'
df_1D=pandas.read_fwf(file_1D)
data_1D=np.array(df_1D).T #reference 2D data from COMSOL
r.set_up_manual_reconstruction_space(data_1D[0], np.zeros(len(data_1D[0]))+L/2)
r.reconstruction_manual()
r.reconstruction_boundaries(np.zeros(4))
phi_MyCode_1D=r.u+r.DL+r.SL

fig, axs=plt.subplots(2,3, figsize=(16,8))

col=[  'pink','c', 'blue']
side=(directness+0.5)*h_ss*2
vline=(y_ss[1:]+x_ss[:-1])/2
axs[0,0].scatter(pos_s[:,0], pos_s[:,1], s=100, c='r')
for c in range(len(pos_s)):
    center=pos_to_coords(r.x, r.y, r.s_blocks[c])
    
    axs[0,0].add_patch(Rectangle(tuple(center-side/2), side, side,
                 edgecolor = col[c],
                 facecolor = col[c],
                 fill=True,
                 lw=5, zorder=0))
axs[0,0].set_title("Position of the point sources")
for xc in vline:
    axs[0,0].axvline(x=xc, color='k', linestyle='--')
for xc in vline:
    axs[0,0].axhline(y=xc, color='k', linestyle='--')
axs[0,0].set_xlim([0,L])
axs[0,0].set_ylim([0,L])
axs[0,0].set_ylabel("y ($\mu m$)")
axs[0,0].set_xlabel("x ($\mu m$)")
phi_1D_COMSOL=data_1D[1,:-1].astype(float)
axs[0,1].scatter(data_1D[0,:-1],phi_1D_COMSOL , s=5, label='COMSOL')
axs[0,1].scatter(data_1D[0,:-1],phi_MyCode_1D[:-1], s=5)
axs[0,1].legend()

axs[0,2].scatter(data_1D[0,:-1],np.abs(phi_1D_COMSOL-phi_MyCode_1D[:-1]))

levs=np.linspace(0, np.max(ref_data[2]),100)
axs[1,0].tricontourf(ref_data[0], ref_data[1], ref_data[2],levels=levs)
axs[1,0].set_title("COMSOL")
axs[1,1].tricontourf(ref_data[0], ref_data[1], phi_MyCode,levels=levs)
axs[1,1].set_title("MYCode")
axs[1,2].tricontourf(ref_data[0], ref_data[1], np.abs(ref_data[2]-phi_MyCode),levels=levs/10)

print("relative error for each flux estimation", (phi_q-q_COMSOL)/q_COMSOL)
L2=np.sum((phi_MyCode-ref_data[2])**2/np.sum(ref_data[2]**2))**0.5
print("L2 norm for the $\phi$-field",L2 )
L2_array=np.append(L2_array, L2)

q_MyCode=np.vstack((q_MyCode, phi_q))
#%% Imposed q #%%

b=1/3
q_array=np.ones(S)*b
phi_FV=np.linalg.solve(r.A_matrix, -r.b_matrix.dot(q_array))
real_C_v_array=r.c_matrix.dot(phi_FV)+ r.d_matrix.dot(q_array)
phi_bar=real_C_v_array-b/K0

phi_com=np.array([0.2985])

