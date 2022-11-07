# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.

Created 22 August 2022 solving the heat equation.
The goal is to compare the SS with the temporal Green's function
"""


import numpy as np 
import matplotlib.pyplot as plt 
import scipy.special as spc

import matplotlib.animation as animation
import pdb

# =============================================================================
# k = 2*np.pi
# w = 2*np.pi
# dt = 0.01
# 
# x = np.linspace(0, 3, 151)
# 
# for i in range(50):
#     t = i * dt
#     y = np.cos(k*x - w*t)
#     if i == 0:
#         line, = plt.plot(x, y)
#     else:
#         line.set_data(x, y)
#     plt.pause(0.01) # pause avec duree en secondes
#     
# plt.show()
# 
# =============================================================================

n=2 #dimensions of the problem 

#let's consider symmetry, therefore cylindrical/espherical coordinates
L=10
N=100
h=L/N
r=np.linspace(h/2, L-h/2, N)

T=0.1
N_t=100
t=np.linspace(0,T,N_t)

Rv=L/100


#%%
def plot_dynamic(r, array):
    plt.figure()
    plt.ylim(0,np.max(array))
    c=0
    line,=plt.plot(r,array[0])
    for i in array[:,0]:
        line.set_data(r, array[c])
        plt.pause(0.01) # pause avec duree en secondes
        c+=1
    
    plt.show()

def oneD_cyl_Laplacian(fluxes, ri, inc_r, D):
    """Returns the stensil for the fluxes in a cylindrical reference system. Two coordinates, 
    longitudinal and radial. Depends on the radial position, the coefficients in the radial 
    direction (north, south), will differ. """
    Diff_flux_east, Diff_flux_west=fluxes
    cntrl,E,W=float(0),float(0),float(0)
    if Diff_flux_east:

        cntrl-=(ri+inc_r/2)*D/(ri*inc_r**2)
        E+=(ri+inc_r/2)*D/(ri*inc_r**2)
        
    if Diff_flux_west:
        cntrl-=(ri-inc_r/2)*D/(ri*inc_r**2)
        W+=(ri-inc_r/2)*D/(ri*inc_r**2)
    
        
    co=np.array([cntrl,E,W])
    return(co)

def get_1D_lap_operator(r, inc_r, D):
    """computes the fluxes with zero flow BC"""
    OP=np.zeros([len(r),len(r)])
    OP[0,0]=-2/inc_r**2
    OP[0,1]=2/inc_r**2
    factor_last=(r[-1]-inc_r/2)/(2*r[-1]*inc_r**2)
    OP[-1,-1]=-factor_last
    OP[-1,-2]=factor_last
    for j in range(len(r)-2):
        i=j+1
        coeffs=oneD_cyl_Laplacian([True, True], r[i], inc_r, D)
        OP[i,i]=coeffs[0]
        OP[i, i+1]=coeffs[1]
        OP[i,i-1]=coeffs[2]
        
    return(OP)




#%% - Implicit resolution 

