#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:25:40 2021

@author: pdavid
"""

import numpy as np 
import matplotlib.pyplot as plt

hr=0.1
Rv=hr
L=10
r=np.arange(Rv+hr/2, L,hr)

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



Lap=get_1D_lap_operator(r, hr, 1)

Lap[-1,:]=0
Lap[-1,-1]=1

RHS=np.zeros(len(r))
RHS[0]=-1

sol=np.linalg.solve(Lap,RHS)

plt.plot(r,sol)
        