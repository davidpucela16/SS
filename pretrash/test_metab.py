#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:18:37 2022

@author: pdavid
"""
import numpy as np 
import matplotlib.pyplot as plt
from module_2D_coupling_FV_nogrid import * 
import reconst_and_test_module as post
import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
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
cells=10
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=5
#Rv=np.exp(-2*np.pi)*h_ss
Rv=0.01
C0=1
K_eff=1/(np.pi*Rv**2)


validation=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=1
print("directness=", directness)
#pos_s=np.array([[x_ss[2], y_ss[2]],[x_ss[4], y_ss[4]]])
#pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6],[2,2]])-np.array([0.25,0.25])
#pos_s/=2
#pos_s=np.array([[1.25,1.25],[1.25,1.75], [1.75,1.75],[1.75,1.25]])
#pos_s=np.array([[4.3,4.3],[4.3,5.5], [3.5,4.5],[3.5,3.5]])


#pos_s=np.array([[0.41,0.41],[0.56,0.56],[0.3,0.7]])*L
pos_s=np.array([[0.51,0.51]])*L
S=len(pos_s)

#Position image

plt.scatter(pos_s[:,0], pos_s[:,1])
plt.title("Position of the point sources")
plt.ylabel("y ($\mu m$)")
plt.xlabel("x ($\mu m$)")
plt.show()
                              
    

n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
n.solve_linear_prob(np.zeros(4))

b=post.reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(n.phi_FV), n.phi_q)), n, L,ratio, directness)
p=b.reconstruction()   
b.reconstruction_boundaries(np.array([0,0,0,0]))
b.rec_corners()
plt.imshow(b.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Steady State ")
plt.colorbar(); plt.show()

M=0.2
phi_0=0.4

p=np.linspace(0,1,100)
if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")

n.Full_Newton(np.ndarray.flatten(n.phi_FV) , np.ndarray.flatten(n.phi_q), 0.005, M, phi_0)
a=post.reconstruction_sans_flux(n.phi[-1], n, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()
plt.imshow(a.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Metabolism")
plt.colorbar(); plt.show()


from module_2D_coupling_FV_nogrid import * 
FV=FV_validation(L, cells*ratio, pos_s, np.ones(S), D, K_eff, Rv)
FV_linear=FV.solve_linear_system()
FV_non_linear=FV.solve_non_linear_system(phi_0,M, 10)
phi_FV=(FV.phi[-1]+FV.Corr_array).reshape(cells*ratio, cells*ratio)
plt.imshow(phi_FV, origin='lower')
plt.title("FV metab reference")
plt.colorbar(); plt.show()

#manual q
q_array=-np.dot(FV.A_virgin.toarray()[FV.s_blocks,:],FV.phi[-1])*FV.h**2/D+M*(1-phi_0/(FV.phi[-1, FV.s_blocks]+FV.Corr_array[FV.s_blocks]+phi_0))

plt.plot(phi_FV[int(cells*ratio/2),:])
plt.plot(a.rec_final[int(cells*ratio/2),:])
plt.plot(b.rec_final[int(cells*ratio/2),:])
plt.show()