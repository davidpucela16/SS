#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:38:00 2022

@author: pdavid
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('/home/pdavid//Bureau/Code/SS/2D_cartesian/SS_code/FV_metab')
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


# In[Set up Geometry]:


#0-Set up the sources
#1-Set up the domain
alpha=200

D=1
L=240
cells=5
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=8
#Rv=np.exp(-2*np.pi)*h_ss
Rv=L/alpha

C0=2*np.pi
K_eff=C0/(np.pi*Rv**2)


x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
y_ss=x_ss
directness=2


pos_s=np.array([[0.5,0.5]])*L
S=len(pos_s)

vline=(y_ss[1:]+x_ss[:-1])/2
plt.scatter(pos_s[:,0], pos_s[:,1])
plt.title("Position of the point sources")
for xc in vline:
    plt.axvline(x=xc, color='k', linestyle='--')
for xc in vline:
    plt.axhline(y=xc, color='k', linestyle='--')
plt.xlim([0,L])
plt.ylim([0,L])
plt.ylabel("y ($\mu m$)")
plt.xlabel("x ($\mu m$)")

C_v_array=np.ones(S)  

def get_plots_through_sources(phi_mat, SS_phi_mat ,pos_s, rec_x,rec_y):
    for i in pos_s:
        pos=coord_to_pos(rec_x, rec_y, i)
        pos_x=int(pos%len(rec_x))
        plt.plot(rec_y, phi_mat[:,pos_x], label="coupling")
        plt.plot(rec_y, SS_phi_mat[:,pos_x], label="validation")
        plt.axvline(x=i[1])
        plt.legend()
        plt.show()
        
        
# Assembly and resolution of the coupling model 

# In[Set up MyCode]:


t=assemble_SS_2D_FD(pos_s, Rv, h_ss,L, K_eff, D, directness) #Initialises the variables of the object
t.pos_arrays() #Creates the arrays that contain the information about the sources 
t.initialize_matrices() 

M=t.assembly_sol_split_problem(np.array([0,0,0,0])) #The argument is the value of the Dirichlet BCs

t.B[-S:]=C_v_array #Robin Boundary condition


# Resolution of the coupling model
sol=np.linalg.solve(M, t.B)
phi_FV=sol[:-S].reshape(len(t.x), len(t.y))
phi_q=sol[-S:]

print("MyCode q value: ", phi_q)


# In[Refined Peaceman Validation]:

peac=get_validation(ratio, t, pos_s, C_v_array, D, K_eff, Rv, L)
sol_FV, len_x_FV, len_y_FV,FV_q_array, FV_B, FV_A, FV_s_blocks,FV_x,FV_y = peac


print("The L2 error with the Peaceman model is:",get_L2(FV_q_array,phi_q ))
L_r=0.56*L
q_analyt=2*np.pi*C_v_array/(2*np.pi/C0-np.log((Rv/L_r)))


# In[Bilinear reconstruction of the microscopic field ]:


#Reconstruction microscopic field
#pdb.set_trace()
a=post.reconstruction_sans_flux(sol, t, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()
plt.imshow(a.rec_final,extent=[0,L,0,L], origin='lower'); plt.colorbar();
plt.title("reconstruction of the coupling model")
plt.show()


get_plots_through_sources(a.rec_final, sol_FV.reshape(len_x_FV,len_y_FV),pos_s, FV_x, FV_y)


# In[Validation refined Solution Splitting]:

#Validation Solution Splitting
SS=full_ss(pos_s, Rv, h_ss/ratio, K_eff, D, L)
v_SS=SS.solve_problem(t.B[-S:])
phi_SS=SS.reconstruct(np.ndarray.flatten(v_SS), SS.phi_q)

plt.imshow(phi_SS, extent=[0,L,0,L],origin='lower');plt.colorbar()
plt.title("Validation refined full solution splitting")
plt.show()


plt.imshow(a.rec_final-phi_SS,extent=[0,L,0,L], origin='lower'); plt.colorbar();
plt.title("Validation - coupling model")
plt.show()




# In[ ]:



get_plots_through_sources(a.rec_final, phi_SS,pos_s, SS.x, SS.y)




print("The L2 error with the Peaceman model is:",get_L2(FV_q_array,phi_q ))
print("The L2 error with the refined SS model is:",get_L2(SS.phi_q,phi_q))
