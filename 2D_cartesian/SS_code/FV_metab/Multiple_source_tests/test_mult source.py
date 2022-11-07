#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 12:54:16 2021

@author: pdavid
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 19:33:13 2021

@author: pdavid
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:52:20 2021

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

def get_plots_through_sources(phi_mat, SS_phi_mat,pos_s, rec_x,rec_y):
    for i in pos_s:
        pos=coord_to_pos(rec_x, rec_y, i)
        pos_x=int(pos%len(rec_x))
        plt.plot(rec_y, phi_mat[:,pos_x], label="coupling")
        plt.plot(rec_y, SS_phi_mat[:,pos_x], label="validation")
        plt.axvline(x=i[1])
        plt.legend()
        plt.show()
        
def get_plots_through_sources_peaceman(phi_mat,peaceman,pos_s, rec_x,rec_y):
    for i in pos_s:
        pos=coord_to_pos(rec_x, rec_y, i)
        pos_x=int(pos%len(rec_x))
        plt.plot(rec_y, phi_mat[:,pos_x], label="coupling")
        plt.scatter(rec_y, peaceman[:,pos_x], label="validation")
        plt.plot()
        plt.axvline(x=i[1])
        plt.legend()
        plt.show()

#0-Set up the sources
#1-Set up the domain
D=1
L=10
cells=10
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=12
#Rv=np.exp(-2*np.pi)*h_ss
Rv=0.1

C0=2*np.pi*Rv*D
K_eff=C0/(np.pi*Rv**2)

#%%


validation=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=1
print("directness=", directness)

#Position image
# =============================================================================
# pos_s1=np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.96,0.43]])
# pos_s2=np.array([[0.27,0.6],[0.55,0.65],[0.59,0.66],[0.67,0.67],[0.13,0.75],[0.15,0.93],[0.2,0.87],[0.28,0.98],[0.8,0.85],[0.83,0.92]])
# pos_s=(np.concatenate((pos_s1, pos_s2))*0.8+0.1)*L
# =============================================================================

a=1.8*Rv
pos_s=np.array([[0.5-a/(2*L), 0.5],[0.5+a/(2*L), 0.5]])*L

#pos_s=np.array([[1,1]])*L/2

S=len(pos_s)
plt.scatter(pos_s[:,0], pos_s[:,1])
plt.title("Position of the point sources")
plt.ylabel("y ($\mu m$)")
plt.xlabel("x ($\mu m$)")
plt.show()

#%%


t=assemble_SS_2D_FD(pos_s, Rv, h_ss,L, K_eff, D, directness)
t.pos_arrays()
t.initialize_matrices()
M=t.assembly_sol_split_problem(np.array([0,0,0,0]))
t.B[-S:]=np.ones(S)
#t.B[-np.random.randint(0,S,int(S/2))]=0
sol=np.linalg.solve(M, t.B)
phi_FV=sol[:-S].reshape(len(t.x), len(t.y))
phi_q=sol[-S:]

# =============================================================================
# m=real_NN_rec(t.x, t.y, sol[:-len(pos_s)], t.pos_s, t.s_blocks, sol[-len(pos_s):], ratio, t.h, 1, t.Rv)
# m.add_singular(1)
# fin_rec=m.add_singular(1)+m.rec
# plt.imshow(fin_rec, origin='lower'); plt.colorbar()
# plt.show()
# print(fin_rec[:,-1])
# =============================================================================


#%%
#Reconstruction microscopic field
#pdb.set_trace()
a=post.reconstruction_sans_flux(sol, t, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()
plt.imshow(a.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model linear")
plt.colorbar(); plt.show()



#%%
#Validation Solution Splitting
SS=full_ss(pos_s, Rv, h_ss/ratio, K_eff, D, L)
v_SS=SS.solve_problem(t.B[-S:])
phi_SS=SS.reconstruct(np.ndarray.flatten(v_SS), SS.phi_q)


#%%
plt.imshow(phi_SS, extent=[-L/2,L/2, -L/2, L/2], origin='lower')
plt.title("validation reconstruction linear" )
plt.colorbar()
plt.show()



plt.scatter(np.arange(len(SS.phi_q)),np.abs(SS.phi_q-phi_q), label="relative error")
plt.plot(SS.phi_q, label="absolute value flux")
plt.title("absolute error of the flux estimation for ratio={} linear".format(ratio))
plt.ylabel("absolute value [$kg m^{-1} s^{-1}$]")
plt.xlabel("source ID")
plt.legend()
plt.show()


plt.scatter(np.arange(len(SS.phi_q)),np.abs(SS.phi_q-phi_q)/np.abs(SS.phi_q))
plt.title("relative error")
plt.show()


plt.imshow(a.rec_final-phi_SS, origin='lower')
plt.title("absolute error of the reconstructed $\phi$ linear")
plt.colorbar(); plt.show()



#%%

get_plots_through_sources(a.rec_final, phi_SS, pos_s, SS.x, SS.y)

print("L2 norm SS q=",get_L2(a.phi_q, SS.phi_q))
print("relative error with SS q= ", get_MRE(a.phi_q, SS.phi_q))
print("relative L2 norm with SS concentration field=",get_L2(np.ndarray.flatten(phi_SS), np.ndarray.flatten(a.rec_final)))


# =============================================================================
# MRE=np.array([])
# for cells in np.arange(3,25):
#     print(cells)
#     ratio=int(40/cells)
#     ratio*=2
#     h_ss=L/cells
#     directness=int(np.around(1/h_ss))
#     print(directness)
#     
#     x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
#     y_ss=x_ss
#     phi_FV, phi_q=solve_problem_model(pos_s, Rv, h_ss, x_ss, y_ss, K_eff, D, directness)
#     SS_phi_FV, SS_phi_q=get_SS_validation(pos_s, Rv, h_ss, ratio, K_eff, D, L, np.ones(S)*C0)
#     MRE=np.append(MRE,get_MRE(SS_phi_q, phi_q))
#                                     
# =============================================================================


#%%

#comparison with no coupling
ref_FV=FV_reference(ratio, t.h, pos_s, np.ones(S), D, K_eff, Rv, L)
noc_sol, noc_lenx, noc_leny,noc_q, noc_B, noc_A, noc_s_blocks,noc_x,noc_y=ref_FV.sol, len(ref_FV.x), len(ref_FV.y), ref_FV.q_array, ref_FV.B, ref_FV.A, ref_FV.s_blocks, ref_FV.x, ref_FV.y


# =============================================================================
# print("relative L2 norm of no coupling=",get_L2(SS.phi_q, noc_q))
# print("relative error with peaceman q= ", get_MRE(SS.phi_q, noc_q))
# print("L2 norm concentration peaceman field=",get_L2(np.ndarray.flatten(np.ndarray.flatten(phi_SS)), noc_sol))
# =============================================================================

#comparison with Peaceman
ref_P=FV_reference(ratio, t.h, pos_s, np.ones(S), D, K_eff, Rv, L, "Peaceman")
p_sol, p_lenx, p_leny,p_q, p_B, p_A, p_s_blocks,p_x,p_y=ref_P.sol, len(ref_P.x), len(ref_P.y), ref_P.q_array, ref_P.B, ref_P.A, ref_P.s_blocks, ref_P.x, ref_P.y



# =============================================================================
# print("relative L2 norm with peaceman q=",get_L2(a.phi_q, p_q))
# print("relative error with peaceman q= ", get_MRE(a.phi_q, p_q))
# print("L2 norm concentration peaceman field=",get_L2(np.ndarray.flatten(a.rec_final), p_sol))
# 
# =============================================================================
errors=[["coupling","SS" , ratio , get_L2(SS.phi_q, phi_q) , get_L2(phi_SS, a.rec_final) , get_MRE(SS.phi_q, phi_q) , get_MRE(phi_SS, a.rec_final)],
        ["coupling","Peaceman", ratio,get_L2(p_q, phi_q), get_L2(p_sol, np.ndarray.flatten(a.rec_final)), get_MRE(p_q, phi_q), get_MRE(p_sol, np.ndarray.flatten(a.rec_final))],
        ["FV","SS",1,get_L2(SS.phi_q, noc_q), get_L2(np.ndarray.flatten(phi_SS), noc_sol), get_MRE(SS.phi_q, phi_q), get_MRE(np.ndarray.flatten(phi_SS), noc_sol)],
        ["FV","Peaceman",1,get_L2(p_q, noc_q), get_L2(p_sol, noc_sol), get_MRE(p_q, phi_q), get_MRE(p_sol, noc_sol)],
        ["Peaceman","SS", 1,get_L2(SS.phi_q, p_q), get_L2(np.ndarray.flatten(phi_SS), p_sol), get_MRE(SS.phi_q, p_q), get_MRE(np.ndarray.flatten(phi_SS), p_sol)]]
        
#%%
from tabulate import tabulate
print(tabulate(errors, headers=["Evaluated model","Validation", "ratio","L^2(q)", "L^2(phi)", "MRE(q)", "MRE(phi)"]))



#%%
#PLOTS FOR THE ABSTRACT
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.size'] = '12'
error_Peac=np.abs((SS.phi_q-p_q)/SS.phi_q)
error_FV=np.abs((SS.phi_q-noc_q)/SS.phi_q)
error_coup=np.abs((SS.phi_q-phi_q)/SS.phi_q)
fig, axs = plt.subplots(1,3, figsize=(18,6))
fig.tight_layout(pad=8.0)
axs[2].scatter(error_FV, error_coup)
axs[2].plot(np.linspace(0, np.max(error_coup)),np.linspace(0, np.max(error_coup)), 'k--')
axs[2].set_ylabel("coarse mesh \n with coupling (%)")
axs[2].set_xlabel("fine mesh \n without coupling (%)")


b=axs[1].imshow(a.rec_final, origin='lower', extent=[0,L,0,L])
axs[1].set_xlabel("$\mu$m")
axs[1].set_ylabel("$\mu$m")
#plt.title("reconstruction coupling")
divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size='10%', pad=0.05)
fig.colorbar(b, cax=cax, orientation='vertical')

NN=post.coarse_NN_rec(t.x, t.y, phi_FV, pos_s, t.s_blocks, phi_q, ratio, h_ss, directness, Rv)

c=axs[0].imshow(NN, origin='lower', extent=[0,L,0,L])
axs[0].set_xlabel("$\mu$m")
axs[0].set_ylabel("$\mu$m")
#plt.title("average cell values")
divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='10%', pad=0.05)
fig.colorbar(c, cax=cax, orientation='vertical')


pos_y=np.array([], dtype=int)
for i in range(len(pos_s)):
    pos_y=np.append(pos_y,coord_to_pos(a.x,a.y, pos_s[i])//len(a.x))
plt.figure()
plt.plot(a.x, a.rec_final[pos_y[0],:])
