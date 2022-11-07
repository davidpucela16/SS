#!/usr/bin/env python
# coding: utf-8

# 

# In[291]:


import numpy as np 
import matplotlib.pyplot as plt
from module_2D_coupling_FV_nogrid import * 
import reconst_and_test_module as post
import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg

plt.rcParams.update({'font.size': 24})
plt.rcParams["figure.figsize"] = (12,12)





def execute_full_problem(h_ss,ratio,S, directness):
    #1-Set up the domain
    D=1
    L=6
    #Rv=np.exp(-2*np.pi)*h_ss
    Rv=0.01
    C0=1
    K_eff=1/(np.pi*Rv**2)
    print("D={}, L={}, h_ss={}, ratio={}, #ofSources={}, Rv={}, K_eff={}".format(D, L, h_ss, ratio, S, Rv, K_eff))
    validation=False
    real_Dirich=True
    x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
    y_ss=x_ss

    pos_s=np.random.random((S,2))*2+2

    if real_Dirich:
        A=A_assembly(len(x_ss), len(y_ss))*D/h_ss**2
        #set dirichlet
        B,A=set_TPFA_Dirichlet(0,A, h_ss, get_boundary_vector(len(x_ss), len(y_ss)), np.zeros(len(x_ss)*len(y_ss)),D)
    else:
        A=A_assembly_Dirich(len(x_ss), len(y_ss))*D/h_ss**2
        B=np.zeros(A.shape[0])

    t=assemble_SS_2D_FD(pos_s, A, Rv, h_ss,x_ss,y_ss, K_eff, D,directness)
    t.pos_arrays()
    t.initialize_matrices()
    t.assembly_sol_split_problem()
    B_v=np.zeros(len(t.uni_s_blocks))

    B_q=np.ones(len(t.s_blocks))
    B_q[np.random.randint(S-1, size=S//2)]=0
    B=np.concatenate((B,B_v,B_q))

    phi=np.linalg.solve(t.M, B)
    phi_FV, phi_v, phi_q=post.separate_unk(t, phi)
    phi_mat=phi_FV.reshape(len(x_ss), len(y_ss))

    plt.imshow(phi_mat, origin="lower"); plt.colorbar()
    plt.title("Coupling model: real concentration \n field reconstruction")
    plt.show()
    o=post.reconstruct_coupling(phi, 1, t,1)
    rec=o.reconstruction(ratio)
    
    plt.imshow(rec, extent=[0,L,0,L], origin='lower')
    plt.title("Couplin model: linear flux microscopic\n reconstruction for the regular term")
    plt.colorbar()
    plt.show()
    
    a=full_ss(pos_s, Rv, h_ss/ratio, K_eff, D,L)
    SS=a.solve_problem(B_q)
    SS=a.reconstruct(a.v, a.phi_q)
    plt.show()
    
    plt.imshow(SS, origin="lower", extent=[0,L,0,L]); plt.colorbar()
    title_string="Validation Full_SS \n refined ($h_{SS}=h_{coupling}$" + "*$\dfrac{1}{%d}$)" % ratio
    plt.title(title_string)
    plt.show()
    #reconstruct the full solution splitting
    a.uni_s_blocks=np.array([])
    a.s_blocks=np.array([])
    a.FV_DoF=np.arange(len(a.v))
    
    sol, xlen2, ylen2,q_array,aa, Y, s_b,x_v, y_v=get_validation(ratio, t, pos_s, B_q, D, K_eff, Rv,L)
    plt.imshow(sol.reshape(ylen2, xlen2), origin='lower'); plt.colorbar()
    title_string="Comparison FV \n refined ($h_{FV}=h_{coupling}$" + "*$\dfrac{1}{%d}$)" % ratio
    plt.title(title_string)
    plt.show()
    
    plt.figure()
    title_string="Comparison of flux estimation (q) through \n the three different schemes for \n a ratio in     discretization: $h_{SS}=h_{FV}=h_{coupling}$" + "*$\dfrac{1}{%d}$ \n and %d sources" % (ratio, S)
    plt.plot(phi_q, label="coupling", marker='*')
    plt.plot(a.phi_q, label="Full_SS",marker='*')
    plt.plot(q_array, label="fine FV",marker='*')
    plt.title(title_string)
    plt.legend()
    
    #error plots
    error_fv=(q_array-a.phi_q)/a.phi_q
    error_coup=(phi_q-a.phi_q)/a.phi_q
    plt.figure(figsize=(12,12))


    fig, axs=plt.subplots(1,2, figsize=(16,10))
    fig.suptitle("Flux estimation for S={}".format(S), fontsize=44)
    im=axs[0].plot(np.arange(S),phi_q, label="coupling")
    im=axs[0].plot(np.arange(S),a.phi_q, label="Full_SS")
    im=axs[0].plot(np.arange(S),q_array, label="fine FV ($h_{FV}=h\cdot\dfrac{1}{%d}$)" % ratio)
    axs[0].legend()
    axs[0].set_title("Absolute flux")
    im=axs[1].plot(np.arange(S),np.abs(error_fv), label="fine FV ($h_{FV}=h\cdot\dfrac{1}{%d}$)" % ratio)
    im=axs[1].plot(np.arange(S),np.abs(error_coup), label="coupling")
    axs[1].set_title("relative error: \n finite volumes vs coupling model")
    fig.tight_layout()
    plt.legend()
    plt.show()
    get_L2(a.phi_q, phi_q, "coupling", 1)
    get_L2(a.phi_q, q_array, "FV", ratio)



# In[343]:


#1-Set up the domain
D=1
L=6
h_ss=0.5
#Rv=np.exp(-2*np.pi)*h_ss
Rv=0.01
C0=1
K_eff=1/(np.pi*Rv**2)

validation=False
real_Dirich=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
y_ss=x_ss
directness=1

# In[344]:


S=30
#pos_s=np.random.random((S,2))*2+2
pos_s=np.array([[2.46800991, 2.09449449],
       [3.90002095, 3.11211454],
       [2.67822887, 3.82370644],
       [3.32933332, 2.61850985],
       [2.83209539, 3.43523803],
       [3.63484514, 2.92362814],
       [2.52487727, 3.07519465],
       [2.14376904, 2.36242376],
       [3.87436686, 2.94906691],
       [3.05843617, 2.50598424],
       [2.04155229, 2.60382951],
       [3.26028751, 3.96245056],
       [2.93729555, 3.17421674],
       [2.99241583, 2.87610836],
       [3.07585313, 3.87112026],
       [3.80090451, 2.86566743],
       [2.70159802, 2.16136797],
       [2.52147811, 3.23152061],
       [3.67427251, 2.26975544],
       [2.60612146, 3.23924454],
       [2.0941359 , 3.95437275],
       [3.22942324, 3.62148788],
       [3.35765912, 3.27505237],
       [2.22440178, 2.58977773],
       [3.36220833, 2.96875815],
       [3.71550785, 3.84410271],
       [2.11501359, 2.08997809],
       [3.36264919, 2.49112144],
       [2.88068122, 2.28266017],
       [3.72234344, 2.35824392]])


# In[345]:


if real_Dirich:
    A=A_assembly(len(x_ss), len(y_ss))*D/h_ss**2
    #set dirichlet
    B,A=set_TPFA_Dirichlet(0,A, h_ss, get_boundary_vector(len(x_ss), len(y_ss)), np.zeros(len(x_ss)*len(y_ss)),D)
else:
    A=A_assembly_Dirich(len(x_ss), len(y_ss))*D/h_ss**2
    B=np.zeros(A.shape[0])


# In[346]:


t=assemble_SS_2D_FD(pos_s, A, Rv, h_ss,x_ss,y_ss, K_eff, D,directness)
t.pos_arrays()
t.initialize_matrices()
t.assembly_sol_split_problem()
B_v=np.zeros(len(t.uni_s_blocks))


# In[347]:


B_q=np.ones(len(t.s_blocks))
B_q[np.random.randint(S-1, size=S//2)]=0
B=np.concatenate((B,B_v,B_q))


# In[348]:


phi=np.linalg.solve(t.M, B)
phi_FV, phi_v, phi_q=post.separate_unk(t, phi)
phi_mat=phi_FV.reshape(len(x_ss), len(y_ss))


# In[349]:


plt.imshow(phi_mat, origin="lower"); plt.colorbar()
plt.title("Coupling model: real concentration \n field reconstruction")
plt.show()


# In[350]:


ratio=5


# In[351]:


o=post.reconstruct_coupling(phi, 1, t,directness)
rec=o.reconstruction(ratio)


# In[352]:


plt.imshow(rec, extent=[0,L,0,L], origin='lower')
plt.title("Couplin model: linear flux microscopic\n reconstruction for the regular term")
plt.colorbar()
plt.show()
    


# In[353]:


sol, xlen2, ylen2,q_array,aa, Y, s_b,x_v, y_v=get_validation(ratio, t, pos_s, B_q, D, K_eff, Rv,L)
plt.imshow(sol.reshape(ylen2, xlen2), origin='lower'); plt.colorbar()
title_string="Comparison FV \n refined ($h_{FV}=h_{coupling}$" + "*$\dfrac{1}{%d}$)" % ratio
plt.title(title_string)
plt.show()


# In[354]:



a=full_ss(pos_s, Rv, h_ss/ratio, K_eff, D,L)
SS=a.solve_problem(B_q)
SS=a.reconstruct(a.v, a.phi_q)
plt.show()
plt.imshow(SS, origin="lower", extent=[0,L,0,L]); plt.colorbar()
title_string="Validation Full_SS \n refined ($h_{SS}=h_{coupling}$" + "*$\dfrac{1}{%d}$)" % ratio
plt.title(title_string)
plt.show()
#reconstruct the full solution splitting
a.uni_s_blocks=np.array([])
a.s_blocks=np.array([])
a.FV_DoF=np.arange(len(a.v))


# In[355]:


grads_v=post.reconstruction_gradients_manual(a.v, a, a.boundary)
v_rec=post.reconstruct_from_gradients(a.v, grads_v, 5, a.x, a.y, a.h)


# In[356]:


plt.figure()
plt.plot(phi_q, label="coupling", marker='*')
plt.plot(a.phi_q, label="Full_SS",marker='*')
plt.plot(q_array, label="fine FV",marker='*')
title_string="Comparison of flux estimation (q) through \n the three different schemes for \n a ratio in discretization: $h_{SS}=h_{FV}=h_{coupling}$" + "*$\dfrac{1}{%d}$ \n and %d sources" % (ratio, S)
plt.title(title_string)
plt.legend()


# In[357]:


#error plots
error_fv=(q_array-a.phi_q)/a.phi_q
error_coup=(phi_q-a.phi_q)/a.phi_q


fig, axs=plt.subplots(1,2, figsize=(18,10))
fig.suptitle("Flux comparison", fontsize=24)
im=axs[0].plot(np.arange(S),phi_q, label="coupling")
im=axs[0].plot(np.arange(S),a.phi_q, label="Full_SS")
im=axs[0].plot(np.arange(S),q_array, label="fine FV ({}xfiner mesh)".format(ratio))
axs[0].legend()
axs[0].set_title("Absolute flux")
im=axs[1].plot(np.arange(S),np.abs(error_fv), label="FV({}xfiner mesh)".format(ratio))
im=axs[1].plot(np.arange(S),np.abs(error_coup), label="couping")
plt.title("relative error: \n finite volumes vs coupling model")
plt.legend()


# In[358]:


def get_L2(val, sol, title, ratio): 
    L2=np.sqrt(np.sum((val-sol)**2))
    string="L^2 norm for the " + title + " with {} refinement= {}".format(ratio, L2)
    print(string)
get_L2(a.phi_q, q_array, "FV", ratio)
get_L2(a.phi_q, phi_q, "coupling", 1)


# In[359]:


plt.imshow(SS-sol.reshape(xlen2, ylen2) , origin="lower", extent=[0,L,0,L]); plt.colorbar(); plt.title("Comparison FV with SS")
plt.show()

# In[363]:


plt.imshow(SS-rec , origin="lower", extent=[0,L,0,L]); plt.colorbar()
plt.show()

# Plots through the surfaces

# =============================================================================
# array_of_rows=np.zeros(len(pos_s), dtype=int)
# c=0
# for i in pos_s:
#     array_of_rows[c]=np.argmin(np.abs(x_v-i[0]))
#     c+=1
# print(array_of_rows)
# 
# 
# 
# 
# c=0
# for i in array_of_rows:
#     plt.figure()
#     plt.plot(y_v, sol.reshape(ylen2,xlen2)[:,i], label="FV")
#     plt.plot(y_v, SS[:,i], label="SS")
#     plt.plot(y_v, rec[:,i], label="coup")
#     plt.scatter(pos_s[c,1], np.array([0.1]),  label="pos_source",s=100, marker="*")
#     plt.axvline(x=pos_s[c,1])
#     plt.title("Refinement={} \n line through source {}, q={}".format(ratio,c, np.around(phi_q[c], decimals=2)))
#     plt.ylabel("$\phi$")
#     plt.xlabel("y")
#     plt.ylim(0,1.1*np.max(SS[:,i]))
#     plt.legend()
#     c+=1
# 
# =============================================================================



