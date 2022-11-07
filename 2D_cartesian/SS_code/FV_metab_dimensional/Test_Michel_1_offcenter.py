#!/usr/bin/env python
# coding: utf-8

# Let's suppose a single source embedded in a region of tissue ($\Omega_\sigma$):
# \begin{cases}
# \Delta \phi = 0 \quad  in \quad \Omega_{\sigma}\\
# \\
#   \phi = 0 \quad at \quad  \partial \Omega_{\sigma}\\
#   \\
#   \nabla \phi \cdot \mathbf{n}_{\sigma} = \dfrac{1}{2} K_{eff} R (\overline{\varphi} - \overline{\phi}) \quad at \quad \partial \Omega_{\sigma \beta}
# \end{cases}  
# <br>
# <br>
# Given the following definitions: <br>
# $\Omega_{\sigma} \rightarrow $ Brain parenchyma <br>
# $\Omega_{\beta} \rightarrow$ Vascular space <br>
# $\partial \Omega_{\beta \sigma} \rightarrow$ Intersection between the vascular space and the parenchyma (vessel wall) <br>
# $\phi \rightarrow$ Concentration field <br>
# $\mathbf{n}_{\sigma} \rightarrow$ normal vector pointing outwards of the $\Omega_{\sigma}$ space<br>
# <br>
# In the Robin boundary condition, we use the following variables: <br>
# $\bar{\varphi} \rightarrow$ cross-section averaged concentration inside the vessel following the work of [Berg et al., 2019] <br>
# $\bar{\phi} \rightarrow$ Approximation of the average concentration at the vessel wall. The work of [Berg et al., 2019] provides an effective intravascular equation which considers the intravascular radial concentration gradients for a constant concentration around the vessel wall, that is $\phi(\mathbf{x}) = \bar{\phi} = constant \quad for \quad \mathbf{x} \in \partial \Omega{\sigma \beta}$ <br>
# $R \rightarrow$ Radius of the vessel (radius of the espace $\Omega_{\beta}$)
# 
# 
# 
# with the integral of the flux being exchanged between the source and the parenchyma:,
# <br>
# $$q=K_{eff} \pi R^2 (\bar{\varphi} - \bar{\phi})$$
# 

# For the resolution of the problem given the hypothesis of an axylsymmetric flux given through the Robin BC, we can easily split the concentration field into two:
# 
# $$\phi = S + v $$
# 
# \begin{cases}
# \Delta S = 0 \quad in  \quad \Omega_{\sigma} \\
# \nabla S \cdot \mathbf{n}_{\sigma} = \dfrac{q}{2 \pi R} \quad at \quad \partial \Omega_{\sigma \beta}
# \end{cases}
# 
# \begin{cases}
# \Delta v = 0 \quad in  \quad \Omega_{\sigma} \\
# v = f(\mathbf{x}) \quad at \quad  \partial \Omega_{\sigma}\\
# \end{cases}
# 
# This way, even though the Robin boundary condition is not satisfied punctually throughout $\partial \Omega_{\beta \sigma}$, it is satisfied on average:
# $$\oint_{\Omega_{\beta \sigma}} \nabla v \cdot \mathbf{n}_{\sigma}dS = 0$$
# $$\oint_{\Omega_{\beta \sigma}} \nabla (v + S)  \cdot \mathbf{n}_{\sigma}dS =  q $$

# It is easy to find the analytical form of the singular term S:
# $$S= - \dfrac{q}{2 \pi} log(\dfrac{1}{||\mathbf{x}_c - \mathbf{x}||}) + C \quad for \quad ||\mathbf{x}_c - \mathbf{x}|| > R$$
# 
# Where $\mathbf{x}_c$ is the position of the center of the circle $\Omega_\beta$, and C represents a constant that appears through the integration of the equation for S; it is given the arbitrary value so the singular term reduces to zero at the vessel wall. Therefore:
# 
# $$S= - \dfrac{q}{2 \pi} log(\dfrac{R}{||\mathbf{x}_c - \mathbf{x}||}) \quad for \quad ||\mathbf{x}_c - \mathbf{x}|| > R$$
# 

# We can define now the value of the function $f(\mathbf{x})$ so it satisfies the boundary condition at $\partial{\Omega_\sigma}$:
# 
# $$f(\mathbf{x})=\dfrac{q}{2 \pi} log(\dfrac{R}{||\mathbf{x}_c - \mathbf{x}||})$$

# The perturbation on the concentration field due to the source is now taken into account by the singular term, which is written as a function of q. This q appears implicit at the Robin boundary condition in the original problem.
# 
# The numerically resolved problem is the one in v, however its BVP is dependent on the value of S at the boundary, which itself depends on q. To be able to solve the problem, we need to explicit q as a function of v.
# 
# $$q=C_0 (\bar{\varphi} - \bar{\phi})$$
# 
# $$\bar{\phi} \approx \dfrac{1}{2 \pi R} \oint_{\partial \Omega_\beta} \phi(\mathbf{x}) dS $$
# 
# $$S(\mathbf{x}_c; \mathbf{x}) = 0 \quad at \quad \mathbf{x} \in \partial \Omega_{\beta \sigma}$$
# 
# Therefore, the value can be estimated solely through v:
# $$\bar{\phi} \approx \dfrac{1}{2 \pi R} \oint_{\partial \Omega_\beta} v(\mathbf{x}) dS $$
# 
# Since v is expected to be very smooth: 
# 
# $$\dfrac{1}{|\partial \epsilon|} \oint_{\partial \epsilon} u dS \approx \dfrac{1}{|\Omega_k|} \iint v dV \quad for \quad \epsilon \in \Omega_k$$
# 
# So we can write:
# 
# $$\bar{\phi} \approx \dfrac{1}{|\Omega_k|} \iint v dV$$ as long as $\Omega_\beta \in \Omega_k$ 
# 
# In the future, $\Omega_k$ will be defined as the finite volume containing the source, so the value of the flux can be calculated as a function of the value of the regular term in the cell that contains it as the following:
# $$q = C_0(\bar{\varphi} - \widetilde{v}_k)$$
# 
# Where $\widetilde{v}_k$ is meant to represent the discrete value of the regular term in the finite volume cell k.

# For the single source problem, the following problem is solved:
# \begin{cases}
# \Delta v = 0 \quad in  \quad \Omega_{\sigma} \\
# v = f(\mathbf{x}) \quad at \quad  \partial \Omega_{\sigma}\\
# \end{cases}
# 
# with:
# 
# $$f(\mathbf{x})=\dfrac{q}{2 \pi} log(\dfrac{R}{||\mathbf{x}_c - \mathbf{x}||})$$
# 
# and:
# $$q = C_0(\bar{\varphi} - \widetilde{v}_k)$$
# 
# Where the PDE is resolved through FV, therefore the discrete values for v are given by:
# $$
# \iint_{\Omega_k} v dV = \widetilde{v}_k
# $$
# for a finite volume occupying $\Omega_k$

# In[ ]:





# In[ ]:



import numpy as np 
import matplotlib.pyplot as plt
from module_2D_coupling_FV_nogrid import * 
import reconst_and_test_module as post
import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg



# In[ ]:


#0-Set up the sources
#1-Set up the domain
D=1
L=5
cells=5
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=20
#Rv=np.exp(-2*np.pi)*h_ss
Rv=0.01
C0=2*np.pi
K_eff=C0/(np.pi*Rv**2)


x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
y_ss=x_ss
directness=2


pos_s=np.array([[0.5,0.5]])*L
S=len(pos_s)

print(pos_s)
print(x_ss)


# Assembly and resolution of the coupling model 

# In[ ]:


t=assemble_SS_2D_FD(pos_s, Rv, h_ss,L, K_eff, D, directness) #Initialises the variables of the object
t.pos_arrays() #Creates the arrays that contain the information about the sources 
t.initialize_matrices() 

M=t.assembly_sol_split_problem(np.array([0,0,0,0])) #The argument is the value of the Dirichlet BCs

t.H0[-S:]=-np.ones(S)*C0 #Robin Boundary condition


# Resolution of the coupling model

# In[ ]:


sol=np.linalg.solve(M, t.H0)
phi_FV=sol[:-S].reshape(len(t.x), len(t.y))
phi_q=sol[-S:]


# Validation for large scale discrepancies between the diameter of the vessel and the domain. This is necessary since the vessel is approximated as a delta function

# In[ ]:


#Validation Solution Splitting
SS=full_ss(pos_s, Rv, h_ss/ratio, K_eff, D, L)
v_SS=SS.solve_problem(t.H0[-S:])
phi_SS=SS.reconstruct(np.ndarray.flatten(v_SS), SS.phi_q)


# In[ ]:


plt.imshow(phi_SS, extent=[0,L,0,L],origin='lower');plt.colorbar()
plt.title("Validation refined full solution splitting")
plt.show()


# Bilinear reconstruction of the microscopic field 

# In[ ]:


#Reconstruction microscopic field
#pdb.set_trace()
a=post.reconstruction_sans_flux(sol, t, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()
plt.imshow(a.rec_final,extent=[0,L,0,L], origin='lower'); plt.colorbar();
plt.title("reconstruction of the coupling model")
plt.show()


plt.imshow(a.rec_final-phi_SS,extent=[0,L,0,L], origin='lower'); plt.colorbar();
plt.title("Validation - coupling model")
plt.show()



# In[ ]:


peac=get_validation(ratio, t, pos_s, np.array([1]), D, K_eff, Rv, L)
sol_FV, len_x_FV, len_y_FV,FV_q_array, FV_B, FV_A, FV_s_blocks,FV_x,FV_y = peac


# In[ ]:


def get_plots_through_sources(phi_mat, SS_phi_mat ,pos_s, rec_x,rec_y):
    for i in pos_s:
        pos=coord_to_pos(rec_x, rec_y, i)
        pos_x=int(pos%len(rec_x))
        plt.plot(rec_y, phi_mat[:,pos_x], label="coupling")
        plt.plot(rec_y, SS_phi_mat[:,pos_x], label="validation")
        plt.axvline(x=i[1])
        plt.legend()
        plt.show()
        
        


get_plots_through_sources(a.rec_final, phi_SS,pos_s, SS.x, SS.y)


print("The L2 error with the Peaceman model is:",get_L2(FV_q_array,phi_q ))
print("The L2 error with the refined SS model is:",get_L2(SS.phi_q,phi_q))


# =============================================================================
# pos=coord_to_pos(SS.x, SS.y, pos_s[0])
# pos_x=int(pos%len(SS.x))
# sol_FV.reshape(len_x_FV, len_y_FV)[49,pos_x]=0
# plt.plot(SS.y, a.rec_final[:,pos_x], label="coupling")
# plt.plot(SS.y, phi_SS[:,pos_x], label="validation")
# plt.plot(SS.y, sol_FV.reshape(len_x_FV, len_y_FV)[:,pos_x], label="Peaceman")
# 
# 
# =============================================================================
# OFF CENTERING



def full_L2_comarison(pos_s, Rv, h_ss, x_ss, y_ss, K_eff, D, directness, C0, ratio, L):

    t=assemble_SS_2D_FD(pos_s, Rv, h_ss,L, K_eff, D, directness) #Initialises the variables of the object
    t.pos_arrays() #Creates the arrays that contain the information about the sources 
    t.initialize_matrices() 

    M=t.assembly_sol_split_problem(np.array([0,0,0,0])) #The argument is the value of the Dirichlet BCs

    t.H0[-S:]=-np.ones(S)*C0 #Robin Boundary condition

    sol=np.linalg.solve(M, t.H0)
    phi_FV=sol[:-S].reshape(len(t.x), len(t.y))
    phi_q=sol[-S:]

    #Validation Solution Splitting
    SS=full_ss(pos_s, Rv, h_ss/ratio, K_eff, D, L)
    v_SS=SS.solve_problem(t.H0[-S:])
    phi_SS=SS.reconstruct(np.ndarray.flatten(v_SS), SS.phi_q)

    plt.imshow(phi_SS,extent=[0,L,0,L] ,origin='lower');plt.colorbar()
    plt.show()


    #Reconstruction microscopic field
    #pdb.set_trace()
    a=post.reconstruction_sans_flux(sol, t, L,ratio, directness)
    p=a.reconstruction()   
    a.reconstruction_boundaries(np.array([0,0,0,0]))
    a.rec_corners()
    
    plt.imshow(a.rec_final,extent=[0,L,0,L], origin='lower'); plt.colorbar();
    plt.title("reconstruction of the coupling model")
    plt.show()
    
    
    plt.imshow(a.rec_final-phi_SS,extent=[0,L,0,L], origin='lower'); plt.colorbar();
    plt.title("Validation - coupling model")
    plt.show()

    peac=get_validation(ratio, t, pos_s, np.ones([S]), D, K_eff, Rv, L)
    sol_FV, len_x_FV, len_y_FV,FV_q_array, FV_B, FV_A, FV_s_blocks,FV_x,FV_y = peac


    get_plots_through_sources(a.rec_final, phi_SS,  pos_s, SS.x, SS.y)

    print("The L2 error with the Peaceman model is:",get_L2(FV_q_array,phi_q ))
    print("The L2 error with the refined SS model is:",get_L2(SS.phi_q,phi_q))
    return(FV_q_array, phi_q)


# Let's evaluate the off-centering

# In[118]:


points=5
off=np.linspace(0,h_ss/2,points)*0.95
matrix_L2_error_Peac_off=np.zeros((points, points))
matrix_L2_error_SS_off=np.zeros((points, points))
ci=0
for i in off:
    cj=0
    for j in off:
        pos_s=np.array([[0.5,0.5]])*L+np.array([i,j])
        L2_errors=full_L2_comarison(pos_s, Rv, h_ss, x_ss, y_ss, K_eff, D, directness, C0, 10, L)
        matrix_L2_error_Peac_off[cj,ci]=L2_errors[0]
        matrix_L2_error_SS_off[cj,ci]=L2_errors[1]
        cj+=1
    ci+=1

plt.imshow(matrix_L2_error_Peac_off-matrix_L2_error_SS_off, extent=[0,0.5,0,0.5],origin='lower'); plt.colorbar()
plt.title("err abs")
plt.show()

plt.imshow((matrix_L2_error_Peac_off-matrix_L2_error_SS_off)/0.5, extent=[0,0.5,0,0.5],origin='lower'); plt.colorbar()
plt.title("err relative approxime")
plt.show()


plt.imshow(matrix_L2_error_SS_off, extent=[0,0.5,0,0.5],origin='lower'); plt.colorbar()
plt.title("")
plt.show()


