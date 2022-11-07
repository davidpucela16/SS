#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:57:07 2021

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

plt.rcParams["figure.figsize"] = (7,7)

#-------------------------------------------------------
#CYLINDRICAL COORDINATES 
#-------------------------------------------------------

def get_sec_der_kernel_sparse(s, inc_s):
    """Gets the laplacian operator (Finite differences) of a line with continuity of the second
    derivative at the extremities"""
    lens=len(s)
    
    #Diagonal values
    data=np.zeros(lens)-2/(inc_s**2)
    row=np.arange(lens)
    col=np.arange(lens)
    
    col=np.concatenate([col, np.arange(lens-1)])
    row=np.concatenate([row, np.arange(lens-1)+1])
    data=np.concatenate([data, np.zeros(lens-1)+1/(inc_s**2)])
    
    col=np.concatenate([col, np.arange(lens-1)+1])
    row=np.concatenate([row, np.arange(lens-1)])
    data=np.concatenate([data, np.zeros(lens-1)+1/(inc_s**2)])
    
    return(np.vstack([data, row, col]))

def get_sec_der(s, inc_s):
    """Gets the laplacian operator (Finite differences) of a line with continuity of the second
    derivative at the extremities"""
    lens=len(s)
    
    #Diagonal values
    data=np.zeros(lens)-2/(inc_s**2)
    row=np.arange(lens)
    col=np.arange(lens)
    
    col=np.concatenate([col, np.arange(lens-1)])
    row=np.concatenate([row, np.arange(lens-1)+1])
    data=np.concatenate([data, np.zeros(lens-1)+1/(inc_s**2)])
    
    col=np.concatenate([col, np.arange(lens-1)+1])
    row=np.concatenate([row, np.arange(lens-1)])
    data=np.concatenate([data, np.zeros(lens-1)+1/(inc_s**2)])
  
    A=sp.sparse.csc_matrix((data, (row, col)), shape=(lens, lens))
    
    ##Second derivative at the end is continous!!
    A[0,:]=A[1,:]
    A[-1,:]=A[-2,:]
    return(A)

def get_first_der(phi_vessel,inc_s):
    """Gets the first derivative matrix for a 1D line"""
    l=len(phi_vessel)
    
    col=np.arange(l-1)
    row=np.arange(l-1)
    data=np.zeros(l-1)-1/inc_s
    
    col=np.concatenate([col, np.arange(l-1)+1])
    row=np.concatenate([row,np.arange(l-1)])
    data=np.concatenate([data,np.zeros(l-1)+1/inc_s])
    
    #For the last row
    col=np.append(col, [l-2, l-1])
    row=np.append(row, [l-1, l-1])
    data=np.append(data, [-1/inc_s, 1/inc_s])
    
    return(sp.sparse.csc_matrix((data, (row, col)), shape=(l,l)))

def Green(ri, Rv):
    return(np.log(Rv/(ri))/(2*np.pi))

print("Being imported")

    
class simplified_assembly_Laplacian_cylindrical():
    """Functions necessary to assemble the laplacian operator. It will take into account the corners and the boundaries 
    when assembling.
    The cylindrical coordinate system is taken into account by the function OUTSIDE of this class called matrix_coeffs which will
    provide the coefficients of the laplacian stensil given the open boundaries of the control volume"""
    def __init__(self,R_max, Rv, r_points,s_points, L, D):
        self.D=D
        print("simplified_assembly_Laplacian_cylindrical calleeeeeeeeeeeeeeeeeeeeeeed")
        self.inc_r=(R_max-Rv)/(r_points)
        self.inc_s=L/(s_points)
        
        self.r=np.linspace(Rv+self.inc_r/2, R_max-self.inc_r/2, r_points)
        self.s=np.linspace(self.inc_s/2,L-self.inc_s/2,s_points)
        self.Rv=Rv
        
        self.f_step=len(self.s)
        self.total=len(self.s)*len(self.r)
        self.set_boundary_and_inner_matrix()
        self.L=L

        
    def set_boundary_and_inner_matrix(self):
        """This function sets the arrays self.boundaries, self.corners, self.inner/outer/east/west 
        which represent the different boundary vector that we are gonna use. The first one has all the 
        IDs of the boundary cells. The rest are which boundary? information. """
        #The following operations are done in a specific order to accurately compute the corners
        #And the boundary vertices only once
        
        outer_boundary=np.arange(self.total-self.f_step, self.total)
        corners_up=outer_boundary[[0,-1]]
        outer_boundary=outer_boundary[1:-1] #Without the corners!!
        
        inner_boundary=np.arange(self.f_step)
        corners_down=inner_boundary[[0,-1]]
        inner_boundary=inner_boundary[1:-1]
        
        west_boundary=np.arange(0,self.total, self.f_step)
        west_boundary=west_boundary[1:-1]

        east_boundary=np.arange(self.f_step-1,self.total, self.f_step)
        east_boundary=east_boundary[1:-1]
                
        self.boundary=np.concatenate([outer_boundary,inner_boundary , east_boundary, west_boundary]) #Does not include the corners!!
        
        corners=np.concatenate([corners_down, corners_up])
        self.corners=corners
        
        #inner matrix IDs
        inside=np.arange(self.total)
        inside=np.delete(inside, np.concatenate([np.ndarray.flatten(self.boundary), corners]))
        self.inside=inside
        
        self.corners=corners
        self.outer_boundary=outer_boundary
        self.inner_boundary=inner_boundary
        self.east_boundary=east_boundary
        self.west_boundary=west_boundary

# =============================================================================
#     def set_flux_BC_v_neglect_der(self, phi_vessel, K_eff):
#         """This function sets the boundary condition (G(r)*q'(s)) in the boundaries. 
#         Normall, the solution is coupled. But here, the values in phi_vessel are given."""
#         K_total=K_eff*np.pi*self.Rv**2*self.Green(self.r)
#         self.K_total=K_total
#         inc_s=self.inc_s
#         der=self.get_gradient(phi_vessel).dot(phi_vessel)
#         east=K_total*der[-1]/inc_s
#         west=-K_total*der[0]/inc_s
#         #Inner is zero
#         return(np.vstack([east, west]))
# =============================================================================
        
    def assembly(self):
        """Laplacian"""
        #To get the coefficients I can use the function for teh fluxes in cylindricals
        fluxes=[True, True, True, True, False, False]
        data=np.array([])
        row=np.array([])
        col=np.array([])
        r=self.r
        for i in self.inside:
            ri=r[i//self.f_step]
            coeffs=matrix_coeff(fluxes, ri, self.inc_r, self.inc_s, self.D, 0)
            [cntrl, up, down, east, west]=coeffs
            data=np.concatenate([data, coeffs])
            row=np.concatenate([row, np.array([i,i,i,i,i])])
            col=np.concatenate([col, np.array([i, i+self.f_step, i-self.f_step, i+1, i-1])])
            
        #Set upper BCs
        for i in self.outer_boundary:
            ri=r[-1]
            fluxes=[False, True, True, True, False, False]
            coeffs=matrix_coeff(fluxes, ri, self.inc_r, self.inc_s, self.D, 0)
            [cntrl, up, down, east, west]=coeffs
            data=np.concatenate([data, [cntrl, down , east, west]])
            row=np.concatenate([row, np.array([i,i,i,i])])
            col=np.concatenate([col, np.array([i,  i-self.f_step, i+1, i-1])])
            
        for i in self.inner_boundary:
            ri=r[0]
            fluxes=[True, False, True, True, False, False]
            coeffs=matrix_coeff(fluxes, ri, self.inc_r, self.inc_s, self.D, 0)

            [cntrl, up, down, east, west]=coeffs
            data=np.concatenate([data, [cntrl, up , east, west]])
            row=np.concatenate([row, np.array([i,i,i,i])])
            col=np.concatenate([col, np.array([i,  i+self.f_step, i+1, i-1])])
        
        for i in self.east_boundary:
            ri=self.r[i//self.f_step]
            fluxes=[True, True, False, True, False, False]
            coeffs=matrix_coeff(fluxes, ri, self.inc_r, self.inc_s, self.D, 0)
            [cntrl, up, down, east, west]=coeffs
            data=np.concatenate([data, [cntrl, up , down ,west]])
            row=np.concatenate([row, np.array([i,i,i,i])])
            col=np.concatenate([col, np.array([i, i+self.f_step, i-self.f_step,  i-1])])
            
        for i in self.west_boundary:
            fluxes=[True, True, True, False, False, False]
            ri=self.r[i//self.f_step]
            coeffs=matrix_coeff(fluxes, ri, self.inc_r, self.inc_s, self.D, 0)
            [cntrl, up, down, east, west]=coeffs
            data=np.concatenate([data, [cntrl, up , down ,east]])
            row=np.concatenate([row, np.array([i,i,i,i])])
            col=np.concatenate([col, np.array([i, i+self.f_step, i-self.f_step,  i+1])])
            
        #Corners
        i=self.corners[0]
        fluxes=[True, False, True, False, False, False]
        ri=self.r[0]
        coeffs=matrix_coeff(fluxes, ri, self.inc_r, self.inc_s, self.D, 0)
        [cntrl, up, down, east, west]=coeffs
        data=np.concatenate([data, [cntrl, up ,east]])
        row=np.concatenate([row, np.array([i,i,i])])
        col=np.concatenate([col, np.array([i, i+self.f_step,  i+1])])
        
        i=self.corners[1]
        fluxes=[True, False, False, True, False, False]
        ri=self.r[0]
        coeffs=matrix_coeff(fluxes, ri, self.inc_r, self.inc_s, self.D, 0)
        [cntrl, up, down, east, west]=coeffs
        data=np.concatenate([data, [cntrl, up ,west]])
        row=np.concatenate([row, np.array([i,i,i])])
        col=np.concatenate([col, np.array([i, i+self.f_step,  i-1])])
        
        i=self.corners[2]
        fluxes=[False, True, True, False, False, False]
        ri=self.r[-1]
        coeffs=matrix_coeff(fluxes, ri, self.inc_r, self.inc_s, self.D, 0)
        [cntrl, up, down, east, west]=coeffs
        data=np.concatenate([data, [cntrl, down ,east]])
        row=np.concatenate([row, np.array([i,i,i])])
        col=np.concatenate([col, np.array([i, i-self.f_step,  i+1])])
        
        i=self.corners[3]
        fluxes=[False, True, False, True, False, False]
        ri=self.r[-1]
        coeffs=matrix_coeff(fluxes, ri, self.inc_r, self.inc_s, self.D, 0)
        [cntrl, up, down, east, west]=coeffs
        data=np.concatenate([data, [cntrl, down ,west]])
        row=np.concatenate([row, np.array([i,i,i])])
        col=np.concatenate([col, np.array([i, i-self.f_step,  i-1])])
        
        self.data=data
        self.row=row
        self.col=col
        
        A=sp.sparse.csc_matrix((data, (row, col)), shape=(self.total, self.total))
        return(A)
    
    
def matrix_coeff(fluxes, ri, inc_r, inc_s, D, Veli):
    """Returns the stensil for the fluxes in a cylindrical reference system. Two coordinates, 
    longitudinal and radial. Depends on the radial position, the coefficients in the radial 
    direction (north, south), will differ. """
    Diff_flux_north, Diff_flux_south, Diff_flux_east, Diff_flux_west, Conv_flux_east, Conv_flux_west=fluxes
    cntrl, N,S,E,W=float(0),float(0),float(0),float(0),float(0)
    if Diff_flux_north:

        cntrl-=(ri+inc_r/2)*D/(ri*inc_r**2)
        N+=(ri+inc_r/2)*D/(ri*inc_r**2)
        
    if Diff_flux_south:
        cntrl-=(ri-inc_r/2)*D/(ri*inc_r**2)
        S+=(ri-inc_r/2)*D/(ri*inc_r**2)
        
    if Diff_flux_east:
        cntrl-=D/inc_s**2
        E+=D/inc_s**2
        
    if Diff_flux_west:
        cntrl-=D/inc_s**2
        W+=D/inc_s**2 
        
    if Conv_flux_east:
        cntrl-=Veli/(inc_s)
        
    if Conv_flux_west:
        W+=Veli/(inc_s)
    co=np.array([cntrl, N,S,E,W])
    return(co)



def get_reference_q(Lap_operator,q_vessel, Rv, inc_r, inc_s, ext_boundary):
    """Provides the numerical solution of the problem with a given inlet flux (q_vessel)
    
    modifies the lap_operator and the RHS in order to accomodate the incomming flux
    
    THE OUTER DIRICHLET ARE SET HERE AS WELL"""
    RHS=np.zeros(Lap_operator.shape[0])
    factor=1/(np.pi*(2*Rv+inc_r)*inc_r)
    print(factor)
    i=np.arange(len(q_vessel))
    RHS[i]=-factor*q_vessel
    
    RHS[ext_boundary]=0
    Lap_operator[ext_boundary,:]=0
    Lap_operator[ext_boundary, ext_boundary]=1    
    sol=np.linalg.solve(Lap_operator, RHS)
    lenr=len(RHS)/len(q_vessel)
    return(sol.reshape(int(lenr), len(q_vessel)))

# =============================================================================
# 
# import cProfile
# import pstats
# cProfile.run('process.run()','RHS_for_v',1)
# stats = pstats.Stats('test_file.stats')
# stats.strip_dirs()
# stats.sort_stats('cumulative')
# stats.print_stats()
# =============================================================================
        



# =============================================================================
# PLOTTING FUNCTIONS, NO CALCULATIONS DONE HERE
# =============================================================================

def plot_profile(i, s, r, sol_split, sol_real, title):
    """i*10 represents the percentage along the line"""
    pos=-1 if i==-1 else int(len(s)*i/10)

    plt.plot(r, sol_split[:,pos], label='r/L split_sol=%f ' % (pos/len(s)), marker='*')
    #plt.plot(k.r, analyt)
    plt.plot(r, sol_real[:,pos],label='r/L real_sol=%f'% (pos/len(s)))
    plt.xlabel("r")
    # =============================================================================
    # plt.plot(k.r, real_v[:,10])
    # plt.plot(k.r, sol[:,10])
    # =============================================================================
    plt.title(title)
    plt.legend()
    
def s_plot_profile(i, s, r, sol_split, sol_real, title):
    """i*10 represents the percentage along the line"""
    pos=int(len(r)*i/10)

    plt.plot(s, sol_split[pos,:], label='s/L split_sol=%d ' % (pos/len(r)), marker='*')
    #plt.plot(k.r, analyt)
    plt.plot(s, sol_real[pos,:],label='s/L real_sol=%d '% (pos/len(r)))
    plt.xlabel("s")
    plt.title(title)
    # =============================================================================
    # plt.plot(k.r, real_v[:,10])
    # plt.plot(k.r, sol[:,10])
    # =============================================================================
    plt.legend()

def plott(sol, title, cmin, cmax):
    levels=np.linspace(cmin, cmax,10)
    plt.contourf(sol, levels)
    plt.ylabel("r")
    plt.xlabel("s")
    plt.title(title)
    plt.colorbar()
    
def compare_full_solutions(sol1, sol2, title_sol1, title_sol2,lev, title):
    font=24
    levels=np.arange(lev[0], lev[1], (lev[1]-lev[0])/10)
    fig, ax=plt.subplots(1,2, figsize=(10,7))
    (ax1,ax2)=ax
    fig.suptitle(title, fontsize=font)
    im=ax1.contourf(sol1, levels)
    ax1.set_title(title_sol1, fontsize=font)
    ax1.set_ylabel('r', fontsize=font)
    ax1.set_xlabel('s', fontsize=font)
    im=ax2.contourf(sol2, levels)
    ax2.set_title(title_sol2, fontsize=font)
    ax2.set_xlabel('s', fontsize=font)
    


    plt.tight_layout()
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar=plt.colorbar(im, ax=ax)
                        
    cbar.ax.set_ylabel('Concentration', fontsize=font)
    





# =============================================================================
# FUNCTIONS TO PROVIDE INTEGRALS COMMONLY VIA BRUTE FORCE
# =============================================================================

def oneD_integral(x, function, xlim):
    h=(np.max(x)-np.min(x))/(len(x)-1)
    indices_x=(x>=xlim[0]) & (x<xlim[1])
    integral=np.sum(function[indices_x])*h
    return(integral)

def threeD_integral(slim, rlim, s,r, function):
    """Integrates the function in cylindrical coordinates"""
    #It will multiply by 2 pi r to do the axyilsymmetric integral
    f=sweep(function,r)
    
    s_min, s_max=slim
    r_min, r_max=rlim
    
    hr=(np.max(r)-np.min(r))/(len(r)-1)
    hs=(np.max(s)-np.min(s))/(len(s)-1)
    
    indices_s=(s>s_min) & (s<s_max)
    indices_r=(r>r_min) & (r<r_max)
    
    vol_enclosed=2*np.pi*hr*np.sum(r[indices_r])*(s_max-s_min)
    
    integral=np.sum(f[indices_r, :][:,indices_s])*hs*hr*2*np.pi
    return([integral, vol_enclosed])
    
def sweep(f,r):
    function=np.zeros(f.shape)
    for i in range(len(r)):
        function[i,:]=f[i,:]*r[i]
    return(function)

def analyt_int(rk, hr, hs, Rv):
    """Function to test the east_west BC"""
    r=np.array([rk-hr/2, rk+hr/2])
    factor=(r**2*(2*np.log(r/Rv)-1))
    integral=factor[1]-factor[0]
    integral=integral/(8*np.pi*hr*hs*rk)
    return(integral)

def analyt_int_2D(L, D, rk,sk,  hr, hs, Rv):
    r=np.array([rk-hr/2, rk+hr/2])
    f_r=(r**2*(2*np.log(r/Rv)-1))
    int_r=f_r[1]-f_r[0]
    
    s=np.array([sk-hs/2, sk+hs/2])
    f_s=0.7*L*np.sin(s/(0.7*L))
    int_s=f_s[1]-f_s[0]
    
    integral=int_r*int_s/(3.92*np.pi*L**2*D*rk*hr*hs)
    return(integral)

def test(L, D, rk,sk,  hr, hs, Rv):
    r=np.array([rk-hr/2, rk+hr/2])
    f_r=(r**2*(2*np.log(r/Rv)-1))
    int_r=f_r[1]-f_r[0]
    
    s=np.array([sk-hs/2, sk+hs/2])
    f_s=0.7*L*np.sin(s/(0.7*L))
    int_s=f_s[1]-f_s[0]
    
    integral=int_r*int_s/(1.96*L**2*D)
    return(integral)



    

class sol_split_q_imposed():
    """This class does nothing for the coupled problem. It is part of the intermediate step where I wanted to test the solution 
    splitting technique with an imposed in flux from the vessel instead of a coupled system"""
    def __init__(self,lap_obj_coarse, lap_obj_fine, q_vessel, q_sec, q_first, lap_operator_coarse):
        self.coarse_r=lap_obj_coarse.r
        self.coarse_s=lap_obj_coarse.s
        self.fine_r=lap_obj_fine.r
        self.fine_s=lap_obj_fine.s
        
        self.fine_hr=lap_obj_fine.inc_r
        self.fine_hs=lap_obj_fine.inc_s
        self.coarse_hr=lap_obj_coarse.inc_r
        self.coarse_hs=lap_obj_coarse.inc_s
        
        self.Rv=lap_obj_fine.Rv
        self.L=lap_obj_coarse.L
        self.Rmax=lap_obj_coarse.r[-1]+self.coarse_hr/2
        
        #the RHS of v is given by:
        self.F=np.outer(Green(self.fine_r, self.Rv), q_sec)
        
        #array of volume of the cells depending on the radius 
        self.vol_fine=np.pi*2*self.fine_hr*self.fine_r*self.fine_hs
        self.vol_coarse=np.pi*2*self.coarse_hr*self.coarse_r*self.coarse_hs
        
        #FULL BOUNDARIES
        self.coarse_east=np.concatenate([[lap_obj_coarse.corners[1]], lap_obj_coarse.east_boundary, [lap_obj_coarse.corners[3]]])
        self.coarse_west=np.concatenate([[lap_obj_coarse.corners[0]], lap_obj_coarse.west_boundary, [lap_obj_coarse.corners[2]]])
        self.coarse_out=np.concatenate([[lap_obj_coarse.corners[2]], lap_obj_coarse.outer_boundary, [lap_obj_coarse.corners[3]]])
        
        self.q_first=q_first
        self.q_sec=q_sec
        self.q_vessel=q_vessel
        
        self.G=Green(self.fine_r, self.Rv)
        
        self.D=lap_obj_fine.D

        self.coarse_lap=lap_operator_coarse
        
        self.sol_sing=np.outer(Green(self.fine_r, self.Rv),q_vessel)
        
        self.north_function=False
        

    def get_RHS_v(self):
        """calculates the RHS term for the creation of v (v being the regular term)
        It is calculated 2 ways, with the analytical integral -> anal and with the numerical integral -> threeD_integral"""
        coarse_r=self.coarse_r
        coarse_s=self.coarse_s
        integral_array=np.zeros([len(self.coarse_r),len(self.coarse_s)])
        anal=np.zeros([len(self.coarse_r),len(self.coarse_s)])
        for i in range(len(coarse_r)):
            for j in range(len(coarse_s)):
                
                slim=coarse_s[j]-self.coarse_hs/2, coarse_s[j]+self.coarse_hs/2
                rlim=coarse_r[i]-self.coarse_hr/2, coarse_r[i]+self.coarse_hr/2
                
                a=threeD_integral(slim, rlim, self.fine_s,self.fine_r, np.outer(self.G,self.q_sec))
                integral_array[i,j]=-a[0]/a[1]   #The operator is always per volume. It will be better to divide by the 
                #volume that was enclosed by the integral function, since it will likely differ from the real volume of 
                #the coarse cell
                
                anal[i,j]=-analyt_int_2D(self.L, self.D, self.coarse_r[i],self.coarse_s[j],  self.coarse_hr, self.coarse_hs, self.Rv)
        self.anal=anal
        self.RHS_noBC=integral_array     
        return(integral_array)
    
    
    def get_east_west_RHS(self):
        c_r=self.coarse_r
        c_hr=self.coarse_hr
        c_s=self.coarse_s
        c_hs=self.coarse_hs
        
        f_hr=self.fine_hr
        f_r=self.fine_r
        RHS_east_west=np.zeros(len(self.coarse_s)*len(self.coarse_r))
        
        #North
        self.factor_north=2*(c_r[-1]+c_hr/4)/(c_r[-1]*c_hr**2)
        
        for i in self.coarse_east:
            pos_r=i//len(self.coarse_s)
            rlim=[c_r[pos_r]-c_hr/2, c_r[pos_r]+c_hr/2]
            slim=[self.L-self.coarse_hs, self.L]
            RHS_east_west[i]=self.q_first[-1]*oneD_integral(f_r, self.G*f_r, rlim)/(c_hs*c_hr*c_r[pos_r])
        
        for i in self.coarse_west:
            pos_r=i//len(self.coarse_s)
            rlim=[c_r[pos_r]-c_hr/2, c_r[pos_r]+c_hr/2]
            slim=[self.L-self.coarse_hs, self.L]
            RHS_east_west[i]=-self.q_first[0]*oneD_integral(f_r, self.G*f_r, rlim)/(c_hs*c_hr*c_r[pos_r])
        
        return(RHS_east_west)
    
    def set_Dirichlet_north(self):
        if self.north_function!=True:
            factor=2*(self.coarse_r[-1]+self.coarse_hr/4)/(self.coarse_r[-1]*self.coarse_hr**2)
            self.factor=factor
            self.RHS_north=np.zeros(len(self.coarse_r)*len(self.coarse_s))
            
            for i in self.coarse_out:
                coord_s=self.coarse_s[i%len(self.coarse_r)]
                print("the s coord is;", coord_s)
                fines=np.argmin((self.fine_s-coord_s)**2)
                singular_boundary_value=self.sol_sing[-1,fines]
                print("s position is :", fines)
                
                self.RHS_north[i]=-factor*singular_boundary_value
                self.coarse_lap[i,i]-=factor
            return(self.RHS_north)
        else:
            print("you have already ran this function, be careful. Use the variable object.RHS_north")
            return(self.RHS_north)
    
    def seg_north_with_ghost(self):
        """Sets the outer Dirichlet boundary conditions with a new creation of a problem with bigger radius. 
        This implies the creation of a new problem with simplify_assembly_laplacia, and the obtention of a new laplacian operator"""
        #Let's create the Laplacian operator with the proper boundary conditions (Dirichlet north and 
        #no flux east and west)
        R_max=self.coarse_r[-1]+1.5*self.coarse_hr
        coarse_r_points=len(self.coarse_r)+1
        coarse_s_points=len(self.coarse_s)
        ghost_coarse=simplified_assembly_Laplacian_cylindrical(R_max, self.Rv,coarse_r_points,coarse_s_points, self.L, self.D)
        ghost_coarse_Lap=ghost_coarse.assembly().toarray()
        pos_h=len(self.q_vessel)//len(self.coarse_r)
        sampling=np.around(np.linspace(pos_h//2,len(self.q_vessel)-pos_h//2, len(self.coarse_r))).astype(int)
        ghost_sing_values=Green(ghost_coarse.r[-1], self.Rv)*self.q_vessel[sampling]
        
        self.ghost_coarse=ghost_coarse
        
        pos_ghost=np.arange(-len(self.coarse_r),0)
        ghost_coarse_Lap[pos_ghost,:]=0
        ghost_coarse_Lap[pos_ghost,pos_ghost]=1
        self.ghost_coarse_Lap=ghost_coarse_Lap
        self.RHS_ghost=np.zeros((len(self.coarse_r)+1)*len(self.coarse_s))
        
        self.pos_ghost=pos_ghost
        self.ghost_sing_values=ghost_sing_values
        self.RHS_ghost_virgin=np.copy(self.RHS_ghost)
        self.RHS_ghost[pos_ghost]=-ghost_sing_values
        return()
        

def get_averaged_solution(c_r,c_s, f_r, f_s, fine_solution, h_r, h_s):
    avg=np.zeros([len(c_r), len(c_s)])
    c=0
    for i in c_r:
       d=0
       pos_r=np.where((f_r-i)**2<h_r**2)[0]
       for j in c_s:
           pos_s=np.where((f_s-j)**2<h_s**2)[0]
           avg[c,d]=np.sum(fine_solution[pos_r,:][:,pos_s])/(len(pos_s)*len(pos_r))
           d+=1
       c+=1
    return(avg)



        