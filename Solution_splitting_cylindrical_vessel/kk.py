# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:14:42 2020
@author: david
"""


import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg
import pandas as pd
import time
from numba import jit, njit

from numba import int32, float32    # import the types




plt.rcParams.update({'font.size': 17})
full_graph=0
mesh_convergence=0



def matrix_coeff(fluxes, ri, inc_r, inc_z, D, Veli):
        Diff_flux_north, Diff_flux_south, Diff_flux_east, Diff_flux_west, Conv_flux_east, Conv_flux_west=fluxes
        cntrl, N,S,E,W=float(0),float(0),float(0),float(0),float(0)
        if Diff_flux_north:

            cntrl-=(ri+inc_r/2)*D/(ri*inc_r**2)
            N+=(ri+inc_r/2)*D/(ri*inc_r**2)
            
        if Diff_flux_south:
            cntrl-=(ri-inc_r/2)*D/(ri*inc_r**2)
            S+=(ri-inc_r/2)*D/(ri*inc_r**2)
            
        if Diff_flux_east:
            cntrl-=D/inc_z**2
            E+=D/inc_z**2
            
        if Diff_flux_west:
            cntrl-=D/inc_z**2
            W+=D/inc_z**2 
            
        if Conv_flux_east:
            cntrl-=Veli/(inc_z)
            
        if Conv_flux_west:
            W+=Veli/(inc_z)
        co=np.array([cntrl, N,S,E,W])
        return(co)
    
matrix_coeff_jit=jit()(matrix_coeff)


def app_main(E,W,N,S,cntrl, main,c, zlen):
    """This function takes the information of the values of the fluxes and returns 
    a matrix with the positions and data to be inserted into the sparse matrix"""
    data=main[0,:]
    row=main[1,:]
    col=main[2,:]
    if E:
        row=np.append(row, c) 
        col=np.append(col, c+1)
        data=np.append(data, E)

    if W:
        row=np.append(row, c) 
        col=np.append(col, c-1)
        data=np.append(data, W)
    if N:
        row=np.append(row, c) 
        col=np.append(col, c+zlen)
        data=np.append(data, N)
    if S:
        row=np.append(row, c) 
        col=np.append(col, c-zlen)
        data=np.append(data, S)
    row=np.append(row, c) 
    col=np.append(col, c)
    data=np.append(data, cntrl)
    return(np.vstack((data,row, col)))

app_main_jit=jit()(app_main)



def efficient_assembly_jit(zlen, D, v_r, t_r, Vel, phi, obj_type, inc_r, inc_z, coeffs, K_m,Rv):

    if obj_type==2:
        r=t_r
    elif obj_type==1:
        r=v_r

    v_rlen=v_r.size
    t_rlen=t_r.size

    row_main=np.array([0])
    col_main=np.array([0])
    data_main=np.array([0.0])
    
    
    
    row_c=np.array([0])
    col_c=np.array([0])
    data_c=np.array([0.0])
    
    cou=np.vstack((data_c,row_c, col_c))
    
    c=0
    
    (CN,CS,CE,CW)=coeffs

    for i in phi:
        #phi is the flatten matrix C of the object (object -> Tissue or vessel). The matrix C has the 
        #information relating the boundaries
        main=np.vstack((data_main,row_main, col_main))
        
        #obj.phi is the one dimensional array with the information encoded for the fluxes
        
        Diff_flux_north,Diff_flux_south,Diff_flux_east,Diff_flux_west, \
        Conv_flux_east, Conv_flux_west=bool(1),bool(1),bool(1),bool(1),bool(1),bool(1)
        #axial coordinate (int)    
        iz=int(c%zlen)
        #radial coordinate (int)
        ir=int(c//zlen)
        lenves=v_rlen*zlen
        lentis=t_rlen*zlen

        
        if obj_type==2:
            irr=v_rlen+ir #this is the actual value on the radius column 
        
        N,S,E,W,cntrl=float(0),float(0),float(0),float(0),float(0)
        coupl_south, coupl_north=float(0),float(0)

        
        if (i%1000)//100==CE:
            Diff_flux_east=False
        if (i%100)//10==CS:
            Diff_flux_south=False
            if obj_type==2:
                #this is the flux that goes through the membrane to the tissue

                coupl_south+=K_m[iz]*Rv/(t_r[ir]*inc_r)
                cntrl-=K_m[iz]*Rv/(t_r[ir]*inc_r)
    
        if (i%10)==CN:
            Diff_flux_north=False
            if obj_type==1:
                coupl_north+=K_m[iz]*Rv/(v_r[ir]*inc_r)
                cntrl-=K_m[iz]*Rv/(v_r[ir]*inc_r)
        if i//1000==CW and obj_type==1:
            Diff_flux_west=False
            Conv_flux_west=False
            Diff_flux_north=False
            Diff_flux_south=False
            Diff_flux_east=False
            Conv_flux_east=False
            
    
        if i//1000==CW and obj_type==2:
            Diff_flux_west=False

        
        fluxes=np.array([Diff_flux_north,Diff_flux_south,Diff_flux_east,Diff_flux_west, \
        Conv_flux_east, Conv_flux_west])
        co=matrix_coeff_jit(fluxes, r[ir], inc_r, inc_z, D, float(Vel[ir]))
        cntrl+=co[0]
        N+=co[1]
        S+=co[2]
        E+=co[3]
        W+=co[4]

        data_m=app_main_jit(E,W,N,S,cntrl, main,c, zlen)
        data_main=data_m[0,:]
        row_main=data_m[1,:]
        col_main=data_m[2,:]

        #Coupling        
        if coupl_south:
            #the object must be the tissue
            row_c=np.append(row_c, c) 
            col_c=np.append(col_c, zlen*(v_rlen-1)+iz) 
            data_c=np.append(data_c, coupl_south)
        if coupl_north:
            #the object must be the vessel
            row_c=np.append(row_c, c) 
            col_c=np.append(col_c, iz) 
            data_c=np.append(data_c, coupl_north)
        
        c+=1
        
        data=np.hstack((np.vstack((data_main, row_main, col_main)), np.vstack((data_c, row_c, col_c))))
        data[0,0]=float(data_main.size)
    return(data)


class assembly_sparse():
    def __init__(self,obj, geom, coeffs):
        #   ASSEMBLY VESSEL MATRIX
        #def assembly_sparse(obj, geom):
        self.K_m=obj.K_m
        self.inc_r, self.inc_z=obj.inc_r, obj.inc_z
        self.Rv=geom["Rv"]
        self.C=obj.C
        self.D=obj.D
        self.vessel_r=geom["vessel_r"]
        self.tissue_r=geom["tissue_r"]
        self.z=geom["z"]
        self.zlen=len(self.z)
        print("assembly function called")
        print("permeability:", np.max(self.K_m))
        self.phi=np.ndarray.flatten(obj.C)
        phi=self.phi
        self.lenves=len(self.vessel_r)*len(self.z)
        lenves=self.lenves
        self.lentis=len(self.tissue_r)*len(self.z)
        lentis=self.lentis
        if obj.typ=="tissue":    
            self.obj_type=2
        if obj.typ=="vessel":
            self.obj_type=1
        self.vel=obj.vel

        data=efficient_assembly_jit(float(self.zlen), float(self.D), self.vessel_r.astype(float),  
                                self.tissue_r.astype(float), self.vel.astype(float), phi, self.obj_type, 
                                float(self.inc_r), float(self.inc_z), coeffs, self.K_m, self.Rv)
        
        self.data=data
        lenmain=int(data[0,0])
        self.data_main=data[0,:lenmain]
        self.data_c=data[0,lenmain:]
        self.row_c=data[1,lenmain:]
        self.col_c=data[2,lenmain:]
        self.row_main=data[1,:lenmain]
        self.col_main=data[2,:lenmain]
        
        data_c=np.array(self.data_c, dtype='f')
        data_main=np.array(self.data_main, dtype='f')
        col_c=np.array(self.col_c, dtype='int')
        row_c=np.array(self.row_c, dtype='int')
        row_main=np.array(self.row_main, dtype='int')
        col_main=np.array(self.col_main, dtype='int')

        
            
        if obj.typ=="tissue":
            
            coupl=sp.sparse.csc_matrix((data_c, (row_c, col_c)), shape=(len(phi),lenves), dtype='f')
            main=sp.sparse.csc_matrix((data_main, (row_main, col_main)), shape=(len(phi), len(phi)), dtype='f')
            Down=sp.sparse.hstack([coupl, main])
            self.Down=Down

            
        elif obj.typ=="vessel":
            coupl=sp.sparse.csc_matrix((data_c, (row_c, col_c)), shape=(len(phi),lentis), dtype='f')
            main=sp.sparse.csc_matrix((data_main, (row_main, col_main)), shape=(len(phi), len(phi)), dtype='f')
            Up=sp.sparse.hstack([main,coupl])
            self.Up=Up
        


            
        
    
        
    

def plot_solution(solution, vessel, tissue, parameters, geom, t, Rv,K_m, *namefig, **l):
    
    zlen=geom["zlen"]
    rlen=geom["rlen"]
    Z=geom["Z"]
    z=Z[0,:]
    Rho=geom["Rho"]
    L_c=geom["L_c"]
    
    sol=np.reshape(solution,(rlen,zlen))
    C=sol
    C=np.append(sol[::-1,:],sol,axis=0)
    Z=np.append(Z[::-1,:],Z,axis=0)
    Rho=np.append(-Rho[::-1,:],Rho,axis=0)
    
    #To fix the amount of contour levels COUNTOUR LEVELS
    lim=l["limF"] if l else np.max(sol)/2
    beg=l["lim0"] if l else 0
    breaks=np.linspace(beg,lim,10)
    
    print("minimum: ", np.min(sol))
    print("max: ", np.max(sol))
    
    #array positions where the permeability is not actually 0
    coupl_s=np.where(K_m!=0)
    
    fig=plt.figure(figsize=(10,15))
    gs=gridspec.GridSpec(3,2)
    ax=fig.add_subplot(gs[0,:])
    fig.suptitle("Contour and C and C_avg through the centerline", fontsize=14, fontweight="bold")
    eDam=parameters["eDam"]
    Dat=parameters["Dat"]
    epsilon=parameters["epsilon"]
    ax.set_title("$\epsilon Da_m=$ {i},$Da_t=$ {j}, $\epsilon$= {ep}, t={o} s".format(i=eDam, j=Dat, ep=epsilon, o=t))    
    CS=ax.contourf(Z,Rho,C,breaks, levels=breaks)
    ax.plot([0,z[-1]],[Rv, Rv], 'r-')
    ax.plot([0,z[-1]],[-Rv, -Rv], 'r-')
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(CS, ticks=breaks, orientation="vertical", format='%.0e')
    cbar.ax.set_ylabel('concentration')
    ax.set_xlabel("z")
    ax.set_ylabel("r")
    ax.grid()
    

    kk=data_proc(solution, vessel, tissue)
    csa=kk.cross_section_average()[coupl_s]
    ax2=fig.add_subplot(gs[1,0])
    #ax2.plot(vessel.z[coupl_s], csa/np.max(csa))
    ax2.plot(vessel.z[coupl_s], csa, label="Vessel average concentration")
    ax2.set_ylabel("conc")
    ax2.set_xlabel("z")
    ax2.set_ylim((0, 1.1))
    leg = ax2.legend(prop={'size': 12});
    ax3=fig.add_subplot(gs[1,1])
    #ax3.plot(vessel.z[coupl_s], kk.get_tissue_wall()[coupl_s]/np.max(csa))
    ax3.plot(vessel.z[coupl_s], kk.get_tissue_wall()[coupl_s], label="Tissue wall concentration")
    ax3.set_xlabel("z")
    ax3.set_ylim((0, 1.1))
    leg = ax3.legend(prop={'size': 12});
    ax4=fig.add_subplot(gs[2,:])
    p=np.where(K_m!=0)
    pos=np.linspace(np.min(np.where(K_m!=0)), np.max(np.where(K_m!=0)),5)

    r=geom["Rho"][:,1]
    pos_tissue=[np.where(r>Rv)][0][0]
    for j in pos:
        i=round(j)
        ax4.plot(sol[pos_tissue,i], r[pos_tissue], label='z={o}'.format(o=round((i-np.min(np.where(K_m!=0)))/len(np.where(K_m!=0)[0]),2)))
    
    leg = ax4.legend(prop={'size': 12});
    ax4.set_xlabel("concentration")
    ax4.set_ylabel("R")
    
    figname=namefig[0] if namefig else "solution_{time}s.pdf".format(time=t)
    print(figname)
    plt.savefig(figname)

      
    
    
# =============================================================================
#     vess=np.where(np.abs(Rho[:,0])<Rv)[0]
#     breaks2=np.linspace(0,1,10)
#     fig2=plt.figure()
#     gs2=gridspec.GridSpec(2,2)
#     ax4=fig2.add_subplot(gs2[0,:])
#     ax4.plot(vessel.z, kk.cross_section_average())
#     ax4.set_ylabel("conc")
#     ax4.set_xlabel("s") 
#     ax5=fig2.add_subplot(gs2[1,:])
#     CS2=ax5.contourf(Z[vess,:],Rho[vess, :],C[vess, :])
#     cbar = fig2.colorbar(CS2, ticks=breaks2, orientation="vertical")
#     ax5.set_ylabel("R")
#     ax5.set_xlabel("s") 
#     plt.show()
# =============================================================================



    
#==============================================================================
#     plt.figure
#     plt.imshow(C)
#     plt.colorbar
#==============================================================================
    
    
    
    return()
    
    
class domain():
    def __init__(self,coeffs, parameters, typ):
        self.typ=typ
        Rv=parameters["R_vessel"]
        L=parameters["length"]
        CN,CS,CE,CW=coeffs
        Rm=parameters["R_max"]
        self.inc_r, self.inc_z=parameters["inc_rho"], parameters["inc_z"]
        z=np.arange(0,L,self.inc_z) 
        rv=np.arange(self.inc_r/2,Rv,self.inc_r) 
        rt=np.arange(np.max(rv)+self.inc_r,Rm,self.inc_r) 
        #total radius
        r=np.append(rv,rt)
        
        if self.typ=="vessel":
            self.Z,self.Rho=np.meshgrid(z,rv)
            self.r=rv
        if self.typ=="tissue":
            self.Z,self.Rho=np.meshgrid(z,rt)
            self.r=rt
        rlen, zlen=self.Z.shape
        C=np.zeros(self.Z.shape, dtype='f')
        self.K_m=parameters["Permeability"]
        C[0,:]+=10*CS
        C[-1,:]+=CN
        C[:,0]+=1000*CW
        C[:,-1]+=100*CE
        phi=np.ndarray.flatten(C)
        
        self.C=C.astype(np.int)
        self.z=z

        self.rlen=rlen
        self.zlen=zlen
        
        phi=np.ndarray.flatten(self.C)
        self.phi=phi
        
        K=parameters["coeff_vel"]

        self.Rv=Rv
        self.vel=K*(Rv**2-self.r**2)
        self.vel[np.where(self.vel<0)]=0
        

        
        
        if self.typ=="vessel":
            self.D=parameters["Diff_vessel"]
        elif self.typ=="tissue":
            self.D=parameters["Diff_tissue"]
        else:
            print("you introduced the type wrong but because I dont know yet how to do exceptions you have to re run the code yourself")
        
        
class solver():
    
    def __init__(self, A, vessel, tissue, parameters, geom):

        self.A=A
        self.zlen=vessel.zlen
        self.v_rlen=vessel.rlen
        self.rt=tissue.r
        self.lent=len(tissue.phi)
        
        self.inlet=parameters["Inlet"]
        self.Mt=parameters["Tissue_consumption"]
        self.phit=tissue.phi
        self.phiv=vessel.phi
        self.phi_total=np.zeros(len(tissue.phi)+len(vessel.phi))
        self.lenprob=self.phi_total.size
        self.lim0=0
        self.limF=0.2
        self.Rv=parameters["R_vessel"] 
        self.geom=geom
        self.K_m=vessel.K_m
        self.it=0
        self.parameters=parameters 
        self.vessel=vessel
        self.tissue=tissue
    
    def set_Neuman_tissue(self,fluxes, r, inc_r, inc_z, D, c, zlen, value):
        ri=r[c//zlen]
        """Fluxes will be in the order North, South, East and West. With a 1 if there is and a 0 if there is not"""
        Diff_flux_north, Diff_flux_south, Diff_flux_east, Diff_flux_west=fluxes
        fluxes=Diff_flux_north, Diff_flux_south, Diff_flux_east, Diff_flux_west, 0, 0
        cntrl, N,S,E,W=float(0),float(0),float(0),float(0),float(0)
        co=matrix_coeff(fluxes, ri, inc_r, inc_z, D, np.zeros(len(self.rt)))
        [cntrl, N,S,E,W]=co
        d=app_main(E,W,N,S,cntrl, np.zeros([3,2]) ,c, zlen)
        data=d[0,2:]
        data_row=d[1,2:]
        data_col=d[2,2:]
        j=0
        for i in data:
            self.A[data_row[j], data_col[j]]=data[j]
            j+=1
        self.phi_total[c]=value
        
    def set_Neuman_east(self, value):
        r=self.rt
        for i in self.get_east()[1:-1]:
            fluxes=[1,1,1,0]
            self.set_Neuman_tissue(fluxes, r, self.tissue.inc_r, self.tissue.inc_z, self.tissue.D, i, self.tissue.zlen, value)
            
    def set_Neuman_west(self, value):
        r=self.rt
        for i in self.get_west()[1:-1]:
            fluxes=[1,1,0,1]
            self.set_Neuman_tissue(fluxes, r, self.tissue.inc_r, self.tissue.inc_z, self.tissue.D, i, self.tissue.zlen, value)
    
    def set_Neuman_north(self, value):
        r=self.rt
        for i in self.get_north():
            fluxes=[1,1,0,1]
            self.set_Neuman_tissue(fluxes, r, self.tissue.inc_r, self.tissue.inc_z, self.tissue.D, i, self.tissue.zlen, value)
            
    def get_north(self):
        return(np.arange(self.lenprob-self.zlen,self.lenprob))
    def get_east(self):
        return(np.arange(self.zlen*(self.v_rlen+1)-1,self.lenprob,self.zlen))
    def get_west(self):
        return(np.arange(self.zlen*self.v_rlen, self.lenprob,self.zlen))
    def get_south(self):
        return(np.arange(self.v_rlen*self.zlen,(self.v_rlen+1)*self.zlen))
    
    def set_Dirichlet_north(self, value):
        #Vector of positions of the boundary of the tissue 
        b=self.get_north()
        self.A[b,:]=0
        self.A[b,b]=1
        self.phi_total[b]=np.zeros(len(b))+value
    
    def set_Dirichlet_east(self,value):
        #Vector of positions of the boundary of the tissue 
        b=self.get_east()
        self.A[b,:]=0
        self.A[b,b]=1
        self.phi_total[b]=np.zeros(len(b))+value
        
    def set_Dirichlet_west(self,value):
        #Vector of positions of the boundary of the tissue 
        b=self.get_west()
        self.A[b,:]=0
        self.A[b,b]=1
        self.phi_total[b]=np.zeros(len(b))+value
    
    def set_Dirichlet_south(self,value):
        #Vector of positions of the boundary of the tissue 
        b=self.get_south()
        self.A[b,:]=0
        self.A[b,b]=1
        self.phi_total[b]=np.zeros(len(b))+value
        
    def set_Dirichlet_vessel(self, value):
        #Vector of positions of the entry in the vessel
        a=np.arange(0, self.zlen*self.v_rlen,self.zlen)
        #Set the coefficients of the matrix to zero except the fixed value positions
        self.A[a,:]=0
        self.A[a, a]=1
        #Set of the fixed values to zero or inlet values 
        phiv=np.zeros(len(self.phiv))
        phiv[a]=value
        self.phiv=phiv
        self.phi_total[:self.phiv.size]=self.phiv
        
    def tissue_consumption(self, Mt):
        #LINEAR CONSUMTION
        print("tissue consumption:", Mt)
        c=np.arange(self.v_rlen*self.zlen, self.lenprob)
        self.A[c,c]-=Mt
    
    def set_tissue_consumption_quadratic(self, Mt):
        M=Mt*self.tissue.r**-2
        print("tissue consumption exponential:", Mt)
        c=np.arange(self.v_rlen*self.zlen, self.lenprob)
        for i in c:
            ri=i//self.zlen-self.v_rlen
            self.A[i,i]-=M[ri]
    
        
    def set_DirichletSS_sparse(self):
        """Function sets zero Dirichlet in the borders of the tissue and the 
        entry of the vessel"""
        
        
        self.set_Dirichlet_vessel(self.inlet)


        self.tissue_consumption(self.Mt)
        
        #REINITIALISATION OF THE VECTOR OF TISSUE PHI!!!
        self.phi_t=np.zeros(len(self.phit))
        
        self.set_Dirichlet_north(0)
        self.set_Dirichlet_east(0)
        self.set_Dirichlet_west(0)
        
        self.A.eliminate_zeros()


    def solve_sparse_SS(self,geom, *string):
        self.sol_SS=sp.sparse.linalg.spsolve(self.A, self.phi_total)

        plot_solution(self.sol_SS, self.vessel, self.tissue, self.parameters, geom, 9999, self.Rv,self.K_m, string[0], lim0=self.lim0, limF=self.limF)
        return(self.sol_SS)
    
    def call_plot_solution(self, phi, time, *string):
        plot_solution(phi, self.vessel, self.tissue, self.parameters, self.geom, time,  self.Rv, self.K_m,string[0],  lim0=self.lim0, limF=self.limF)
        


    
class data_proc():
    def __init__(self,phi, vessel, tissue):
        """phi should be the solution, that now we are gonna process"""
        self.v=phi[:vessel.rlen*vessel.zlen]
        self.t=phi[vessel.rlen*vessel.zlen:] 
        self.v_r=vessel.r
        self.z=vessel.z
        self.t_r=tissue.r
        
    def cross_section_average(self):
        self.v=self.v.reshape(len(self.v_r), len(self.z))
        average=np.zeros(len(self.z))
        for i in range(len(self.z)):
            average[i]=np.sum(self.v[:,i])
        
        self.average=average/len(self.v_r)
        return(self.average)
    
    def get_tissue_wall(self):
        self.C_t=self.t.reshape(len(self.t_r), len(self.z))
        return(self.C_t[0,:])   


def plot_tissue(a, tissue, vessel):
    
    b=a[vessel.rlen*vessel.zlen:]
    tis=b.reshape(tissue.rlen, tissue.zlen)
    
    f=plt.figure()
    plt.contourf(tissue.Z, tissue.Rho, tis)
    plt.colorbar()
    plt.xlabel("S through centerline")
    plt.ylabel("r")
    plt.title("Concentration contour in the tissue")



    eDam=round(np.max(K_m)*Rv/Dv,4)
    parameters["eDam"]=eDam
    Dat=round(L_c**2*Mt/(parameters["Inlet"]*Dt),4)
    parameters["Dat"]=Dat
    epsilon=np.round(Rv/L_c, 2)
    parameters["epsilon"]=epsilon
    
    #Create class
    k=solver(A,vessel ,tissue,  parameters, geom)
    #Set Dirichlet in the entry of the vessel and the borders of the tissue 
    k.set_DirichletSS_sparse()
    k.set_Dirichlet_south(2)
    

    eDam=round(np.max(K_m)*Rv/Dv,4)
    parameters["eDam"]=eDam
    Dat=round(L_c**2*Mt/(parameters["Inlet"]*Dt),4)
    parameters["Dat"]=Dat
    epsilon=np.round(Rv/L_c, 2)
    parameters["epsilon"]=epsilon
    
    string="$\epsilon$ $Da_m$={i},$Da_t$= {j}, $\epsilon$= {ep}.pdf".format(i=eDam, j=Dat, ep=epsilon)
    start=time.time()
    k.solve_sparse_SS(geom,string)
    k.call_plot_solution(k.sol_SS, 9999, "Dirichlet South")
    end=time.time()
    print("solve function took: {t}s".format(t=end-start))
    return(k.sol_SS)


#CONVERGENCE TEST        
#To check solution resolution/Discretization:
    
class sol_cl():
    def __init__(self, solution,rv, rt, z):
        self.rt=rt
        self.rv=rv
        self.r=np.append(rv,rt)
        self.z=z
        self.Z,self.Rho=np.meshgrid(z,self.r)
        self.solution=solution