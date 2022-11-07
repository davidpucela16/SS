#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:21:06 2021

@author: pdavid
"""
import numpy as np
from module_2D_coupling import * 

def reconstruction_gradients_manual(solution, assembly_object, boundary):
    t=assembly_object
    corners=t.get_corners()
    phi=solution
    
    #gradient assembly
    grad=np.zeros((len(t.FV_DoF),4))
    for i in np.delete(t.FV_DoF,boundary):
        j=np.array([len(t.x), -len(t.x), 1,-1])+i
        grad[i,:]=(phi[j]-phi[i])/t.h
    
    for i in boundary[0]: #north
        if i not in corners:
           n=np.array([-len(t.x), 1,-1])+i
           grad[i,[1,2,3]]=(phi[n]-phi[i])/t.h
           grad[i,0]=-phi[i]*2/t.h
           
    for i in boundary[1]: #south
        if i not in corners:
           n=np.array([len(t.x), 1,-1])+i
           grad[i,[0,2,3]]=(phi[n]-phi[i])/t.h
           grad[i,1]=-phi[i]*2/t.h 
           
    for i in boundary[2]: #east
        if i not in corners:
           n=np.array([len(t.x),-len(t.x),-1])+i
           grad[i,[0,1,3]]=(phi[n]-phi[i])/t.h
           grad[i,2]=-phi[i]*2/t.h 
           
    for i in boundary[3]: #east
        if i not in corners:
           n=np.array([len(t.x),-len(t.x),1])+i
           grad[i,[0,1,2]]=(phi[n]-phi[i])/t.h
           grad[i,3]=-phi[i]*2/t.h 
           
    #corners
    c=0
    for i in np.array([boundary[1,0],boundary[1,-1], boundary[0,0], boundary[0,-1]]):
        if c==0:
            bord=np.array([1,3])
        elif c==1:
            bord=np.array([1,2])
        elif c==2:
            bord=np.array([0,3])
        elif c==3:
            bord=np.array([0,2])
        no_bord=np.delete(np.arange(4), bord)
        neigh=np.array([len(t.x),-len(t.x),1,-1])[no_bord]+i
        grad[i,no_bord]=(phi[neigh]-phi[i])/t.h
        grad[i,bord]=-phi[i]*2/t.h
        
        c+=1
    return(grad)
    
    
def get_grad_s_block(solution, x,y, block_ID, s_blocks, pos_s, h):
    
    xlen, ylen=len(x), len(y)
    coord_block=np.array([x[block_ID%xlen], y[block_ID//ylen]])
    s_ID=np.where(s_blocks==block_ID)[0]
    q=solution[xlen*ylen+len(np.unique(s_blocks))*9+s_ID]
    grad_S=np.zeros((4,0))
    for i in s_ID:
        grad_S=np.concatenate([grad_S, np.array([get_trans(h,h, pos_s[i]-coord_block)]).T], axis=1)
    grad=np.dot(grad_S, q)/h
    return(grad)

# =============================================================================
# def get_full_grad_s_block(ass_object, solution):
#     t=ass_object
#     
#     xlen=len(t.x)
#     ylen=len(t.y)
#     
#     neigh=np.array([xlen, -xlen, 1,-1])
#     
#     T=-np.dot(t.c, solution[-2:])[neigh]
#     Flux=np.dot(t.b,solution[xlen*ylen:-len(t.s_blocks)])*t.D
#     return(Flux)
# =============================================================================
    
def separate_unk(ass_object, solution):
    """gets three arrays with the FV solution, reg terms, and sources"""
    xy=len(ass_object.x)*len(ass_object.y)
    s=len(ass_object.s_blocks)
    return(solution[:xy], solution[xy:-s], solution[-s:])    
    
def manual_test_s_block_grads(ass_object, phi, block_ID):
    t=ass_object
    D=t.D
    xlen=len(t.x)
    neigh=np.array([xlen, -xlen, 1,-1])+block_ID
    #first the fluxes that are calculated with the matrix
    p_FV, p_v, p_q=separate_unk(t, phi)
    rel_flux_v=np.dot(t.b, p_v)[neigh]
    rel_flux_G=np.dot(t.c, p_q)[neigh]
    
    pos_v0=np.where(np.unique(t.s_blocks)==block_ID)[0][0]*9
    dat=np.array([0.5,1,0.5,-0.5,-1,-0.5])/t.h
    _,phi_v,_=separate_unk(t, phi)
    grad_v=np.zeros(4)
    #now calculated manually:
    for i in range(4):
        if i==0: #north
            col=np.array([0,1,2,3,4,5])+pos_v0
            grad=np.dot(phi_v[col], dat)
        if i==1: #south
            col=np.array([6,7,8,3,4,5])+pos_v0
            grad=np.dot(phi_v[col], dat)
        if i==2: #east
            col=np.array([2,5,8,1,4,7])+pos_v0
            grad=np.dot(phi_v[col], dat)
        if i==3: #west
            col=np.array([0,3,6,1,4,7])+pos_v0
            grad=np.dot(phi_v[col], dat)
        grad_v[i]=grad
    Flux_v=-grad_v*D*t.h
    Flux_G=-get_grad_s_block(phi, t.x,t.y, block_ID, t.s_blocks, t.pos_s, t.h)*D*t.h
    
    print("The matrix calculated v_flux: ",rel_flux_v*t.h**2)
    print("The manually calculated v_flux: ",Flux_v)
    print("The matrix calculated G_flux: ",rel_flux_G*t.h**2)
    print("The manually calculated G_flux: ",Flux_G)
    print("The total calculated flux: ",rel_flux_G*t.h**2+rel_flux_v*t.h**2)
    print("The total calculated flux: ",Flux_G+Flux_v)
    
    print("div v", np.sum(Flux_v))
    print("total leaving flux", np.sum(Flux_G+Flux_v))
    
    return(grad_v+get_grad_s_block(phi, t.x,t.y, block_ID, t.s_blocks, t.pos_s, t.h))
            

#def micro_reconstruction_linear(ass_object, solution, h_new, L):
    
def reconstruct_block_value(ass_object, solution, block_ID):
    """Interesting function that will return the value reconstructed from the 
    gradients from teh function manual_test_s_block_grads. It shows that the 
    "block pressure" normally calculated is generally a gross approximation"""
    t=ass_object
    phi=solution
    xlen=len(t.x)
    neigh=block_ID+np.array([xlen, -xlen,1,-1])
    grads=manual_test_s_block_grads(t, phi, block_ID)
    
    values=np.zeros(4)
    for i in range(4):
        values[i]=phi[neigh[i]]-grads[i]
    return(values)
        
    
class reconst_microscopic():
    def __init__(self, ass_object, solution):
        self.t=ass_object
        self.phi=solution
        
        
    def reconstruct(self, ratio,L):
        t=self.t
        h=t.h/ratio
        num=int(L//h)
        h=L/num
        x=np.linspace(h/2, L-h/2, num)
        y=x
        
        self.x, self.y=x,y
        grads=reconstruction_gradients_manual(self.phi, t,get_boundary_vector(len(t.x), len(t.y)))  
        self.sol=np.zeros((len(y), len(x)))
        
        for i in range(len(t.x)*len(t.y)):
            local_sol,pos_x,pos_y=self.reconstruction_single_block(i, grads,t)
            xx,yy=np.arange(len(x))[pos_x], np.arange(len(y))[pos_y]
            self.sol[yy[0]:yy[0]+len(yy),xx[0]:xx[0]+len(xx)]=local_sol
            
        #block_x=np.where()
        return(self.sol)
    
    def reconstruction_single_block(self, block_ID, grads,ass_object):
        t=ass_object #coarse SS object
        xlim=t.x[block_ID%len(t.x)]+np.array([-t.h/2,t.h/2])
        ylim=t.y[int(block_ID//len(t.x))]+np.array([-t.h/2,t.h/2])
        
        pos_x=((self.x>=xlim[0]) & (self.x<xlim[1]))
        
        pos_y=((self.y>=ylim[0]) & (self.y<ylim[1]))
        
        local_sol=np.zeros((np.sum(pos_y), np.sum(pos_x)))
        c=0
        
        for i in self.x[pos_x]:
            d=0
            for j in self.y[pos_y]:
                rel_pos=np.array([i,j])-pos_to_coords(t.x, t.y, block_ID)
                if block_ID not in t.s_blocks:
                    inc_x=grads[block_ID,2]*rel_pos[0] if rel_pos[0]>=0 else grads[block_ID,3]*np.abs(rel_pos[0])
                    inc_y=grads[block_ID, 0]*rel_pos[1] if rel_pos[1]>=0 else grads[block_ID, 1]*np.abs(rel_pos[1])
                    
                    local_sol[d,c]=self.phi[block_ID] + inc_y + inc_x
                else:
                    s_ID=np.where(t.s_blocks==block_ID)[0]
                    phi_FV, phi_v, phi_q=separate_unk(t, self.phi)
                    pos_v0=np.where(block_ID==np.unique(t.s_blocks))[0][0]*9
                    pos_v, coeffs=FD_linear_interp(rel_pos, t.h)
                    v_values=phi_v[pos_v0+pos_v]
                    
                    s_IDs=np.where(t.s_blocks==block_ID)[0]
                    
                    Sing=np.array([])
                    for k in np.where(t.s_blocks==block_ID)[0]:
                        dist=np.linalg.norm(t.pos_s[k]-np.array([i,j]))
                        S=Green(t.pos_s[k], np.array([i,j]), t.Rv) if dist>t.Rv else 0
                        Sing=np.append(Sing, S) 
                        
                    Sing=np.dot(Sing, phi_q[s_IDs])
                    local_sol[d,c]=np.dot(v_values, coeffs)+Sing
                    
                d+=1
            c+=1
        return(local_sol, pos_x, pos_y)
    
    def recostruction_source_block(self, block_ID, ass_object):
        t=ass_object
        
        
        

    