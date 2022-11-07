#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:21:06 2021

@author: pdavid
"""
import os 
import numpy as np
from module_2D_coupling_FV_nogrid import * 


class reconst_microscopic():
    def __init__(self, ass_object, solution):
        self.t=ass_object
        self.phi=solution
        self.grad_v=np.zeros((len(self.t.uni_s_blocks),4))
        c=0
        for i in self.t.uni_s_blocks:
            #pdb.set_trace()
            self.grad_v[c],_,_=manual_test_s_block_grads(self.t, self.phi, i)
            
            c+=1
    def reconstruct(self, ratio,L):
        """Right now, it uses the function reconstruction gradients manual to calculate the gradient"""
        t=self.t
        h=t.h/ratio
        num=int(L//h)
        h=L/num
        x=np.linspace(h/2, L-h/2, num)
        y=x
        
        self.x, self.y=x,y
        grads=reconst_real_fluxes_posteriori(self.phi, t)  
        self.sol=np.zeros((len(y), len(x)))
        self.grads=grads
        
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
                rel_pos=np.array([i,j])-pos_to_coords(t.x, t.y, block_ID) #relative position 
                if block_ID not in t.s_blocks:
                    inc_x=grads[block_ID,2]*rel_pos[0] if rel_pos[0]>=0 else grads[block_ID,3]*np.abs(rel_pos[0])
                    inc_y=grads[block_ID, 0]*rel_pos[1] if rel_pos[1]>=0 else grads[block_ID, 1]*np.abs(rel_pos[1])
                    
                    local_sol[d,c]=self.phi[block_ID] + inc_y + inc_x
                else:
                    
                    local_sol[d,c]=self.recons_source_block(block_ID, rel_pos,i,j)
                    
                    
                d+=1
            c+=1
        return(local_sol, pos_x, pos_y)
    
    def recons_source_block(self,block_ID, rel_pos,i,j):
        t=self.t
        phi_FV, phi_v, phi_q=separate_unk(t, self.phi)
        s_IDs=np.where(t.s_blocks==block_ID)[0] #contained sources in the block
        Sing=np.array([])
        #pdb.set_trace()
        for k in np.where(t.s_blocks==block_ID)[0]: #this loop calculates the singular term
            dist=np.linalg.norm(t.pos_s[k]-np.array([i,j]))
            S=Green(t.pos_s[k], np.array([i,j]), t.Rv) if dist>t.Rv else 0
            Sing=np.append(Sing, S) 
            
        Sing=np.dot(Sing, phi_q[s_IDs])
        grad_reg=self.grad_v[np.where(t.uni_s_blocks==block_ID)[0][0]]
        inc_x=grad_reg[2]*rel_pos[0] if rel_pos[0]>=0 else grad_reg[3]*np.abs(rel_pos[0])
        inc_y=grad_reg[0]*rel_pos[1] if rel_pos[1]>=0 else grad_reg[1]*np.abs(rel_pos[1])
        
        return(Sing+inc_x+inc_y+phi_v[np.where(t.uni_s_blocks==block_ID)[0][0]])
        

        
        


def reconst_real_fluxes_posteriori(full_solution, ass_object):
    """This function will return the gradients calculated a posteriori.
        -> for the FV the TPFA is used
        -> for the source blocks the a posteriori calculation of the gradients is used """
    phi=full_solution
    phi_FV, phi_v, phi_q=separate_unk(ass_object, phi)
    t=ass_object
    grads=reconstruction_gradients_manual(phi, t, t.boundary)
    neigh=np.array([len(t.x), -len(t.x), +1, -1])
    sides=np.array([1,0,3,2])
    for i in np.unique(t.s_blocks):
        _,_,grads[i]=manual_test_s_block_grads(t, phi, i)
        c=0
        
        for j in i+neigh:
            if j not in t.s_blocks:
                grads[j, sides[c]]=-grads[i,c]
            c+=1
    return(grads)
        
def reconstruction_gradients_manual(solution, assembly_object, boundary):
    """I think this function always assumes zero Dirichlet"""
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
    
    
def get_flux_singular(full_solution, x,y, block_ID, s_blocks, pos_s, h):
    """Only function that calculates the singular part of the gradient of a source block
    Tested?
    
    The transmissibilities are integrated so they need they calculate the flux"""
    xlen, ylen=len(x), len(y)
    coord_block=np.array([x[block_ID%xlen], y[block_ID//ylen]])
    s_ID=np.where(s_blocks==block_ID)[0]
    q=full_solution[xlen*ylen+len(np.unique(s_blocks))+s_ID]
    flux_S=np.zeros((4,0))
    for i in s_ID:
        flux_S=np.concatenate([flux_S, np.array([get_trans(h,h, pos_s[i]-coord_block)]).T], axis=1)
    grad=np.dot(flux_S, q)/h
            
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
    """Reconstructs the values of the gradients in a source block a posteriori"""
    t=ass_object
    D=t.D
    xlen=len(t.x)
    neigh=np.array([xlen, -xlen, 1,-1])+block_ID
    
    k=block_ID
    m_pos_k=np.where(t.uni_s_blocks==k)[0][0] #position within the Mid matrix 
    ind_s_neigh=np.in1d(neigh, t.s_blocks) 
    s_neigh=neigh[ind_s_neigh] #FV ID of the neighbours with sources
    
    pos_s_neigh=np.where(t.uni_s_blocks==s_neigh)[0] #DEF matrix position of neighbours with sources 


    
    phi_FV,phi_v,phi_q=separate_unk(t, phi)
    Flux_v=np.zeros(4)
    
    sources_k=np.where(t.s_blocks==k)[0]
    Green_k=t.get_singular_term(block_ID, "real") #get the array to obtain the value of singular term 
                                                    #at neighbours cell centers
    #now calculated manually:
    #pdb.set_trace()
    for i in range(4):
        G0k=t.get_singular_term(k, "full_real")
        G0k_short=G0k[-4:,:]
        if neigh[i] not in t.s_blocks:
            flux=(phi_FV[neigh[i]]-np.dot(G0k[i,:], phi_q)-phi_v[m_pos_k])
# =============================================================================
#         else:
#             m_pos_neigh=np.where(t.uni_s_blocks==neigh[i])[0][0]
#             Green_neigh=t.get_singular_term(neigh[i], "real")
#             j=np.array([1,0,3,2])[i]
#             grad=phi_v[m_pos_neigh]-np.dot(Green_k[i,:], phi_q)-phi_v[m_pos_k]+np.dot(Green_neigh[j,:],phi_q)
#             grad/=t.h
# =============================================================================
        else:
            m=neigh[i]
            m_pos_m=np.where(t.uni_s_blocks==m)[0][0]
            c=get_side(k, m, t.x)

            op_sides=np.array([1,0,3,2])
            G0m=t.get_singular_term(m, "full_real")
            G0m_short=G0m[-4:,:]
            
            Tk=assemble_array_block_trans(t.s_blocks, t.pos_s, k, t.h, t.h ,t.x,t.y)
            Tm=assemble_array_block_trans(t.s_blocks, t.pos_s, m, t.h, t.h ,t.x,t.y)
            
            flux=phi_v[m_pos_m]-phi_v[m_pos_k]+np.dot(G0m_short[op_sides[c]], phi_q) \
            -np.dot(G0k_short[c], phi_q)-0.5*np.dot(Tm[op_sides[c]], phi_q)-0.5*np.dot(Tk[c], phi_q)
            
        Flux_v[i]=flux*t.D
    grad_v=Flux_v/(t.D*t.h)
    Flux_G=-get_flux_singular(phi, t.x,t.y, block_ID, t.s_blocks, t.pos_s, t.h)*t.D
    grad_G=Flux_G/(t.D*t.h)
    
    print("The manually calculated v_flux: ",Flux_v)
    print("The manually calculated G_flux: ",Flux_G)
    Flux_phi=Flux_v+Flux_G
    print("The total calculated flux: ",Flux_phi)
    
    print("total leaving flux", np.sum(Flux_phi))
    
    return(grad_v, grad_G,-Flux_phi/(D*t.h))
            

#def micro_reconstruction_linear(ass_object, solution, h_new, L):
    
def reconstruct_block_value(ass_object, solution, block_ID):
    """Interesting function that will return the value reconstructed from the 
    gradients from teh function manual_test_s_block_grads. It shows that the 
    "block pressure" normally calculated is generally a gross approximation
    
    The return value is the real peaceman equivalent pressure calculated from each neighbourg"""
    t=ass_object
    phi=solution
    xlen=len(t.x)
    neigh=block_ID+np.array([xlen, -xlen,1,-1])
    _,_,grads=manual_test_s_block_grads(t, phi, block_ID)
    
    values=np.zeros(4)
    for i in range(4):
        values[i]=phi[neigh[i]]-grads[i]
    return(values)
        
    

    