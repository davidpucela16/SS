#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:21:06 2021

@author: pdavid
"""
import os 
import numpy as np
from module_2D_coupling_FV_nogrid import * 


def reconstruction_gradients_manual(solution, assembly_object, boundary, *zero_Dirich):
    """I think this function always assumes zero Dirichlet
    
    Add the argument zero_Dirich if 
    """
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
           if zero_Dirich:
               grad[i,0]=-phi[i]*2/t.h
           
    for i in boundary[1]: #south
        if i not in corners:
           n=np.array([len(t.x), 1,-1])+i
           grad[i,[0,2,3]]=(phi[n]-phi[i])/t.h
           if zero_Dirich:
               grad[i,1]=-phi[i]*2/t.h 
           
    for i in boundary[2]: #east
        if i not in corners:
           n=np.array([len(t.x),-len(t.x),-1])+i
           grad[i,[0,1,3]]=(phi[n]-phi[i])/t.h
           if zero_Dirich:
               grad[i,2]=-phi[i]*2/t.h 
           
    for i in boundary[3]: #east
        if i not in corners:
           n=np.array([len(t.x),-len(t.x),1])+i
           grad[i,[0,1,2]]=(phi[n]-phi[i])/t.h
           if zero_Dirich:
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
        if zero_Dirich:
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


        

def manual_test_s_block_grads(ass_object, phi, block_ID, directness):
    """Reconstructs the values of the gradients in a source block a posteriori"""
    
    t=ass_object
    D=t.D
    xlen=len(t.x)
    neigh=np.array([xlen, -xlen, 1,-1])+block_ID
    
    k=block_ID
    
    p_k=pos_to_coords(t.x, t.y, k)
    m_pos_k=np.where(t.uni_s_blocks==k)[0][0] #position within the Mid matrix 
    ind_s_neigh=np.in1d(neigh, t.s_blocks) 
    s_neigh=neigh[ind_s_neigh] #FV ID of the neighbours with sources
    
    #pos_s_neigh=np.in1d(t.uni_s_blocks,s_neigh) #DEF matrix position of neighbours with sources 

    phi_FV,phi_v,phi_q=separate_unk(t, phi)
    Flux_v=np.zeros(4)
    Full_flux=np.zeros(4)
    
    #sources_k=np.where(t.s_blocks==k)[0]
    #Green_k=t.get_singular_term(block_ID, "real") #get the array to obtain the value of singular term 
                                                    #at neighbours cell centers
    loc_neigh=get_neighbourhood(directness, len(t.x), block_ID) #local neighbourhood
    
    sources_neigh_k=np.in1d(t.s_blocks,loc_neigh)
    loc_bar_neigh=np.delete(loc_neigh, np.where(loc_neigh==block_ID)[0][0]) #local neighbourhood excluding k
    Trans_k=assemble_array_block_trans(t.s_blocks, t.pos_s, k, t.h, t.h,t.x,t.y)
    #now calculated manually:
    for m in range(4):
        p_m=pos_to_coords(t.x, t.y, neigh[m])
        if neigh[m] not in t.uni_s_blocks:
            
            G_term_bar_neigh=np.dot(get_Green_neigh_array(p_k, loc_bar_neigh, t.s_blocks, t.pos_s, t.Rv),phi_q)
            G_term_k=np.dot(get_Green_neigh_array(p_m, np.array([k]), t.s_blocks, t.pos_s, t.Rv),phi_q)
            Flux_v[m]=phi_v[m_pos_k]-phi[neigh[m]]+G_term_bar_neigh+G_term_k
            
            Full_flux[m]=phi_v[m_pos_k]-phi[neigh[m]]+G_term_bar_neigh+G_term_k-np.dot(Trans_k[m], phi_q)
        else:
            m_pos_l=np.where(t.uni_s_blocks==neigh[m])[0][0] #position within the Mid matrix 
            neigh_l=get_neighbourhood(directness, len(t.x), neigh[m])
            Q_dom_kl=get_uncommon(loc_neigh, neigh_l, len(t.x))
            Q_dom_lk=get_uncommon(neigh_l, loc_neigh, len(t.x))
            
            #sources_kl=np.arange(len(t.s_blocks))[np.in1d(t.s_blocks, Q_dom_kl)]
            #sources_lk=np.arange(len(t.s_blocks))[np.in1d(t.s_blocks, Q_dom_lk)]
            
            G_term=np.dot(get_Green_neigh_array(p_m, Q_dom_lk, t.s_blocks, t.pos_s, t.Rv), phi_q)
            G_term-=np.dot(get_Green_neigh_array(p_m, Q_dom_kl, t.s_blocks, t.pos_s, t.Rv), phi_q)
            Flux_v[m]=phi_v[m_pos_k]-phi_v[m_pos_l] - G_term #local reg term flux
            
            surf_p1, surf_p2=get_surf_points(k, t.x, t.y, m, t.h)
            normal=(p_m-p_k)/np.linalg.norm(p_m-p_k)
            Full_flux[m]=phi_v[m_pos_k]-phi_v[m_pos_l]- \
            np.dot(array_trans_random_surf(sources_neigh_k, t.pos_s,
                                           surf_p1, surf_p2, normal), phi_q)
                                                                                      
    G_flux=np.dot(Trans_k, phi_q)
    return(Full_flux,Flux_v, G_flux)

def get_trans_random_surf(surf_p1, surf_p2, s):
    """Will return the transmissibility from a source situated at s to a surface defined
    by its initial and ending point"""
    p0=surf_p1-s
    p1=surf_p2-s
    angle=np.arctan(p1[1]/p1[0])-np.arctan(p0[1]/p1[0])
    return(np.abs(angle)/2/np.pi)

def array_trans_random_surf(boolean_of_sources, pos_s, surf_p1, surf_p2,normal):
    """returns the array to multiply """
    array=np.zeros(len(pos_s))
    middle=0.5*surf_p1+0.5*surf_p2
    for i in np.arange(len(pos_s))[boolean_of_sources]:
        sc=np.dot(middle-pos_s[i], normal)
        array[i]=get_trans_random_surf(surf_p1, surf_p2, pos_s[i])*np.sign(sc)
    return(array)
        
        


def get_surf_points(block_ID, x,y, c,h):
    """returns the initial and ending points of the block's surface c"""
    a=pos_to_coords(x, y, block_ID)
    if c==0:
        points=a+np.array([[-1,1],[1,1]])*h/2
    elif c==1:
        points=a+np.array([[-1,-1],[1,-1]])*h/2
    elif c==2:
        points=a+np.array([[1,-1],[1,1]])*h/2
    elif c==3:
        points=a+np.array([[-1,-1],[-1,1]])*h/2
    return(points)



#def micro_reconstruction_linear(ass_object, solution, h_new, L):
    
def reconstruct_from_gradients(phi, grads, ratio, orig_x, orig_y, orig_h):

    h=orig_h/ratio 
    L=orig_x[-1]+orig_h/2
    
    num=int(L//h)
    h=L/num
    x=np.linspace(h/2, L-h/2, num)
    y=x
    rec=np.zeros((len(y), len(x)))
    for k in range(len(orig_x)*len(orig_y)):
        #pdb.set_trace()
        pos_x, pos_y, local_sol=reconstruct_FV_block(grads, orig_x, orig_y, k, phi, x,y, orig_h)
        pos_x=np.arange(len(pos_x))[pos_x]
        pos_y=np.arange(len(pos_y))[pos_y]
        rec[pos_y[0]:pos_y[0]+len(pos_y), pos_x[0]:pos_x[0]+len(pos_x)]=local_sol
    return(rec)

def reconst_sol_split(phi_v, phi_q, grads, ratio, orig_x, orig_y, orig_h, pos_s, Rv):
    phi=phi_v
    h=orig_h/ratio 
    L=orig_x[-1]+orig_h/2
    
    num=int(L//h)
    h=L/num
    x=np.linspace(h/2, L-h/2, num)
    y=x
    rec=np.zeros((len(y), len(x)))
    for k in range(len(orig_x)*len(orig_y)):
        #pdb.set_trace()
        pos_x, pos_y, local_sol=reconstruct_FV_block(grads, orig_x, orig_y, k, phi, x,y, orig_h)
        pos_x=np.arange(len(pos_x))[pos_x]
        pos_y=np.arange(len(pos_y))[pos_y]
        rec[pos_y[0]:pos_y[0]+len(pos_y), pos_x[0]:pos_x[0]+len(pos_x)]=local_sol
        c=0
        pdb.set_trace()
        for i in pos_s:
            g_value=Green(i,np.array([x[pos_x[len(x)//2]],y[pos_y[len(y)//2]]]), Rv)
            rec[pos_y[0]:pos_y[0]+len(pos_y), pos_x[0]:pos_x[0]+len(pos_x)]+=g_value*phi_q[c]
            c+=1

def reconstruct_FV_block(grads, orig_x,orig_y, block_ID, phi, x, y, orig_h):
    xlim=orig_x[block_ID%len(orig_x)]+np.array([-orig_h/2,orig_h/2])
    ylim=orig_y[int(block_ID//len(orig_x))]+np.array([-orig_h/2,orig_h/2])
    
    pos_x=((x>=xlim[0]) & (x<xlim[1]))
    pos_y=((y>=ylim[0]) & (y<ylim[1]))
    
    local_sol=np.zeros((np.sum(pos_y), np.sum(pos_x)))+phi[block_ID] 

    c=0
    for i in x[pos_x]:
        d=0
        for j in y[pos_y]:
            rel_pos=np.array([i,j])-pos_to_coords(orig_x, orig_y, block_ID)
            inc_x=grads[block_ID,2]*rel_pos[0] if rel_pos[0]>=0 else grads[block_ID,3]*np.abs(rel_pos[0])
            inc_y=grads[block_ID, 0]*rel_pos[1] if rel_pos[1]>=0 else grads[block_ID, 1]*np.abs(rel_pos[1])
            local_sol[d,c]+=inc_x+inc_y
            d+=1
        c+=1

    return(pos_x, pos_y, local_sol)
        

def reconstruct_source_block(phi_q,v_block, block_ID, orig_h, orig_x, orig_y, x, y, directness, s_blocks, pos_s, Rv):
    
    xlim=orig_x[block_ID%len(orig_x)]+np.array([-orig_h/2,orig_h/2])
    ylim=orig_y[int(block_ID//len(orig_x))]+np.array([-orig_h/2,orig_h/2])
    
    pos_x=((x>=xlim[0]) & (x<xlim[1]))
    pos_y=((y>=ylim[0]) & (y<ylim[1]))
    
    local_sol=np.zeros((np.sum(pos_y), np.sum(pos_x)))+v_block 
    neigh=get_neighbourhood(directness, len(orig_x), block_ID)
    sources=np.arange(len(s_blocks))[np.in1d(s_blocks, neigh)]
    c=0
    for i in x[pos_x]:
        d=0
        for j in y[pos_y]:
            g=np.zeros(len(s_blocks))
            for p in sources: #goes through the ID of the sources 
                g[p]=Green(pos_s[p], np.array([i,j]), Rv)
            local_sol[d, c]+=np.dot(g,phi_q)
            
            d+=1
        c+=1
    return(local_sol)
            
    
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

def get_sub_x_y(orig_x, orig_y, orig_h, ratio):
    h=orig_h/ratio 
    L=orig_x[-1]+orig_h/2
    num=int(L//h)
    h=L/num
    x=np.linspace(h/2, L-h/2, num)
    y=x
    return(x,y)

def modify_matrix(original, pos_x, pos_y, value_matrix, x_micro_len):
    
    array_pos=np.array([], dtype=int)
    for j in pos_y:
        array_pos=np.concatenate((array_pos, pos_x+j*x_micro_len))
    original_flat=np.ndarray.flatten(original)
    original_flat[array_pos]=np.ndarray.flatten(value_matrix)
    return(original_flat.reshape(original.shape))

class reconstruct_coupling():
    def __init__(self, solution, directness_neigh, ass_object, directness):
        self.dir=directness_neigh
        phi_FV, phi_v, phi_q=separate_unk(ass_object, solution)
        self.t=ass_object
        grads_FV=reconstruction_gradients_manual(phi_FV, self.t, self.t.boundary)
        for i in self.t.uni_s_blocks:
            neigh=np.array([len(self.t.x), -len(self.t.x),1,-1])+i
            Flux,_,_=manual_test_s_block_grads(self.t, solution, i, directness)
            grads_FV[neigh, np.array([1,0,3,2])]=Flux/(self.t.D*self.t.h)
        self.FV, self.v, self.q=phi_FV, phi_v, phi_q
        self.grads_FV=grads_FV
    def reconstruction(self, ratio):
        t=self.t
        rec=reconstruct_from_gradients(self.FV, self.grads_FV, ratio, t.x, t.y, t.h)
        for i in t.uni_s_blocks:
            x,y=get_sub_x_y(t.x, t.y, t.h, ratio)
            self.x, self.y=x,y
            xlim=t.x[i%len(t.x)]+np.array([-t.h/2,t.h/2])
            ylim=t.y[int(i//len(t.x))]+np.array([-t.h/2,t.h/2])
            
            pos_x=((x>=xlim[0]) & (x<xlim[1]))
            pos_y=((y>=ylim[0]) & (y<ylim[1]))
            v_block=self.v[np.where(t.uni_s_blocks==i)[0][0]]
            local_sol=reconstruct_source_block(self.q,v_block, i, t.h, t.x, t.y, x, y, self.dir, t.s_blocks, t.pos_s, t.Rv)
            rec=modify_matrix(rec, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], local_sol, len(x))
        return(rec)