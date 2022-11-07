#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:59:14 2021

@author: pdavid

MAIN MODULE FOR THE SOLUTION SPLIT COUPLING MODEL IN 2D!

This is the first coupling module that I manage to succeed with some type of coupling

the coupling with the negihbouring FV works quite well.

The problem arises when coupling two contiguous source blocks. Since there is no 
continuity enforced explicitly the solution does not respec C1 nor C0 continuity.

Furthermore, 
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import scipy as sp
from scipy import sparse
import math
import pdb

def A_assembly(cxlen, cylen):
    """This function assembles the laplacian operator for cartesian coordinates in 2D
    - It does not take into account the discretization size h, therefore, it must be homogeneous
    - For the same reason, if the discretization size is different than 1 there will be a factor to 
      multiply/divide the operator
    - As well, there is no diffusion coefficient considered
    
    INPUT -> the x and y length
    OUTPUT -> FD Laplacian operator
    """
    A=np.zeros([cxlen* cylen, cxlen*cylen])
    north=np.arange(cxlen* (cylen-1), cxlen* cylen)
    south=np.arange(cxlen)
    west=np.arange(0,cxlen* cylen, cxlen)
    east=np.arange(cxlen-1, cxlen* cylen,cxlen)
    
    boundary=np.concatenate([north, south, east, west])
    
    corners=np.array([0,cxlen-1,cxlen*(cylen-1), cxlen*cylen-1])
    
    for i in range(cxlen*cylen):
        if i not in boundary:
            A[i,i]-=4
            A[i,i+1]+=1
            A[i,i-1]+=1
            A[i,i+cxlen]+=1
            A[i,i-cxlen]+=1
            
        else:        
            if i in north:
                others=[1,-1,-cxlen]
            if i in south:
                others=[1,-1,cxlen]
            if i in east:
                others=[cxlen, -cxlen, -1]
            if i in west:
                others=[cxlen, -cxlen, 1]
                
            if i==0:
                #corner sudwest
                others=[1,cxlen]
            if i==cxlen-1:
                #sud east
                others=[-1,cxlen]
            if i==cxlen*(cylen-1):
                #north west
                others=[1,-cxlen]
            if i==cxlen*cylen-1:
                others=[-1,-cxlen]
            
            A[i,i]=-len(others)
            for n in others:
                A[i,i+n]=1
        
    return(A)

def A_assembly_Dirich(cxlen, cylen):
    """This function assembles the laplacian operator for cartesian coordinates in 2D
    - It does not take into account the discretization size h, therefore, it must be homogeneous
    - For the same reason, if the discretization size is different than 1 there will be a factor to 
      multiply/divide the operator
    - As well, there is no diffusion coefficient considered
    
    INPUT -> the x and y length
    OUTPUT -> FD Laplacian operator
    """
    A=np.zeros([cxlen* cylen, cxlen*cylen])
    north=np.arange(cxlen* (cylen-1), cxlen* cylen)
    south=np.arange(cxlen)
    west=np.arange(0,cxlen* cylen, cxlen)
    east=np.arange(cxlen-1, cxlen* cylen,cxlen)
    
    boundary=np.concatenate([north, south, east, west])
    
    corners=np.array([0,cxlen-1,cxlen*(cylen-1), cxlen*cylen-1])
    
    for i in range(cxlen*cylen):
        if i not in boundary:
            A[i,i]-=4
            A[i,i+1]+=1
            A[i,i-1]+=1
            A[i,i+cxlen]+=1
            A[i,i-cxlen]+=1
            
        else:        
            if i in north:
                others=[1,-1,-cxlen]
            if i in south:
                others=[1,-1,cxlen]
            if i in east:
                others=[cxlen, -cxlen, -1]
            if i in west:
                others=[cxlen, -cxlen, 1]
                
            if i==0:
                #corner sudwest
                others=[1,cxlen]
            if i==cxlen-1:
                #sud east
                others=[-1,cxlen]
            if i==cxlen*(cylen-1):
                #north west
                others=[1,-cxlen]
            if i==cxlen*cylen-1:
                others=[-1,-cxlen]
            
            A[i,i]=-4
            for n in others:
                A[i,i+n]=1
        
    return(A)



class assemble_SS_2D_FD():
    def __init__(self, pos_s, A, Rv, h,x,y, K_eff, D,directness):          
        self.x=x
        self.y=y
        self.C_0=K_eff*np.pi*Rv**2
        self.pos_s=pos_s #exact position of each source
        
        self.n_sources=self.pos_s.shape[0]
        self.A=A
        self.Rv=Rv
        self.h=h
        self.D=D
        self.boundary=get_boundary_vector(len(x), len(y))
        self.directness=directness

    def pos_arrays(self):
        #pos_s will dictate the ID of the sources by the order they are kept in it!
        source_FV=np.array([]).astype(int)
        uni_s_blocks=np.array([], dtype=int)
        for u in self.pos_s:
            r=np.argmin(np.abs(self.x-u[0]))
            c=np.argmin(np.abs(self.y-u[1]))
            source_FV=np.append(source_FV, c*len(self.x)+r) #block where the source u is located
            if c*len(self.x)+r not in uni_s_blocks:
                uni_s_blocks=np.append(uni_s_blocks, c*len(self.x)+r)
            
        self.FV_DoF=np.arange(len(self.x)*len(self.y))
        self.s_blocks=source_FV #for each entry it shows the block of that source
        self.uni_s_blocks=uni_s_blocks
        
        total_sb=len(np.unique(self.s_blocks)) #total amount of source blocks
        self.total_sb=total_sb
        
    def get_singular_term(self, block_ID, typ):
        """Returns the arrays of singular terms in the real neighbours or the phantom ones
        TEST AGAIN
        
        block_ID must be given in int form """
        
# =============================================================================
#         if type(block_ID)!=int and type(block_ID)!=np.int:
#             print("error in the type of the block_ID")
# =============================================================================
        
        pos_block_cent=np.array([self.x[block_ID%len(self.x)], self.y[block_ID//len(self.x)]])
        
        block_s=np.where(self.s_blocks==block_ID)[0] #IDs of sources in this block(always an array)
        if typ=="phantom":
            unk_pos=np.array([[-self.h, self.h/2],[-self.h/2, self.h],[self.h/2, self.h],[self.h, self.h/2],
                              [self.h,-self.h/2],[self.h/2, -self.h],[-self.h/2, -self.h],[-self.h, -self.h/2]])
        elif typ=="real":
            unk_pos=np.array([[0,1],[0,-1],[1,0],[-1,0]])*self.h
        
        elif typ=="full_real":
            #Will return the positions of the neighbours and the interfaces 
            unk_pos=np.array([[0,1],[0,-1],[1,0],[-1,0],[0,1/2],[0,-1/2],[1/2,0],[-1/2,0]])*self.h
        ret_array=np.zeros((unk_pos.shape[0],len(self.s_blocks)))
        c=0 #The counter c marks the position within the array of each DoF. It begins with north 
            #or northwest and continues in the same sense as the clock
        for i in unk_pos: #goes through each of the positions needed
            for j in block_s: #goes through each of the sources in this block
                ret_array[c,j]=Green(self.pos_s[j]-pos_block_cent,i,self.Rv)/self.D
            c+=1

        return(ret_array)
    
# =============================================================================
#     
#     def get_pos_DoF(self, block_ID):
#         """returns the absolute position of the DoFs"""
#         p=np.array([])
#         abs_pos_v=np.array([[-1,1],[0,1],[1,1],[-1,0],[0,0],[1,0],[-1,-1],[0,-1],[1,-1]])*self.h/2
#         abs_pos_v+=pos_to_coords(self.x, self.y, block_ID)
#         return(abs_pos_v)          
# =============================================================================
    
    def initialize_matrices(self):
        self.D_matrix=np.zeros([len(np.unique(self.s_blocks)), len(self.FV_DoF)])
        self.E_matrix=np.zeros([len(np.unique(self.s_blocks)), len(np.unique(self.s_blocks))])
        self.F1_matrix=np.zeros([len(np.unique(self.s_blocks)), len(self.s_blocks)])
        self.F2_matrix=np.zeros([len(np.unique(self.s_blocks)), len(self.s_blocks)])
        self.B_matrix=np.zeros([len(self.FV_DoF), len(np.unique(self.s_blocks))])
        self.C1_matrix=np.zeros([len(self.FV_DoF), len(self.s_blocks)])
        self.C2_matrix=np.zeros([len(self.FV_DoF), len(self.s_blocks)])
        
        self.G_matrix=np.zeros((len(self.s_blocks), len(self.FV_DoF)))
        self.H_matrix=np.zeros([len(self.s_blocks),len(np.unique(self.s_blocks))])
        self.I_matrix=np.zeros([len(self.s_blocks),len(self.s_blocks)])
    
    def assembly_sol_split_problem(self):
        #First of all it is needed to remove the source_blocks from the main matrix
        for i in np.unique(self.s_blocks):
            neigh=i+np.array([len(self.x), -len(self.x), 1,-1])

        
        self.initialize_matrices()
        
        for i in self.uni_s_blocks:
            self.mod_abc_matrix(i)
            self.mod_DEF_matrix(i)
            self.mod_GHI_matrix(i)
        
        for k in self.uni_s_blocks:
            self.C2_matrix[k,np.where(self.s_blocks==k)[0]]+=1
        
        self.C_matrix=self.C1_matrix+self.C2_matrix
        self.F_matrix=self.F1_matrix+self.F2_matrix
        Up=np.hstack((self.A, self.B_matrix, self.C_matrix))
        Mid=np.hstack((self.D_matrix, self.E_matrix, self.F_matrix))
        Down=np.hstack((self.G_matrix, self.H_matrix, self.I_matrix))
        
        
        
        self.Up=Up
        self.Mid=Mid
        self.Down=Down
        
        M=np.vstack((Up, Mid, Down))
        self.M=M
        return(M)
    
    def mod_abc_matrix(self, block_ID):
        """Modifies A, B and C matrices to include solution splitting fluxes from the s_blocks. 
        The function is called for every s_block"""
        #pdb.set_trace()
        k=block_ID
        p_k=pos_to_coords(self.x, self.y,k)
        neigh=np.array([len(self.x), -len(self.x), 1,-1])+k
        
        ind_s_neigh=np.in1d(neigh, self.s_blocks) #neighbours with sources
        ns_neigh=neigh[np.invert(ind_s_neigh)] #neighbours without sources
        
        self.A[ns_neigh,k]=0 #we eliminate the FV flux from the neighs without sources
        
        self.B_matrix[ns_neigh, np.where(self.uni_s_blocks==k)[0]]=self.D/self.h**2
        
        trans=assemble_array_block_trans(self.s_blocks, self.pos_s, k, self.h, self.h,self.x,self.y)[np.invert(ind_s_neigh),:]
        self.C2_matrix[ns_neigh,:]=-trans*self.D/self.h**2
        
        loc_neigh=get_neighbourhood(self.directness, len(self.x), block_ID) #local neighbourhood
        loc_bar_neigh=np.delete(loc_neigh, np.where(loc_neigh==block_ID)[0][0]) #local neighbourhood excluding k
        
        for F in ns_neigh:
            p_F=pos_to_coords(self.x, self.y,F)
            
            P_tilde_k_m=get_Green_neigh_array(p_F, np.array([k]), self.s_blocks, self.pos_s, self.Rv)
            S_bar_k_k=get_Green_neigh_array(p_k, loc_bar_neigh, self.s_blocks, self.pos_s, self.Rv)
            self.C1_matrix[F]+=(P_tilde_k_m+S_bar_k_k)*self.D/self.h**2
        return()
        
    def mod_DEF_matrix(self, block_ID):
        """This function here assembles the FD scheme without a subgrid"""
        k=block_ID
        pos_k=np.where(self.uni_s_blocks==k)[0][0]
        neigh=np.array([len(self.x), -len(self.x), 1,-1])+k
        loc_neigh_k=get_neighbourhood(self.directness, len(self.x), block_ID)
        for l in neigh:
            if l in self.s_blocks:
                pos_l=np.where(self.uni_s_blocks==l)[0][0]
                p_l=pos_to_coords(self.x, self.y, l)
                loc_neigh_l=get_neighbourhood(self.directness, len(self.x), l)
                self.E_matrix[pos_k, pos_l]+=1
                self.E_matrix[pos_k, pos_k]-=1
                unc_lk=get_uncommon(loc_neigh_l,loc_neigh_k , len(self.x))
                unc_kl=get_uncommon(loc_neigh_k, loc_neigh_l, len(self.x))
                
                S_unc_lk=get_Green_neigh_array(p_l, unc_lk, self.s_blocks, self.pos_s, self.Rv)
                S_unc_kl=get_Green_neigh_array(p_l, unc_kl, self.s_blocks, self.pos_s, self.Rv)
                self.F1_matrix[pos_k,:]+=S_unc_lk-S_unc_kl
            else:
                self.D_matrix[pos_k, l]+=1
                self.E_matrix[pos_k, pos_k]-=1
                p_l=pos_to_coords(self.x, self.y, l)
                S_loc_k=get_Green_neigh_array(p_l, loc_neigh_k, self.s_blocks, self.pos_s, self.Rv)
                self.F1_matrix[pos_k,:]-=S_loc_k
        
        return()
    
    def mod_GHI_matrix(self,block_ID):
        #pdb.set_trace()
        k=block_ID
        pos_k=np.where(self.uni_s_blocks==k)[0]
        loc_neigh_k=get_neighbourhood(self.directness, len(self.x), block_ID)
        sources_neigh=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks,loc_neigh_k)]
        sources_in=np.where(self.s_blocks==block_ID)[0] #sources that need to be solved in this block
        for i in sources_in:
            #other=np.delete(sources, i)
            other=np.delete(sources_neigh, np.where(sources_neigh==i))
            self.H_matrix[i,pos_k]=1 #regular term 
            self.I_matrix[i,i]=1/self.C_0 
            for j in other:
                self.I_matrix[i,j]+=Green(self.pos_s[i],self.pos_s[j], self.Rv)

    def get_grad_v(self, k, n):
        
        if n in self.uni_s_blocks:
            A,b,c1,c2=self.get_grad_source(k,n)
        else:
            d,e,f=self.get_grad_non_source(k,n)
        
        return(A,b,c1)
    
    def get_grad_non_source(self, k, n):
        """returns the arrays to add to the matrices A, b, and C to calculate the 
        flux (total, not relative to volume) that is exchanged between the given block
        and its non source neighbuorg.
        the array C1 takes care of the modification of the singular term to consider 
        the gradient of the regular term 
        In the matrix C2 the gradient cause by the singular term is considered
        
        Therefore, 
        
        k is the ID of the current block (the gradient will be given in the direction of k->n)
        n is the ID of the neighbourg
        
        grad_v=[A B C1]*phi
        grad_sing=[C2]*phi_q"""
        A=np.zeros(self.A.shape[1])
        B=np.zeros(len(np.unique(self.s_blocks)))

        
        A[n]=+1
        B[k]=-1
        pos_k=pos_to_coords(self.x, self.y, k)
        pos_n=pos_to_coords(self.x, self.y, n)
        neigh_k=get_neighbourhood(self.directness, len(self.x), k)
        loc_Omk=get_uncommon(neigh_k, np.array([k]), len(self.x)) #neighbourhood without k
        
        C1=-get_Green_neigh_array(pos_k, loc_Omk, self.s_blocks, self.pos_s, self.Rv)
        C1-=get_Green_neigh_array(pos_n, np.array([k]), self.s_blocks, self.pos_s, self.Rv)
        
        c=get_side(k,n,self.x)
        C2=assemble_array_block_trans(self.s_blocks, self.pos_s, k, self.h, self.h,self.x,self.y)[c]
        
        return(A,B,C1,C2)
    
    def get_grad_source(self, k, n ):
        
        d=np.zeros(self.A.shape[1])
        e=np.zeros(len(np.unique(self.s_blocks)))
        
        k_neigh=get_neighbourhood(self.directness, len(self.x), k)
        n_neigh=get_neighbourhood(self.directness, len(self.x), n)
        unc_n=get_uncommon(n_neigh, k_neigh, len(self.x))
        unc_k=get_uncommon(k_neigh, n_neigh, len(self.x))
        
        p_n=pos_to_coords(self.x, self.y, n)
        
        f=get_Green_neigh_array(p_n, unc_n, self.s_blocks, self.pos_s, self.Rv)
        f-=get_Green_neigh_array(p_n, unc_k, self.s_blocks, self.pos_s, self.Rv)
        return(d,e,f)

    def get_corners(self):
        corners=np.array([0,len(self.x)-1,len(self.x)*(len(self.y)-1), len(self.x)*len(self.y)-1])
        self.corners=corners
        return(corners)
        

    
def get_neighbourhood(directness, xlen, block_ID):
    if directness==1:
        loc_neigh=np.array([xlen-1, xlen, xlen+1, -1,0,1,-xlen-1, -xlen, -xlen+1])+block_ID
    elif directness==2: 
        arr=np.arange(-2,3)
        loc_neigh=np.concatenate((arr-xlen*2,arr-xlen, arr, arr+xlen, arr+2*xlen ))+block_ID
    return(loc_neigh)

def get_uncommon(k_neigh, n_neigh, xlen):
    """returns the cells of the first neighbourhood that has not in common with the
    second neighbourhood"""
    
    neigh_k_unc=k_neigh[np.invert(np.in1d(k_neigh, n_neigh))]
    return(neigh_k_unc)

def get_Green_neigh_array(p_x, neigh, s_blocks, pos_s, Rv):
    """Returns the array to multiply the array of source_fluxes that will calculate
    the value of the singular term of the sources in the given neigh at the given point p_x
    
    $\sum_{j \in neigh} G(x_j, p_x)$"""
    array=np.zeros(len(s_blocks))
    sources=np.arange(len(s_blocks))[np.in1d(s_blocks, neigh)]
    for i in sources:
        value=Green(pos_s[i], p_x, Rv)
        array[i]=value
    return(array)


    
@njit
def get_side(i, neigh, x):
    """It will return the side (north=0, south=1, east=2, west=3) the neighbour lies at"""
    lx=len(x)
    c=-5
    if i//lx == neigh//lx: #the neighbour lies in the same horizontal
        if neigh<i:
            c=3 #west
        else:
            c=2
    else: #the neighbours do not belong to the same horizontal
        if neigh>i:
            c=0
        else:
            c=1
    return(c)
        

@njit
def uni_vector(v0, vF):
    norm=np.sqrt(np.sum((vF-v0)**2))
    return((vF-v0)/norm)
    
def pos_to_coords(x, y, ID):
    xpos=ID%len(x)
    ypos=ID//len(x)
    return(np.array([x[xpos], y[ypos]]))

def coord_to_pos(x,y, coord):
    pos_x=np.argmin((coord[0]-x)**2)
    pos_y=np.argmin((coord[1]-y)**2)
    return(int(pos_x+pos_y*len(x)))

def get_trans(hx, hy, pos):
    """Computes the transmissibility from the cell's center to the surface
    ARE WE CONSIDERING THE DIFFUSION COEFFICIENT IN THE GREEN'S FUNCTION
    
    WHAT HAPPENS IF THE SOURCE FALLS RIGHT ON THE BORDER OF THE CELL"""
    #pos=position of the source relative to the cell's center
    a=np.array([-hx/2,hy/2]) #north west corner
    b=np.array([hx/2,hy/2]) #north east corner
    c=np.array([hx/2,-hy/2]) #south east corner
    d=np.array([-hx/2,-hy/2]) #south west corner
    theta=np.zeros((4,2))
    for i in range(4):
        if i==0: #north
            en=np.array([0,1]) #normal to north surface
            c1=a
            c2=b
        if i==1: #south
            en=np.array([0,-1])
            c1=d
            c2=c
        if i==2: #east
            en=np.array([1,0])
            c1=c
            c2=b
        if i==3: #west
            en=np.array([-1,0])
            c1=d
            c2=a
            
        theta[i,0]=np.arccos(np.dot(en, (c1-pos)/np.linalg.norm(c1-pos)))
        theta[i,1]=np.arccos(np.dot(en, (c2-pos)/np.linalg.norm(c2-pos)))
    return((-theta[:,0]-theta[:,1])/(2*np.pi))


def assemble_array_block_trans(s_blocks, pos_s, block_ID, hx, hy,x,y):
    """This function will return the to multiply the q unknown in order to assemble the gradient 
    produced by the sources in this block
    Therefore in the final array, there will be a zero in the sources that do not lie in this block and 
    the value of the transmissibility (for the given surface) for the sources that do lie within
    
    There are four lines in the output array, comme d'habitude the first one is north, then south, east
    and west"""
    sources=np.where(s_blocks==block_ID)[0]
    p_s=pos_s[sources]
    cord_block=pos_to_coords(x,y,block_ID)
    trans_array=np.zeros((4, len(s_blocks)))
    for i in range(len(sources)):
        a=get_trans(hx, hy, p_s[i]-cord_block)
        trans_array[:,sources[i]]=a
    return(trans_array)
        

def get_boundary_vector(xlen, ylen):
    #3- Set up the boundary arrays
    north=np.arange(xlen* (ylen-1), xlen* ylen)
    south=np.arange(xlen)
    west=np.arange(0,xlen* ylen, xlen)
    east=np.arange(xlen-1, xlen* ylen,xlen)
    return(np.array([north, south, east, west]))


@njit
def v_linear_interpolation(cell_center, x_pos, h):
    """this function is designed to give the coefficients that will multiply the values on the faces 
    of the cell to obtain a linear interpolation"""
    d=np.zeros(4)
    for i in range(4):
        if i==0:#a
            e=cell_center+np.array([-h/2,h/2])
        if i==1:#b
            e=cell_center+np.array([h/2,h/2])
        if i==2:#c
            e=cell_center+np.array([h/2,-h/2])
        if i==3:#d
            e=cell_center+np.array([-h/2,-h/2])
        d[i]=np.linalg.norm(x_pos-e)
    
    alpha=1/np.sum(1/d)
    return(alpha/d)


def FD_linear_interp(x_pos_relative, h):
    """returns the positions within the element to interpolate (pos) and the coefficients
    Bilinear interpolation"""
    x,y=x_pos_relative
    if x>=0:
        x1=0; x2=h/2
        if y>=0:
            y1=0; y2=h/2
            pos=np.array([4,5,1,2])
        if y<0:
            y1=-h/2; y2=0
            pos=np.array([7,8,4,5])
    elif x<0:
        x1=-h/2; x2=0
        if y>=0:
            y1=0; y2=h/2
            pos=np.array([3,4,0,1])
        if y<0:
            y1=-h/2; y2=0
            pos=np.array([6,7,3,4])
    r=np.array([(x2-x)*(y2-y),(x-x1)*(y2-y),(x2-x)*(y-y1),(x-x1)*(y-y1)])*4/h**2
    return(pos,r) 

@njit
def Green(q_pos, x_coord, Rv):
    """returns the value of the green function at the point x_coord, with respect to
    the source location (which is q_pos)"""
    if np.linalg.norm(q_pos-x_coord)>Rv:
        g=np.log(Rv/np.linalg.norm(q_pos-x_coord))/(2*np.pi)
    else:
        g=0
    return(g)


def grad_Green_norm(normal, pos1, pos2):
    """From the relative position of the source (with respect to the center of the block)
    rel_calc_point is the point where the gradient is calculated"""
    r=pos1-pos2
    sc=np.dot(r, normal)
    return(sc/(2*np.pi*np.linalg.norm(r)**2))

def block_grad_Green_norm_array(pos_s,s_blocks,  block_ID, pos_calcul, norm):
    """returns the array that will multiply the sources of the block_ID to obtain
    the gradient of the given block's singular term"""
    s_IDs=np.where(s_blocks==block_ID)[0]
    grad_green=np.zeros(pos_s.shape[0])
    for i in s_IDs:
        grad_green[i]=grad_Green_norm(norm, pos_s[i], pos_calcul)
    return(grad_green)

#block_grad_Green_norm_array(t.pos_s, t.s_blocks, 21, np.array([0,1])+t.h, np.array([0,1]))  
                      

def set_TPFA_Dirichlet(Dirichlet,operator,  h, boundary_array, RHS,D):

    c=0
    for i in boundary_array:
        C=(h/2)**-2
        
        operator[i,i]-=C
        RHS[i]-=C*Dirichlet

        c+=1
    return(RHS, operator)
    



def Green_2d_integral(pos_source,  function, h, Rv):
    """INPUTS:
        - pos_soure -> the position of the origin of the G-function IN RELATION TO THE CELL CENTER
        - surface -> in string form north, south, east or west
        - function -> in string form T (for the gradient) or R (for the original G-function)"""
    L=h
    h_local=h/100
    #x=np.linspace(h_local/2, L-h_local/2, 100)-L/2
    x=np.linspace(0, L, 100)-L/2
    y=x
    s_x=np.concatenate([[x],[x],[np.zeros(len(x))+L/2],[np.zeros(len(y))-L/2]], axis=0)-pos_source[0]
    s_y=np.concatenate([[np.zeros(len(x))+L/2],[np.zeros(len(y))-L/2], [y],[y]], axis=0)-pos_source[1]
    d_field=np.sqrt(s_x**2+s_y**2)
    normals_x=np.concatenate([[np.zeros(len(x))],[np.zeros(len(x))],[np.zeros(len(y))+1],[np.zeros(len(y))-1]], axis=0)
    normals_y=np.concatenate([[np.zeros(len(x))+1],[np.zeros(len(x))-1],[np.zeros(len(y))],[np.zeros(len(y))]], axis=0)
    
    cos_theta=np.zeros(d_field.shape)
    for i in range(4):
        cos_theta[i]=(s_x[i]*normals_x[i]+s_y[i]*normals_y[i])/d_field[i]
    
    if function=="R":
        f=np.log(Rv/d_field)/(2*np.pi)
    elif function=="T":
        f=cos_theta/(2*np.pi*d_field)
    else:
        print("wrong function entered")
    

    integral=np.sum(f, axis=1)*h_local
    return(integral)
    
class full_ss():
    """Class to solve the solution split problem with point sources in a 2D domain"""
    def __init__(self, pos_s, Rv, h, K_eff, D,L):          
        #x=np.linspace(-h/2, L+h/2, int(L//h)+2)
        x=np.linspace(h/2,L-h/2, int(L//h))
        y=x
        self.x, self.y=x,y
        self.xlen, self.ylen=len(x), len(y)
        self.C_0=K_eff*np.pi*Rv**2
        self.pos_s=pos_s #exact position of each source
        self.h=h
        self.n_sources=self.pos_s.shape[0]
        self.Rv=Rv
        self.h=h
        self.D=D
        self.boundary=get_boundary_vector(len(x), len(y))
        
        source_FV=np.array([], dtype=int)
        for u in self.pos_s:
            r=np.argmin(np.abs(self.x-u[0]))
            c=np.argmin(np.abs(self.y-u[1]))
            source_FV=np.append(source_FV, c*len(self.x)+r) #block where the source u is located
        self.s_blocks=source_FV
    
    def solve_problem(self,B_q):
        self.setup_problem(B_q)
        v=np.linalg.solve(self.A, self.B)
        self.phi_q=v[-len(self.pos_s):]
        print(self.phi_q)
        v_mat=v[:-len(B_q)].reshape((len(self.x), len(self.y)))
        plt.imshow(v_mat, origin='lower'); plt.colorbar(); plt.title("regular term")
        self.v=np.ndarray.flatten(v_mat)
        return(v_mat)
        
    def setup_problem(self, B_q):
        len_prob=self.xlen*self.ylen+len(self.pos_s)
        A=A_assembly(self.xlen, self.ylen)*self.D/self.h**2
        
        B=np.zeros(len_prob)
        A=np.hstack((A, np.zeros((A.shape[0], len(self.pos_s)))))
        A=np.vstack((A,np.zeros((len(self.pos_s),A.shape[1]))))
        A=self.setup_boundary_zero_Dirich(A)
        self.A=A

        B[-len(self.s_blocks):]=B_q
        A[-len(self.s_blocks),:]=0
        #pdb.set_trace()
        pos_s=np.arange(len(self.x)*len(self.y), A.shape[0])
        A[pos_s,pos_s]=1/self.C_0
        A[pos_s,self.s_blocks]=1

        
        c=0
        for i in self.pos_s:
            arr=np.delete(np.arange(len(self.pos_s)),c)
            d=0
            pos_s0=len(self.x)*len(self.y)
            for j in arr:
                self.A[ pos_s0+c,pos_s0+j] += Green(self.pos_s[j], i, self.Rv)
                d+=1
            c+=1
        
        self.B=B
        self.A=A
                
    def setup_boundary(self,A):
        for i in np.ndarray.flatten(self.boundary):
            cord=pos_to_coords(self.x, self.y, i)
            A[i,:]=0
            A[i,i]=1
            for c in range(len(self.pos_s)):
                A[i, -len(self.s_blocks)+c]=Green(cord, self.pos_s[c], self.Rv) 
        return(A)
    
    def setup_boundary_zero_Dirich(self, A):
        """Translates the zero Dirich into a Neuman BC for the SS problem"""
        for i in np.ndarray.flatten(self.boundary):
            cord=pos_to_coords(self.x, self.y, i)
            for c in range(len(self.pos_s)):
                A[i, -len(self.s_blocks)+c]-=2*Green(cord, self.pos_s[c], self.Rv)*self.D/self.h**2 
            A[i,i]-=2*self.D/self.h**2 
        return(A)
    
    def reconstruct(self, v_sol, phi_q):
        x,y=self.x, self.y
        phi=np.zeros(len(x)*len(y))
        for i in range(len(x)):
            for j in range(len(y)):
                dis_pos=j*len(x)+i
                cords=np.array([x[i], y[j]])
                g=0
                for c in range(len(self.pos_s)):
                    s=self.pos_s[c]
                    g+=Green(s,cords, self.Rv)*phi_q[c] if np.linalg.norm(s-cords) > self.Rv else 0
                    
                phi[dis_pos]=v_sol[dis_pos]+g
        self.phi=phi
        return(phi.reshape(len(y), len(x)))
    
    def reconstruct_inf(self, phi_q, ratio):
        h=self.h/ratio
        L=self.x[-1]+self.h/2
        num=int(L//h)
        h=L/num
        x=np.linspace(h/2, L-h/2, num)
        y=x
        phi=np.zeros(len(x)*len(y))
        #pdb.set_trace()
        for i in range(len(x)):
            for j in range(len(y)):
                dis_pos=j*len(x)+i
                cords=np.array([x[i], y[j]])
                g=0
                for c in range(len(self.pos_s)):
                    s=self.pos_s[c]
                    g+=Green(s,cords, self.Rv)*phi_q[c] if np.linalg.norm(s-cords) > self.Rv else 0
                    
                phi[dis_pos]=g
        self.phi_inf=phi
        return(phi.reshape(len(y), len(x)))
    
    def get_corners(self):
        corners=np.array([0,len(self.x)-1,len(self.x)*(len(self.y)-1), len(self.x)*len(self.y)-1])
        self.corners=corners
        return(corners)
                    


def get_validation(ratio, SS_ass_object, pos_s, phi_j, D, K_eff, Rv, L):
    t=SS_ass_object
    C_0=K_eff*np.pi*Rv**2
    h=t.h/ratio
    num=int(L//h)
    h=L/num
    x=np.linspace(h/2, L-h/2, num)
    y=x
    
    A=A_assembly(len(x), len(y))*D/h**2
    A_virgin=A_assembly_Dirich(len(x), len(y))*D/h**2

    #set dirichlet
    B,A=set_TPFA_Dirichlet(0,A, h, get_boundary_vector(len(x), len(y)), np.zeros(len(x)*len(y)),D)
    #Set sources
    s_blocks=np.array([], dtype=int)
    c=0
    for i in pos_s:
        x_pos=np.argmin(np.abs(i[0]-x))
        y_pos=np.argmin(np.abs(i[1]-y))
        
        block=y_pos*len(x)+x_pos
        A[block, block]-=C_0/h**2
        B[block]-=C_0/h**2*phi_j[c]
        s_blocks=np.append(s_blocks, block)
        c+=1
    sol=np.linalg.solve(A,B)
    
    q_array=-np.dot(A_virgin[s_blocks],sol)*h**2/D
    
    return(sol, len(x), len(y),q_array, B, A, s_blocks,x,y)  





