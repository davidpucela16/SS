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


def assemble_source_trans(source_ID, coup_cell_ID, hx, hy, x, y, source_pos):
    xlen ,ylen=len(x), len(y)
    
    coords_coup_cell=pos_to_coords(x,y,coup_cell_ID)
    pos=source_pos-coords_coup_cell #position relative to the center of the cell
    theta=get_trans(hx, hy, pos)
    north=np.sum(theta[0,:])/(2*np.pi)
    south=np.sum(theta[1,:])/(2*np.pi)
    east=np.sum(theta[2,:])/(2*np.pi)
    west=np.sum(theta[3,:])/(2*np.pi)
    
    total=xlen*ylen
    ID=[source_ID+total] #since the sources are after the tissue in the matrix assembly    
    coeffs=[1] #so far with no coupling.....
    
    r=coup_cell_ID
    ID=ID+[r+xlen, r-xlen,r+1, r-1]
    coeffs=coeffs+[north, south, east, west]

    return(ID, coeffs)

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

@njit
def v_linear_element(cell_center, x_pos, h):
    """CPS4R linear element"""
    N=np.zeros(4)
    pos=x_pos-cell_center
    ep=2*pos[0]/h
    nu=2*pos[1]/h
    N[0]=0.25*(1-ep)*(1+nu)
    N[1]=0.25*(1+ep)*(1+nu)
    N[2]=0.25*(1+ep)*(1-nu)
    N[3]=0.25*(1-ep)*(1-nu)
    return(N)

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

def get_v_neigh_norm_FVneigh(j, xlen):
    """j is the DoF of the v term
    In the normal array, the first position is always the direct neighbourg"""
    if j==0:
        v_neigh=np.array([1,3])
        FV_neigh=np.array([[xlen,xlen-1], [-1,xlen-1]])
        n=np.array([[0,1],[-1,0]])
    elif j==2:
        v_neigh=np.array([1,5])
        FV_neigh=np.array([[xlen,xlen+1], [1,xlen+1]])
        n=np.array([[0,1],[1,0]])
    elif j==6:
        v_neigh=np.array([3,7])
        FV_neigh=np.array([[-xlen,-xlen-1], [-1,-xlen-1]])
        n=np.array([[0,-1],[-1,0]])
    elif j==8:
        v_neigh=np.array([5,7])
        FV_neigh=np.array([[-xlen,-xlen+1],[1,-xlen+1]])
        n=np.array([[0,-1],[1,0]])
    elif j==1:
        v_neigh=np.array([0,2,4])
        FV_neigh=np.array([xlen])
        n=np.array([0,1])
    elif j==3:
        v_neigh=np.array([0,4,6])
        FV_neigh=np.array([-1])
        n=np.array([-1,0])
    elif j==5:
        v_neigh=np.array([2,4,8])
        FV_neigh=np.array([1])
        n=np.array([1,0])
    elif j==7:
        v_neigh=np.array([4,6,8])
        FV_neigh=np.array([-xlen])
        n=np.array([0,-1])
    elif j==4:
        v_neigh=np.array([1,3,5,7])
        FV_neigh=np.array(())
        n=np.array(())
    return(v_neigh, FV_neigh, n)
    

@njit
def Green(q_pos, x_coord, Rv):
    """returns the value of the green function at the point x_coord, with respect to
    the source location (which is q_pos)"""
    return(np.log(Rv/np.linalg.norm(q_pos-x_coord))/(2*np.pi))

def Green_array(pos_s, s_blocks, block_ID, pos_calcul,Rv):
    s_IDs=np.where(s_blocks==block_ID)[0]
    green=np.zeros(pos_s.shape[0])
    for i in s_IDs:
        green[i]=Green(pos_s[i], pos_calcul, Rv)
    return(green)

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
                
class assemble_SS_2D_FD():
    def __init__(self, pos_s, A, Rv, h,x,y, K_eff, D):          
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
        

    def pos_arrays(self):
        #pos_s will dictate the ID of the sources by the order they are kept in it!
        source_FV=np.array([]).astype(int)
        for u in self.pos_s:
            r=np.argmin(np.abs(self.x-u[0]))
            c=np.argmin(np.abs(self.y-u[1]))
            source_FV=np.append(source_FV, c*len(self.x)+r) #block where the source u is located
        
        source_DoF=np.array([])
        self.FV_DoF=np.arange(len(self.x)*len(self.y))
        self.s_blocks=source_FV #for each entry it shows the block of that source
        
        total_sb=len(np.unique(self.s_blocks)) #total amount of source blocks
        self.total_sb=total_sb
        
    def get_singular_term(self, block_ID, typ):
        """Returns the arrays of singular terms in the real neighbours or the phantom ones
        TEST AGAIN"""
        pos_block_cent=np.array([self.x[block_ID%len(self.x)], self.y[block_ID//len(self.x)]])
        
        block_s=np.where(self.s_blocks==block_ID)[0] #IDs of sources in this block(always an array)
        if typ=="phantom":
            unk_pos=np.array([[-self.h, self.h/2],[-self.h/2, self.h],[self.h/2, self.h],[self.h, self.h/2],
                              [self.h,-self.h/2],[self.h/2, -self.h],[-self.h/2, -self.h],[-self.h, -self.h/2]])
        elif typ=="real":
            unk_pos=np.array([[0,self.h],[-self.h,0],[self.h,0],[0,-self.h]])
        
        
        ret_array=np.zeros((unk_pos.shape[0],len(block_s)))
        c=0 #The counter c marks the position within the array of each DoF. It begins with north 
            #or northwest and continues in the same sense as the clock
        for i in unk_pos: #goes through each of the positions needed
            G=0
            for j in range(len(block_s)): #goes through each of the sources in this block
                ret_array[c,j]=Green(self.pos_s[j]-pos_block_cent,i,self.Rv)
            c+=1
            
        return(ret_array)
    
    def bord_coupling(self, k_block, l_block, normal_k_l, v_DoF, *corner):
        """k is the original block that is being coupled to the exterior
           l is the neighbour (current block we are working on)
           normal_k_l is the outside pointing normal for the block k
           
           returns an array of two DoF, the first one is the matching DoF with 
           the neighbour and the second one is the closest one following the normal
           
           So to obtain the gradient of the regular term pointing outwards from the given 
           v_DoF (which is the original block): (toret[1]-toret[0])/(h/2)"""
        normal=normal_k_l
        pos_v0_l=np.where(np.unique(self.s_blocks)==l_block)[0][0]*9
        if (normal==np.array([0,1])).all():
            if v_DoF==0:
                v=np.array([6,3])
                if corner[0]: v+=2
            if v_DoF==1:
                v=np.array([7,4])
            if v_DoF==2:
                v=np.array([8,5])
                if corner[0]: v-=2
        if (normal==np.array([0,-1])).all():
            if v_DoF==6:
                v=np.array([0,3])
                if corner[0]: v+=2
            if v_DoF==7:
                v=np.array([1,4])
            if v_DoF==8:
                v=np.array([2,5])
                if corner[0]: v-=2
        if (normal==np.array([1,0])).all():
            if v_DoF==2:
                v=np.array([0,1])
                if corner[0]: v+=6
            if v_DoF==5:
                v=np.array([3,4])
            if v_DoF==8:
                v=np.array([6,7])
                if corner[0]: v-=6
        if (normal==np.array([-1,0])).all():
            if v_DoF==0:
                v=np.array([2,1])
                if corner[0]: v+=6
            if v_DoF==3:
                v=np.array([5,4])
            if v_DoF==6:
                v=np.array([8,7])
                if corner[0]: v-=6
        toret=v+pos_v0_l
        return(toret)
    
    def get_pos_DoF(self, block_ID):
        """returns the absolute position of the DoFs"""
        p=np.array([])
        abs_pos_v=np.array([[-1,1],[0,1],[1,1],[-1,0],[0,0],[1,0],[-1,-1],[0,-1],[1,-1]])*self.h/2
        abs_pos_v+=pos_to_coords(self.x, self.y, block_ID)
        return(abs_pos_v)          
               
        
        
    def assembly_sol_split_problem(self):
        #First of all it is needed to remove the source_blocks from the main matrix
        for i in np.unique(self.s_blocks):
            neigh=i+np.array([len(self.x), -len(self.x), 1,-1])
            self.A[neigh,i]=0
            self.A[neigh, neigh]+=1
        
        self.b_matrix()
        self.c_matrix()
        
        
        self.DEF_matrix()
        self.g=np.zeros((len(self.s_blocks), len(self.FV_DoF)))
        self.H_matrix()
        self.I_matrix(self.C_0)
        
        Up=np.hstack((self.A, self.b, self.c))
        Mid=np.hstack((self.D_matrix, self.E_matrix, self.F_matrix))
        Down=np.hstack((self.g, self.H, self.I))
        
        self.Up=Up
        self.Mid=Mid
        self.Down=Down
        
        M=np.vstack((Up, Mid, Down))
        self.M=M
        return(M)
        
    def DEF_matrix(self):
        """This function here assembles the FD scheme"""
        xlen=len(self.x)
        self.E_matrix=np.zeros((9*len(np.unique(self.s_blocks)), 9*len(np.unique(self.s_blocks))))
        self.D_matrix=np.zeros((9*len(np.unique(self.s_blocks)), len(self.x)*len(self.y)))
        self.F_matrix=np.zeros((9*len(np.unique(self.s_blocks)),len(self.s_blocks)))
        for i in np.unique(self.s_blocks): #Goes through each of the source blocks
     
            self.i=i
            pos_i=pos_to_coords(self.x, self.y, i) #absolute position of the current source block
            
            pos_v0=np.where(np.unique(self.s_blocks)==i)[0][0]*9 #initial array position of the DoFs
            
            corners=np.array([0,2,8,6])  #the otrder is like that to correspond with the phantom_green function
            abs_pos_v=np.array([[-1,1],[0,1],[1,1],[-1,0],[0,0],[1,0],[-1,-1],[0,-1],[1,-1]])*self.h/2+pos_to_coords(self.x, self.y, i)
            non_corners=np.array([1,3,5,7])
            
            #First, assembly of the corners DoFs:
            #pdb.set_trace()
            for j in corners:
                
                v_neigh, FV_neigh, n=get_v_neigh_norm_FVneigh(j, xlen)
                #In FV_neigh, the first one is always the direct neighbourg
                FV_neigh+=i
                v_neigh+=pos_v0
                #FD scheme with the inside
                self.E_matrix[pos_v0+j, v_neigh]+=1/(self.h/2)
                self.E_matrix[pos_v0+j, pos_v0+j]=-len(v_neigh)/(self.h/2)
                for m in FV_neigh: #code for the coupling with the outside
                    ind=np.in1d(m,self.s_blocks)
                    pos_direct_neigh=pos_to_coords(self.x, self.y, m[0])
                    normal_k=uni_vector(pos_i, pos_direct_neigh)
                    
                    if np.sum(ind)==0: #none are sources -> phantom 
                        pos_calcul=(pos_to_coords(self.x, self.y, m[0])+pos_to_coords(self.x, self.y, m[1]))/2
                        sing_phantom=Green_array(self.pos_s, self.s_blocks, i,pos_calcul,self.Rv)
                        self.D_matrix[pos_v0+j, m]+=1/self.h
                        self.F_matrix[pos_v0+j,:]-=sing_phantom/(self.h/2)
                        self.E_matrix[pos_v0+j, pos_v0+j]-=2/self.h                
                    else: #There are sources
                        #pdb.set_trace()
                        if np.sum(ind)==1:
                            l=m[ind] #the coupling is done with the source block to 
                                    # gain accuracy
                        elif np.sum(ind)==2: #both are sources 
                            l=m[0] #-> I take the direct neigh
                            
                        
                        print("normal_k",normal_k)
                        print("j",j)
                        print("l", l)
                        corner=False if l==m[0] else True 
                        v_neigh_l=self.bord_coupling(i, l, normal_k, j,corner) #absolute IDs of the neighbour for the gradient calculation
                        #in v_neigh the first one is always the shared one
                        
                        f_array=-block_grad_Green_norm_array(self.pos_s,
                                self.s_blocks, i, abs_pos_v[j], normal_k)
                        f_array-=block_grad_Green_norm_array(self.pos_s,
                                self.s_blocks, l, abs_pos_v[j], normal_k)
                        
                        self.E_matrix[pos_v0+j,v_neigh_l]=np.array([-1, 1])/(self.h/2)
                        self.F_matrix[pos_v0+j,:]+=f_array
            #Then assemble of the non-corners
            #pdb.set_trace()
            for j in non_corners:
                v_neigh, FV_neigh, n=get_v_neigh_norm_FVneigh(j, xlen)
                FV_neigh+=i
                v_neigh=v_neigh+pos_v0
                #FD scheme
                self.E_matrix[pos_v0+j, v_neigh]+=1/(self.h/2)
                self.E_matrix[pos_v0+j, pos_v0+j]=-len(v_neigh)/(self.h/2)
                
                if FV_neigh not in self.s_blocks: #FV block
                
                    m=FV_neigh[0]
                    print("m")
                    print(m)
                    pos_calcul=pos_to_coords(self.x, self.y, m)
                    print("pos_calcul", pos_calcul)
                    sing_phantom=Green_array(self.pos_s, self.s_blocks, i,pos_calcul,self.Rv)
                    self.D_matrix[pos_v0+j, m]+=2/self.h
                    self.F_matrix[pos_v0+j,:]-=sing_phantom/(self.h/2)
                    self.E_matrix[pos_v0+j, pos_v0+j]-=2/self.h 
                    print("sing_phantom/(self.h/2)", sing_phantom/(self.h/2))
                else: 
                    #pdb.set_trace()
                    #normal_k=uni_vector(pos_i, pos_direct_neigh)
                    normal_k=n
                    l=FV_neigh[0] #l represents the neighbourg
                    v_neigh_l=self.bord_coupling(i, l, n, j) #absolute IDs of the neighbour for the gradient calculation
                    print("v_neigh_l",v_neigh_l)
                    print("pos_v0 neigh", np.where(np.unique(self.s_blocks)==i)*9)
                    
                    
                    pos_direct_neigh=pos_to_coords(self.x, self.y, l)
                    
                    
                    f_array=-block_grad_Green_norm_array(self.pos_s,
                    self.s_blocks, i, abs_pos_v[j], normal_k)
                    f_array-=block_grad_Green_norm_array(self.pos_s,
                    self.s_blocks, l, abs_pos_v[j], normal_k)
                    
                    print("E_matrix before", self.E_matrix[pos_v0+j,v_neigh_l])
                    self.E_matrix[pos_v0+j,v_neigh_l]=np.array([-1 ,1])/(self.h/2)
                    self.F_matrix[pos_v0+j,:]+=f_array
                    print("E_matrix after", self.E_matrix[pos_v0+j,v_neigh_l])
                
                    
            #remember the 4th DoF
            self.E_matrix[pos_v0+4,pos_v0+4]=-4/(self.h/2)
            self.E_matrix[pos_v0+4, pos_v0+np.array([1,3,5,7])]=1/(self.h/2)



    def b_matrix(self):
        
        self.b=np.zeros((len(self.FV_DoF), 9*len(np.unique(self.s_blocks))))
        c=0
        b_data=-self.D/(2*self.h**2)*np.array([1,2,1,-1,-2,-1])
        for j in np.unique(self.s_blocks):
            dof_0=9*c #position in the rows of the zeroth DoF of the v term in the b matrix
            for i in range(4):
                if i==0: #north
                    b_col=np.array([0,1,2,3,4,5], dtype=int)
                    neigh=j+len(self.x)
                elif i==1: #south
                    b_col=np.array([6,7,8,3,4,5], dtype=int)
                    neigh=j-len(self.x)
                elif i==2: #east
                    b_col=np.array([2,5,8,1,4,7], dtype=int)
                    neigh=j+1
                elif i==3: #west
                    b_col=np.array([0,3,6,1,4,7], dtype=int)
                    neigh=j-1                    
                b_row=np.zeros(6, dtype=int)+neigh
                
                self.b[b_row, b_col+dof_0]=b_data
                
            c+=1
            
        return(self.b)
            
    def c_matrix(self):
        self.c=np.zeros((len(self.FV_DoF), len(self.pos_s)))
        c=0
        for i in self.pos_s:
            block_center=np.array([self.x[self.s_blocks[c]%len(self.x)],
                                   self.y[self.s_blocks[c]//len(self.x)]])
            T=get_trans(self.h, self.h, i-block_center)                 
            neigh=np.array([len(self.x),-len(self.x), 1,-1])+self.s_blocks[c]
            self.c[neigh, c]=-T*self.D/self.h**2
            
            self.c[self.s_blocks[c], c]=self.D/self.h**2
            
            c+=1
            
    def I_matrix(self, C_0):
        self.I=np.zeros((len(self.s_blocks), len(self.s_blocks)))
        for j in np.unique(self.s_blocks): #goes through each of the source blocks 
            a=np.where(self.s_blocks==j)[0]
            self.a=a
            if len(a)==1:
                self.I[a[0],:]=0
                self.I[a[0],a[0]]=1/C_0
            else:
                #pdb.set_trace()
                for k in a: #goes through each of the sources inside the block
                    b=np.delete(a,np.where(a==k)[0]) #b is the array with the ID of the other sources in the block
                    self.I[k,k]=1/C_0
                    for l in b:
                        self.I[k,l]=Green(self.pos_s[l], self.pos_s[k], self.Rv)
    def H_matrix(self):
        self.H=np.zeros((len(self.s_blocks), 9*len(np.unique(self.s_blocks))))
        for i in range(len(self.s_blocks)): #goes through each source
            rel_pos=self.pos_s[i]-pos_to_coords(self.x, self.y, self.s_blocks[i])
            p,r=FD_linear_interp(rel_pos, self.h)
            
            pos_v0=9*np.where(np.unique(self.s_blocks)==self.s_blocks[i])[0][0]
            
            p+=pos_v0
            self.H[i,p]=r
    def get_corners(self):
        corners=np.array([0,len(self.x)-1,len(self.x)*(len(self.y)-1), len(self.x)*len(self.y)-1])
        self.corners=corners
        return(corners)
        
        
            
                
                


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
        x=np.linspace(h/2, L-h/2, int(L//h))
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
        phi=np.linalg.solve(self.A, self.B)
        self.phi_q=phi[-len(self.pos_s):]
        print(self.phi_q)
        phi_mat=phi[:-len(B_q)].reshape((len(self.x), len(self.y)))
        plt.imshow(phi_mat, origin='lower'); plt.colorbar()
        self.phi=np.ndarray.flatten(phi_mat)
        return(phi_mat)
        
    def setup_problem(self, B_q):
        len_prob=self.xlen*self.ylen+len(self.pos_s)
        A=A_assembly_Dirich(self.xlen, self.ylen)*self.D/self.h**2
        
        B=np.zeros(len_prob)
        A=np.hstack((A, np.zeros((A.shape[0], len(self.pos_s)))))
        A=np.vstack((A,np.zeros((len(self.pos_s),A.shape[1]))))
        
        A=self.setup_boundary(A)
        self.A=A
        #pdb.set_trace()

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
    
    def reconstruct_inf(self, phi_bar, phi_q):
        x,y=self.x, self.y
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
                    
                phi[dis_pos]=g+np.sum(phi_bar)
        self.phi_inf=phi
        return(phi.reshape(len(y), len(x)))
                    


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





