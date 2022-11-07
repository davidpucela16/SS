#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:59:14 2021
@author: pdavid
MAIN MODULE FOR THE SOLUTION SPLIT COUPLING MODEL IN 2D!
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import scipy as sp
from scipy import sparse
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
    ARE WE CONSIDERING THE DIFFUSION COEFFICIENT IN THE GREEN'S FUNCTION"""
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

def get_Green(hx, hy, pos_respect_cell_center, Rv):
    p=pos_respect_cell_center #position with respect to the cell's center
    n=np.array([0, hy])
    s=np.array([0,-hy])
    e=np.array([hx,0])
    w=np.array([-hx,0])
    array=np.array([Green(p, n, Rv), Green(p, s, Rv), Green(p, e, Rv), Green(p, w, Rv)])
    return(array)
    


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
def Green(q_pos, x_coord, Rv):
    """returns the value of the green function at the point x_coord, with respect to
    the source location (which is q_pos)"""
    return(np.log(Rv/np.linalg.norm(q_pos-x_coord))/(2*np.pi))

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

def get_array_it_surf(pp, x, h, cell_coords):
    if pp==0:#north
        e=len(x)
        pp_neigh=1 #surface of the neighbouring sharing surface (south=1)
        pos_surf=cell_coords+np.array([0.5,0])*h
    if pp==1:#south
        e=-len(x)
        pp_neigh=0 #surface of the neighbouring sharing surface (north=0)
        pos_surf=cell_coords+np.array([-0.5,0])*h
    if pp==2:#east
        e=1
        pp_neigh=3 #surface of the neighbouring sharing surface (west=3)
        pos_surf=cell_coords+np.array([0,0.5])*h
    if pp==3:#west
        e=-1
        pp_neigh=2 #surface of the neighbouring sharing surface (east=2)
        pos_surf=cell_coords+np.array([0,-0.5])*h
    return(e,pp_neigh, pos_surf)

class assemble_SS_2D():
    def __init__(self, pos_s, A, Rv, h,x,y, K_eff, D):
        self.x=x
        self.y=y
        self.C0=K_eff*np.pi*Rv**2
        self.pos_s=pos_s #exact position of each source
        
        self.n_sources=self.pos_s.shape[0]
        self.A=A
        self.Rv=Rv
        self.h=h
        self.D=D
        
        self.get_coup_cells()
        
        c_d_len=4*len(np.unique(self.coup_cells))+self.n_sources
        
        self.b=np.zeros((len(x)*len(y), c_d_len))
        self.c=np.zeros((c_d_len, len(x)*len(y)))
        self.d=np.zeros((c_d_len, c_d_len))
        
        self.boundary=get_boundary_vector(len(x), len(y))
        
        
    def get_coup_cells(self):
        """
        coup_cells -> the position is the ID of the source, and the value is the cell it lies on 
        cells_coup -> array with the FV cells that contain a source
        sources_list -> for each position i of the array it contains a list with the ID of the sources that
                        lie on the FV cell identified as cells_coup[i]
        cell_source_position -> each position corresponds to the ID of the source, the value returns the 
                                position of the FV cell in the cells_coup and therefore in the c_d matrix
        """
        self.coup_cells=np.array([], dtype=int)
        for i in range(self.pos_s.shape[0]):
            self.coup_cells=np.append(self.coup_cells, coord_to_pos(self.x, self.y, self.pos_s[i,:]))
        _, idx = np.unique(self.coup_cells, return_index=True)
        self.cells_coup=self.coup_cells[np.sort(idx)]
        self.sources_list=[]
        for i in self.cells_coup:
            self.sources_list.append(np.where(self.coup_cells==i)[0])
        
        self.source_list_index=np.zeros(len(self.x)*len(self.y))
        self.source_list_index[self.cells_coup]=np.arange(len(self.cells_coup))
        
        self.cell_source_position=np.zeros(len(self.coup_cells))
        for i in range(len(self.coup_cells)):
            self.cell_source_position[i]=np.where(self.cells_coup==self.coup_cells[i])[0]
            
    def second_equation(self,i, pos_v):
        #Second equation
        cell=self.coup_cells[i]
        #The second equation is independent on the neighbours
        #"mass conservation"
        
        self.A[cell, :]=0
        self.A[cell,cell]=-4
        self.b[cell, pos_v]=1
        return()
        
    def third_equation(self,i, pos_v):
        
        cell=self.coup_cells[i]
        cell_center=pos_to_coords(self.x, self.y, cell)
        T=get_trans(self.h, self.h, self.pos_s[i]-cell_center)
        R=get_Green(self.h, self.h, self.pos_s[i]-cell_center, self.Rv)
        #The third equation couples with the neighbours through flux 
        for pp in range(4): #Goes through each of the sides 
            if pp==0:#north
                e=len(self.x)
                #ppp=1 #surface of the neighbouring sharing surface (south=1)
                contact_vertices=np.array([0,1])
            if pp==1:#south
                e=-len(self.x)
                #ppp=0 #surface of the neighbouring sharing surface (north=0)
                contact_vertices=np.array([2,3])
            if pp==2:#east
                e=1
                #ppp=3 #surface of the neighbouring sharing surface (west=3)
                contact_vertices=np.array([1,2])
            if pp==3:#west
                e=-1
                #ppp=2 #surface of the neighbouring sharing surface (east=2)
                contact_vertices=np.array([0,3])
            non_contact_vertices=np.delete(np.arange(4), contact_vertices)
            
            neigh=cell+e #ID of the neighbouring cell (in the A matrix)
            print(neigh)
            if cell+e not in self.cells_coup:
                #neighbour is not a well block
                #Second equation, flux continuation
                self.A[neigh, neigh]+=1/self.h**2
                self.A[neigh, cell]-=2/(self.h**2)
                self.b[neigh, pos_v[pp]]+=1/(self.h**2)
                self.b[neigh, i]+=T[pp]/self.h**2
                
                #coupling equation
                print("pos_v[pp]= ", pos_v[pp])
                self.c[pos_v[pp], neigh]=-1
                self.d[pos_v[pp], pos_v[pp]]=2
                self.c[pos_v[pp], cell]=-1
                self.d[pos_v[pp], i]=T[pp]/2+R[pp]/self.h
                
            else: #The neighbour is a well block
                print("neighbour called, YOU STILL NEED TO REVIEW THIS COUPLING")
# =============================================================================
#             #if the neighbour is a well block the coupling is quite straight forward
#                 pos_surf_neigh=self.n_sources + 4*np.where(self.cells_coup==cell+e)[0][0]+ppp
#                 if e>0:
#                     #the current cell is the lower indexed one, so here the flux continuity must be 
#                     #applied
#                     
#                     self.d[pos_surf_current,pos_surf_current]=4
#                     self.d[pos_surf_current,i]=grad_G[pp]
#                     self.c[pos_surf_current,cell]=-2
#                     self.c[pos_surf_current,neigh]=-2
#                     self.d[pos_surf_neigh,pos_surf_current]=1
#                 
#                 if e<0:
#                     #the current cell is the lower indexed one 
#                     self.d[pos_surf_neigh,i]=grad_G[pp]
#                     self.d[pos_surf_current,pos_surf_current]=-1
# =============================================================================
        return()
            
    def assemble_SS_problem(self):
        
        for i in range(len(self.coup_cells)): #will go through each sources 
        #i= ID source
            if np.sum(self.coup_cells==self.coup_cells[i])>1:
                j=i
                print("multiple sources")
            elif np.sum(self.coup_cells==self.coup_cells[i])==1: #only one source in that cell, therefore i represents the ID of the source
                print("single source")    
                i=int(i)
                self.i=i
                pos, coeff= first_equation(i, self.C0, self.coup_cells, self.cell_source_position,self.x, self.y, self.h ,self.n_sources, self.pos_s)
                pos_v=pos[1:,1] #positions of the 0,1,2,3 v unknowns for this cell in the c_d matrix
                print("pos[:,1]= ", pos[:,1])
                self.d[i,pos[:,1]]=coeff
                self.second_equation(i, pos_v)
                self.third_equation(i, pos_v)
        return()
    
    def assemble_full_system(self,phi_source):
        self.assemble_SS_problem()
        M=np.hstack((self.A, self.b))
        M=np.vstack((M, np.hstack((self.c, self.d))))
        
        RHS=np.zeros(len(self.x)*len(self.y)+self.n_sources+4*len(self.cells_coup))
        
        Dirichlet=0
        boundary_array=get_boundary_vector(len(self.x), len(self.y))
        self.boundary_array=boundary_array
        RHS, M=set_TPFA_Dirichlet(Dirichlet,M,  self.h, boundary_array, RHS)
        
        RHS[len(self.x)*len(self.y):len(self.x)*len(self.y)+self.n_sources]=phi_source
        self.M=M
        self.RHS=RHS
        return()
                
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
        
   
# =============================================================================
#     def get_connect_matrices(self):
#         """Connectivity and identification arrays:
#                 - FV_DoF => array with the DoF for the large FV cells
#                 - s_DoF => """
#         #pos_s will dictate the ID of the sources by the order they are kept in it!
#         source_FV=np.array([]).astype(int)
#         for u in self.pos_s:
#             print(u)
#             r=np.argmin(np.abs(self.x-u[0]))
#             c=np.argmin(np.abs(self.y-u[1]))
#             source_FV=np.append(source_FV, c*len(self.x)+r)
#         
#         source_DoF=np.array([])
#         FV_DoF=list()       
#         p=0
#         for i in range(len(self.x)*len(self.y)):
#             if i in np.unique(source_FV):
#                 c=0
#                 a=np.where(source_FV==i)[0] #IDs of sources in this block 
#                 source_DoF=np.concatenate([source_DoF, np.arange(c,c+len(a))])
#                 c+=len(a) #amount of sources in the block
#             
#                 FV_DoF.append(np.arange(9)+c)
#                 c+=9
#             else:
#                 FV_DoF.append(p)
#                 p+=1
#                 
#         self.FV_DoF=FV_DoF
#         self.s_DoF=source_DoF
#         self.s_blocks=source_FV  #this list will have the same length as s_DoF
# =============================================================================

    def pos_arrays(self):
        #pos_s will dictate the ID of the sources by the order they are kept in it!
        source_FV=np.array([]).astype(int)
        for u in self.pos_s:
            print(u)
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
            unk_pos=np.array([[-self.h/2, self.h],[self.h/2, self.h],[self.h, self.h/2],[self.h,
                        -self.h/2],[self.h/2, -self.h],[-self.h/2, -self.h],[-self.h, -self.h/2],
                             [-self.h, self.h/2]])
        elif typ=="real":
            unk_pos=np.array([[0,self.h],[self.h,0],[0,-self.h],[-self.h,0]])

        
        ret_array=np.zeros((len(block_s),unk_pos.shape[0]))
        c=0 #The counter c marks the position within the array of each DoF. It begins with north 
            #or northwest and continues in the same sense as the clock
        for i in unk_pos: #goes through each of the positions needed
            G=0
            for j in range(len(block_s)): #goes through each of the sources in this block
                ret_array[j,c]=Green(self.pos_s[j]-pos_block_cent,i,self.Rv)
            c+=1
        return(ret_array)
        
    def assembly_sol_split_problem(self):
        #First of all it is needed to remove the source_blocks from the main matrix
        for i in np.unique(self.s_blocks):
            neigh=i+np.array([len(self.x), -len(self.x), 1,-1])
            self.A[neigh,i]=0
            self.A[neigh, neigh]+=1
        
        self.b_matrix()
        self.c_matrix()
        self.de_matrix()
        self.f_matrix()
        self.g=np.zeros((len(self.s_blocks), len(self.FV_DoF)))
        self.H_matrix()
        self.I_matrix(self.C_0)
        
        Up=np.hstack((self.A, self.b, self.c))
        Mid=np.hstack((self.d, self.e, self.f))
        Down=np.hstack((self.g, self.H, self.I))
        
        M=np.vstack((Up, Mid, Down))
        self.M=M
        return(M)
        
    def de_matrix(self):
        """Will assemble both d and e matrices """

        self.d=np.zeros((9*self.total_sb, len(self.FV_DoF)))
        self.e=np.zeros((9*self.total_sb, 9*self.total_sb))
        self.FD_scheme_DE()
        for i in range(len(np.unique(self.s_blocks))):
            ID=np.unique(self.s_blocks)[i]
            self.d[self.row_d+i*9, self.col_d+ID]=self.dat_d
            self.e[self.row_e+i*9, self.col_e+i*9]=self.dat_e
        
    def FD_scheme_DE(self):
        """This function will code finite difference scheme for the local problem
        There is interpolation for the finite volume cells around the given block"""
        lx=len(self.x)
        neighbours=np.array([lx-1, lx, lx+1, -1,1,-lx-1,-lx,-lx+1])
        self.row_d=np.array([0,0,0,1,2,2,2,3,5,6,6,6,7,8,8,8])
        self.col_d=neighbours[np.array([0,1,3,1,1,2,4,3,4,3,5,6,6,4,6,7])]       
        self.dat_d=np.array([2,1,1,2,1,2,1,2,2,1,2,1,2,1,1,2])/2
        
        self.row_e=np.array([0,0,0,1,1,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,6,6,6,7,7,7,7,8,8,8])
        self.col_e=np.array([0,1,3,0,1,2,4,1,2,5,0,3,4,6,1,3,4,5,7,2,4,5,8,3,6,7,4,6,7,8,5,7,8])
        self.dat_e=np.array([-4,1,1,1,-4,1,1,1,-4,1,1,-4,1,1,1,1,-4,1,1,1,1,-4,1,1,-4,1,1,1,-4,1,1,1,-4])
        
    
    def f_matrix(self):
        """self sufficient function to assemble the f matrix"""
        f=np.zeros((9*len(np.unique(self.s_blocks)),len(self.s_blocks)))
        for i in np.unique(self.s_blocks):
            pos_v0=np.where(np.unique(self.s_blocks)==i)[0][0]*9
            sources=np.where(self.s_blocks==i)[0]
            Sr=self.get_singular_term(i, "real")
            Sp=self.get_singular_term(i, "phantom")
            c=np.where(self.s_blocks==i)[0]
            
            a=np.array([Sp[:,-1]+Sp[:,0], Sr[:,0], Sp[:,1]+Sp[:,2], Sr[:,-1], 
                                            np.zeros(len(sources)),Sr[:,1], Sp[:,6]+Sp[:,5], Sr[:,2],
                                            Sp[:,3]+Sp[:,4]])
            f[pos_v0:pos_v0+9,c]=-a
            self.f=f
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
                
                self.b[b_row, b_col]=b_data
                
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
        
        
            
                
                
                
def first_equation(i, C0, coup_cells,cell_source_position,  x, y, h, n_sources, pos_s):
    """only modifies d matrix"""
    position=np.zeros((0,2), dtype=int)
    coeff=np.zeros(0)
    position=np.concatenate((position, np.array([[i,i]])))
    coeff=np.append(coeff, 1/C0)
    
    cell=coup_cells[i]
    cell_center=pos_to_coords(x, y, cell)
    #v_bar=v_linear_element(cell_center, pos_s[i], h)
    v_bar=v_linear_interpolation(cell_center, pos_s[i], 2*h)
    pos_v_n=n_sources+4*cell_source_position[i] #position in the c_d matrix of the north v unknown for this source
    
    
    position=np.concatenate((position, np.array([[i,pos_v_n]])))
    position=np.concatenate((position, np.array([[i,pos_v_n+1]])))
    position=np.concatenate((position, np.array([[i,pos_v_n+2]])))
    position=np.concatenate((position, np.array([[i,pos_v_n+3]])))
    coeff=np.concatenate((coeff, v_bar))
    
    return(position.astype(int) , coeff)




def set_TPFA_Dirichlet(Dirichlet,operator,  h, boundary_array, RHS,D):

    c=0
    for i in boundary_array:
        C=(h/2)**-2
        
        operator[i,i]-=C
        RHS[i]-=C*Dirichlet

        c+=1
    return(RHS, operator)
    



def single_cell_reconstruction(pos_source, source_strength, v_value, h_ratio,h_original, Rv):
    """everything in function of cell center! therefore cell_center_local=(0,0)"""
    L=h_original
    h=L/h_ratio
    x_local=np.linspace(h/2, L-h/2, h_ratio)-L/2
    y_local=x_local
    rec_field=np.zeros((len(y_local), len(x_local)))
    for i in range(len(y_local)):
        for j in range(len(x_local)):
            coeff_lin=v_linear_interpolation(np.array([0,0]), pos_source, h)

            if np.sum(pos_source.shape)>3:
                print("still to write multiple sources per cell")
            else:
                G=Green(pos_source, np.array([x_local[j], y_local[i]]), Rv)
            rec_field[i,j]=G*source_strength+coeff_lin.dot(v_value)
    return(rec_field)
    

def single_neigh_reconstruction(solution, cell, h_original,coup_cells, h_ratio, old_x, 
                                old_y, pos_s, cell_source_position, Rv, boundary):
    x=old_x
    y=old_y
    L=h_original
    h=L/h_ratio
    x_local=np.linspace(h/2, L-h/2, h_ratio)-L/2
    y_local=x_local
    rec_field=np.zeros((len(y_local), len(x_local)))
    
    #first get surface values 
    phi_surf=get_cell_surface_values(old_x, old_y, cell, solution,h_original, 
                                     coup_cells,cell_source_position, pos_s, Rv, boundary)  
    #then interpolate
    for i in range(len(y_local)):
        for j in range(len(x_local)):
            coeffs=v_linear_interpolation(np.array([0,0]), np.array([x_local[j], y_local[i]]), h_original)
            print(coeffs)
            print(phi_surf)
            rec_field[i,j]=coeffs.dot(phi_surf)
    return(rec_field)
    


def get_cell_surface_values(x, y, cell_ID, solution,h, coup_cells,cell_source_position, pos_s, Rv, boundary):
    phi_surf=np.zeros(4)
    cell_coords=pos_to_coords(x, y, cell_ID)
    for pp in range(4): #Goes through each of the sides 
        e, ppp, pos_surf=get_array_it_surf(pp, x, h, cell_coords)
        neigh=cell_ID+e
        b=boundary[pp,:]
        if neigh in coup_cells:
            #the neighbour is a well block
            print("reconstruction is not prepared for multiple sources")
            source=np.where(coup_cells==neigh)[0][0]
            phi_s=Green(pos_s[source], pos_surf, Rv)*solution[len(x)*len(y)+source]
            
            pos_v_n=int(len(coup_cells)+4*cell_source_position[source])
            v_surf=solution[len(x)*len(y)+pos_v_n+ppp]
            
            phi_surf[pp]=phi_s+v_surf
            print("neigh_cell={}, surf={}, value={}".format(neigh, ppp, phi_s+v_surf))
        elif cell_ID in b:
            print("0 Dirichlet!!!!")
            phi_surf[pp]=0
        else:
            #the neighbour is a normal block 
            print(solution[neigh])
            print(neigh)
            phi_surf[pp]=(solution[cell_ID]+solution[neigh])/2
        
    return(phi_surf)
    
    
    
    
    
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
    
class reconstruction_microscopic_field():
    def __init__(self, assembly_SS_object,solution, h_ratio):
        self.t=assembly_SS_object
        self.solution=solution
        self.coup_cells=self.t.coup_cells
        self.h_ratio=h_ratio
        self.h=assembly_SS_object.h/h_ratio
        L_old=self.t.x[-1]-self.t.x[0]+self.t.h
        self.x=np.linspace(self.h/2, self.t.x[-1]+self.t.h/2-self.h/2,int(L_old/self.h))
        self.y=self.x
        
        self.q=solution[len(self.t.x)*len(self.t.y)+np.arange(len(self.t.coup_cells))]
        self.solution_FV=solution[:len(self.t.x)*len(self.t.y)]
        self.sol_SS=solution[len(self.t.x)*len(self.t.y):]
        self.eeee=np.zeros((len(self.y), len(self.x)))
    

        
    def reconstruction(self):
        self.counter_non_well=0
        t=self.t
        h_ratio=self.h_ratio
        for i in range(len(self.solution_FV)):
            self.i=i
            coords=pos_to_coords(t.x, t.y, i)
            x_pos=np.where((self.x<coords[0]+t.h/2) & (self.x>coords[0]-t.h/2))[0]
            y_pos=np.where((self.y<coords[1]+t.h/2) & (self.y>coords[1]-t.h/2))[0]
            if i in t.coup_cells:
            #reconstruction via solution splitting:
                qi=np.where(t.cells_coup==i)[0]
                pos_v_n=int(t.n_sources+4*t.cell_source_position[qi])
                pos_v=np.arange(4)+pos_v_n
                
                r=single_cell_reconstruction(t.pos_s[qi]-coords, self.q[qi], self.sol_SS[pos_v], h_ratio, t.h, t.Rv)
                self.eeee[y_pos[0]:y_pos[-1]+1,x_pos[0]:x_pos[-1]+1]=r
                #print("source ", np.where(t.coup_cells==i)[0])
                
            else:
            #reconstruction via flux conservation
                self.counter_non_well+=1
                r=single_neigh_reconstruction(self.solution, i, t.h,t.coup_cells, self.h_ratio, t.x, 
                                t.y, t.pos_s, t.cell_source_position, t.Rv, t.boundary)

                
                self.eeee[y_pos[0]:y_pos[-1]+1,x_pos[0]:x_pos[-1]+1]=r
        return(self.eeee)
    
    
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
        

        B[-len(self.s_blocks):]=B_q
        A[-len(self.s_blocks),:]=0
        A[-len(self.s_blocks):,-len(self.s_blocks):]=1/self.C_0
        A[-len(self.s_blocks):,self.s_blocks]=1
        
        
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
        pdb.set_trace()
        for i in range(len(x)):
            for j in range(len(y)):
                dis_pos=j*len(x)+i
                cords=np.array([x[i], y[j]])
                g=0
                for c in range(len(self.pos_s)):
                    s=self.pos_s[c]
                    g+=Green(s,cords, self.Rv)*phi_q[c] if np.linalg.norm(s-cords) > self.Rv else 0
                    
                phi[dis_pos]=g+phi_bar
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
