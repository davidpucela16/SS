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
import matplotlib.gridspec as gridspec

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
        x=np.linspace(h/2,L-h/2, int(L/h))
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
        num=int(L/h)
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
    num=int(L/h)
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

def get_cord(p, C_x, C_y):
    a,b=p%len(C_x),p//len(C_x)
    return((C_x[a], C_y[b]))


def set_boundary(B, north, south, east, west, C_x, C_y, type_boundary, q, pq, h, D):
    """Secuentially calls the get_b_value function in order to assemble the boundary
    conditions for the infinite domain solution
    
    As well here, we can choose the Neuman or Dirichlet BC"""
    for i in east:
        b=get_cord(i, C_x, C_y)
        B[i]+=get_b_value(h,q, D, pq, b, "east", type_boundary)
        
    for i in west:
        b=get_cord(i, C_x, C_y)
        B[i]+=get_b_value(h,q, D, pq, b,"west", type_boundary)
        
    for i in north:
        b=get_cord(i, C_x, C_y)
        B[i]+=get_b_value(h,q, D, pq, b,"north", type_boundary)
    
    for i in south:
        b=get_cord(i, C_x, C_y)
        B[i]+=get_b_value(h,q, D, pq, b,"south", type_boundary)
    
    return(B)

def get_b_value(h,q_value,D, pq, b, side, string):
    """Sets up the boundary values for the infinite domain solution by setting the out flux from the domain 
    of the boundary Pressure"""
    d=np.linalg.norm(np.array(b)-np.array(pq))
    if string=="Flux":
        return(h*q_value*costheta(pq,b,side)/(2*np.pi*d*D))
        
    elif string=="Pressure":
        return(-q_value*np.log(1/d)/(2*np.pi*D))
    
def costheta(q, b,side):
    qx,qy=q
    bx,by=b
    v=np.array([bx-qx,by-qy])
    print(v)
    if side=="north":
        normal=np.array([0,1])
    elif side=="west":
        normal=np.array([-1,0])
    elif side=="east":
        normal=np.array([1,0])
    elif side=="south":
        normal=np.array([0,-1])
    costheta=np.dot(normal,v)/np.linalg.norm(v)
    return(costheta)

def get_L2(validation, phi):
    L2=np.sqrt(np.sum((validation-phi)**2))
    return(L2)