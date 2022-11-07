#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:59:14 2021

@author: pdavid
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

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

def pos_to_coords(x, y, ID):
    xpos=ID%len(x)
    ypos=ID//len(x)
    return(np.array([x[xpos], y[ypos]]))

def coord_to_pos(x,y, coord):
    pos_x=np.argmin((coord[0]-x)**2)
    pos_y=np.argmin((coord[1]-y)**2)
    return(int(pos_x+pos_y*len(x)))

def get_trans(hx, hy, pos):
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
    return(-theta[:,0]-theta[:,1])

def get_Green(hx, hy, pos_respect_cell_center, Rv):
    p=pos_respect_cell_center #position with respect to the cell's center
    n=np.array([0, hy/2])
    s=np.array([0,-hy/2])
    e=np.array([hx/2,0])
    w=np.array([-hx/2,0])
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
            
        self.cells_coup=np.unique(self.coup_cells)
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
        grad_G=get_trans(self.h, self.h, self.pos_s[i]-cell_center)
        G=get_Green(self.h, self.h, self.pos_s[i]-cell_center, self.Rv)
        #The third equation couples with the neighbours through flux 
        for pp in range(4): #Goes through each of the sides 
            if pp==0:#north
                e=len(self.x)
                ppp=1 #surface of the neighbouring sharing surface (south=1)
            if pp==1:#south
                e=-len(self.x)
                ppp=0 #surface of the neighbouring sharing surface (north=0)
            if pp==2:#east
                e=1
                ppp=3 #surface of the neighbouring sharing surface (west=3)
            if pp==3:#west
                e=-1
                ppp=2 #surface of the neighbouring sharing surface (east=2)
            pos_surf_current=pos_v[pp]
            print(pos_surf_current)
            
            neigh=cell+e #ID of the neighbouring cell (in the A matrix)
            print(neigh)
            if cell+e not in self.cells_coup:
                #neighbour is not a well block
                #Second equation, flux continuation
                self.A[neigh, neigh]+=1/self.h**2
                self.A[neigh, cell]+=1/self.h**2
                self.b[neigh, pos_surf_current]-=2/self.h**2
                self.b[neigh, i]-=grad_G[pp]/self.h**2
                
                self.c[pos_surf_current, neigh]=-1
                self.c[pos_surf_current, cell]=-2
                self.d[pos_surf_current, pos_surf_current]=3
                self.d[pos_surf_current, i]=grad_G[pp]+G[pp]
                
            else: #The neighbour is a well block
                print("neighbour called, YOU STILL NEED TO REVIEW THIS COUPLING")
            #if the neighbour is a well block the coupling is quite straight forward
                pos_surf_neigh=self.n_sources + 4*np.where(self.cells_coup==cell+e)[0][0]+ppp
                if e>0:
                    #the current cell is the lower indexed one, so here the flux continuity must be 
                    #applied
                    
                    self.d[pos_surf_current,pos_surf_current]=4
                    self.d[pos_surf_current,i]=grad_G[pp]
                    self.c[pos_surf_current,cell]=-2
                    self.c[pos_surf_current,neigh]=-2
                    self.d[pos_surf_neigh,pos_surf_current]=1
                
                if e<0:
                    #the current cell is the lower indexed one 
                    self.d[pos_surf_neigh,i]=grad_G[pp]
                    self.d[pos_surf_current,pos_surf_current]=-1
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
                pos_v=pos[1:,1] #positions of the north, south, east, west v unknowns for this cell in the c_d matrix
                for p in range(len(coeff)):
                    self.d[pos[p]]=coeff[p]
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
                
                
                
                
                
def first_equation(i, C0, coup_cells,cell_source_position,  x, y, h, n_sources, pos_s):
    """only modifies d matrix"""
    position=np.zeros((0,2), dtype=int)
    coeff=np.zeros(0)
    position=np.concatenate((position, np.array([[i,i]])))
    coeff=np.append(coeff, 1/C0)
    
    cell=coup_cells[i]
    cell_center=pos_to_coords(x, y, cell)
    v_bar=v_linear_interpolation(cell_center, pos_s[i], h)
    
    pos_v_n=n_sources+4*cell_source_position[i] #position in the c_d matrix of the north v unknown for this source
    
    
    position=np.concatenate((position, np.array([[i,pos_v_n]])))
    position=np.concatenate((position, np.array([[i,pos_v_n+1]])))
    position=np.concatenate((position, np.array([[i,pos_v_n+2]])))
    position=np.concatenate((position, np.array([[i,pos_v_n+3]])))
    coeff=np.concatenate((coeff, v_bar))
    
    return(position.astype(int) , coeff)




def set_TPFA_Dirichlet(Dirichlet,operator,  h, boundary_array, RHS):

    c=0
    for i in boundary_array:
        C=(h/2)**-2
        
        operator[i,i]-=C
        RHS[i]-=C*Dirichlet

        c+=1
    return(RHS, operator)
    
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
                print("source ", np.where(t.coup_cells==i)[0])
                
            else:
            #reconstruction via flux conservation
                self.counter_non_well+=1
                r=single_neigh_reconstruction(self.solution, i, t.h,t.coup_cells, self.h_ratio, t.x, 
                                t.y, t.pos_s, t.cell_source_position, t.Rv, t.boundary)

                
                self.eeee[y_pos[0]:y_pos[-1]+1,x_pos[0]:x_pos[-1]+1]=r
        return(self.eeee)



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
    
    
    
    
    
    
    
    
    
    
    