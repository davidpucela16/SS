#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:15:08 2021

@author: pdavid
"""

import numpy as np
import matplotlib.pyplot as plt
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
    


def get_cord(p, C_x, C_y):
    a,b=p%len(C_x),p//len(C_x)
    return((C_x[a], C_y[b]))



def A_Dirichlet(A, boundary):
    A[boundary, :]=0
    A[boundary, boundary]=1
    return(A)

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

def get_b_value(h,q_value,D, pq, b, side, string):
    """Sets up the boundary values for the infinite domain solution by setting the out flux from the domain 
    of the boundary Pressure"""
    d=np.linalg.norm(np.array(b)-np.array(pq))
    if string=="Flux":
        return(h*q_value*costheta(pq,b,side)/(2*np.pi*d*D))
        
    elif string=="Pressure":
        return(-q_value*np.log(1/d)/(2*np.pi*D))

def set_boundary(B, north, south, east, west, C_x, C_y, type_boundary, q, pq):
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

#1- Set up the domain and the geometry arrays
L=2
h=0.05

x=np.arange(0,L,h) #Cell faces

C_x=np.zeros(len(x)-1) #Cell centeres
for i in range(len(C_x)):
    C_x[i]=(x[i]+x[i+1])/2
    
y=x
C_y=C_x

cylen,cxlen=len(C_y), len(C_x)

X,Y=np.meshgrid(x,y)
C_X,C_Y=np.meshgrid(C_x,C_y)


#2- array of unknowns
B=np.zeros(cxlen*cylen)

#3- Set up the boundary arrays
north=np.arange(cxlen* (cylen-1), cxlen* cylen)
south=np.arange(cxlen)
west=np.arange(0,cxlen* cylen, cxlen)
east=np.arange(cxlen-1, cxlen* cylen,cxlen)
boundary=np.concatenate([north, south, east, west])



A=A_assembly(cxlen, cylen)

B=np.zeros(cylen*cxlen)
pos_1=20*cxlen+20
#pos_2=20*cxlen+cxlen-30

R=0.01
#Set up of the sources
q=0.2*np.pi
D=1

boundary_type="Flux"
pq1=get_cord(pos_1, C_x, C_y)
B=set_boundary(B, north, south, east, west, C_x, C_y, boundary_type, q , pq1)

#pq2=get_cord(pos_2, C_x, C_y)
#B=set_boundary(B, north, south, east, west, C_x, C_y, boundary_type, q , pq2)

B[pos_1]=-q/D
#B[pos_2]=-q/D


phi_bar=1

if boundary_type=="Flux":
    #Because we solve the system in a weak sense it is important to give at least one 
    #value to the pressure to ground the solution
    
    #Let's set it through phi_bar and Peaceman coupling
    
    A[pos_1,:]=0
    A[pos_1, pos_1]=1
    B[pos_1]=phi_bar-q/(np.pi*2)*np.log(0.2*h/R)
if boundary_type=="Pressure":
    A=A_Dirichlet(A,boundary)




    
phi=np.linalg.solve(A,B).reshape(cxlen, cylen)
phin=np.linalg.solve(A,B)
plt.contourf(np.linalg.solve(A,B).reshape(cxlen, cylen))
plt.colorbar()
plt.show()

fig=plt.figure(figsize=(10,7))
gs=gridspec.GridSpec(3,1)
ax=fig.add_subplot(gs[0,:])
fig.suptitle("Contour", fontsize=14, fontweight="bold")
breaks=np.linspace(np.min(phi), np.max(phi), 6)
ax.set_title("Title")    
CS=ax.contourf(C_X,C_Y,phi,breaks, levels=breaks)
row=np.argmax(phi)//cxlen
phi_max=phi[row,:]

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig.colorbar(CS, ticks=breaks, orientation="vertical", format='%.0e')
cbar.ax.set_ylabel('concentration')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid()


ax2=fig.add_subplot(gs[1,:])
ax2.plot(C_x, phi[20,:] , 'r-')

# =============================================================================
# ax3=fig.add_subplot(gs[2,:])
# ax3.plot(C_x, phi[20,:] , 'r-')
# 
# =============================================================================


    
        



