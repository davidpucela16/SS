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
    """This function assembles the laplacian operator"""
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

def norm(v):
    u=np.array([v]) 
    return(np.sqrt(np.sum(u**2)))

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
    costheta=np.dot(normal,v)/norm(v)
    return(costheta)

def get_b_value(h,q_value,D, pq, b, side, string):
    d=norm(np.array(b)-np.array(pq))
    if string=="Flux":
        return(h*q_value*costheta(pq,b,side)/(2*np.pi*d*D))
        
    elif string=="Pressure":
        return(q_value*np.log(1/d)/(2*np.pi*D))

def set_boundary(B, north, south, east, west, C_x, C_y, type_boundary, q, pq):
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

L=50
h=1

x=np.arange(0,L+1,h) #Cell faces

C_x=np.zeros(len(x)-1) #Cell centeres
for i in range(len(C_x)):
    C_x[i]=(x[i]+x[i+1])/2
    
y=x
C_y=C_x

cylen,cxlen=len(C_y), len(C_x)

X,Y=np.meshgrid(x,y)
C_X,C_Y=np.meshgrid(C_x,C_y)



B=np.zeros(cxlen*cylen)


north=np.arange(cxlen* (cylen-1), cxlen* cylen)
south=np.arange(cxlen)
west=np.arange(0,cxlen* cylen, cxlen)
east=np.arange(cxlen-1, cxlen* cylen,cxlen)

boundary=np.concatenate([north, south, east, west])

A=A_assembly(cxlen, cylen)

B=np.zeros(cylen*cxlen)
pos_1=24*cxlen+24
pos_2=20*cxlen+cxlen-30



q=1
D=1

pq=get_cord(pos_1, C_x, C_y)
B=set_boundary(B, north, south, east, west, C_x, C_y, "Flux", q , pq)

pq=get_cord(pos_2, C_x, C_y)
B=set_boundary(B, north, south, east, west, C_x, C_y, "Flux", q , pq)

B[pos_1]=-q/D
B[pos_2]=-q/D





#Because we solve the system in a weak sense it is important to give at least one 
#value to the pressure to ground the solution

A[0,:]=0
A[0,0]=1
B[0]=0





    
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
ax2.plot(C_x, phi[24,:] , 'r-')

ax3=fig.add_subplot(gs[2,:])
ax3.plot(C_x, phi[20,:] , 'r-')



    
        


def test_transmissibility(position_tuple):
    add=0
    x,y=position_tuple
    pp=np.zeros(4)
    for i in range(4):
        if i==0: #north
            a=1-y
            b=x
        if i==1: #south
            a=y
            b=x
        if i==2: #east
            a=1-x
            b=y
        if i==3: #west
            a=x
            b=y
        theta1=np.arctan((1-b)/a)
        theta2=np.arctan(b/a)
        pp[i]=(theta1+theta2+np.sin(2*theta1)/2+np.sin(theta2)/2)/a
        print(pp[i])
        
    print(np.sum(pp))
    return()
        
def test_transmissibility2(pos):
    print("position={}".format(pos))
    trans=np.zeros(4)
    for i in range(4):
        if i==0:#north
            a=np.array([0,1])
            b=np.array([1,1])
        if i==1:
            a=np.array([0,0])
            b=np.array([1,0])
        if i==2:
            a=np.array([1,0])
            b=np.array([1,1])
        if i==3:
            a=np.array([0,0])
            b=np.array([0,1])
        tau=b-a
        xb=b-pos
        xa=a-pos
        trans[i]=np.log((np.linalg.norm(xb)+1+np.dot(tau,xa))/(np.linalg.norm(xa)+np.dot(tau,xa)))
        print(np.sum(trans)/(2*np.pi))
    return(trans)
        
            

def pos_to_coords(x, y, ID):
    xpos=ID%len(x)
    ypos=ID//len(x)
    return(np.array([x[xpos], y[ypos]]))

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
    return(theta)


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

