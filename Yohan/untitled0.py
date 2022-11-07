#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:53:14 2021

@author: pdavid
"""

from module_2D_FV_SolSplit import *
import numpy as np 
import matplotlib.pyplot as plt


#1- Set up the domain and the geometry arrays
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

#pos=np.array([24*cxlen+24,20*cxlen+cxlen-30])
pos=np.array([[24*cxlen+24]])

#Set up of the sources
q=1
D=1

boundary_type="Flux"

pq=np.zeros((len(pos),2))

for i in range(len(pos)):
    pq[i]=get_cord(pos[i],C_x, C_y)
    B=set_boundary(B, north, south, east, west, C_x, C_y, boundary_type, q , pq[i],h,D)
    B[pos[i]]=-q/D #set mass conservation on the cell of the source





if boundary_type=="Flux":
    #Because we solve the system in a weak sense it is important to give at least one 
    #value to the pressure to ground the solution
    
    A[0,:]=0
    A[0,0]=1
    B[0]=0
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
ax2.plot(C_x, phi[24,:] , 'r-')

plt.plot(C_x, q/D*np.log(1/C_x))


ax3=fig.add_subplot(gs[2,:])
ax3.plot(C_x, phi[20,:] , 'r-')

