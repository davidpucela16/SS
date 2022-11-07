#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:52:38 2021

@author: pdavid
"""

import assembly_cartesian as ac
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg
import time



def visualize_laplacian2(lap, x, y,z, *k):
    with open("lap_RHS_pos.txt", "w") as text_file:
        if k:
            c=k[0]; 
        else:
            c=0
            
        for i in lap:
            print("\n", file=text_file)
            print("pos= {}".format(c), file=text_file)
            matrix=i.reshape(len(z), len(y), len(x))
            pos=ac.array_to_tuple(c, x, y, z)
            
            print("xy plane", file=text_file)
            print(matrix[pos[2],:,:], file=text_file)
            
            if pos[2]<len(z)-1:
                print("z+1 plane", file=text_file)
                print(matrix[pos[2]+1,:,:], file=text_file)
            else:
                print("This is north boundary", file=text_file)
                
            if pos[2]>0:
                print("z-1 plane", file=text_file)
                print(matrix[pos[2]-1,:,:], file=text_file)
            else:
                print("this is south boundary", file=text_file)
            c+=1

def visualize_laplacian(lap, x, y,z):
    with open("Output.txt", "w") as text_file:
        c=0
        for i in lap:
            print("\n", file=text_file)
            print("pos= {}".format(c), file=text_file)
            matrix=i.reshape(len(z), len(y), len(x))
            pos=ac.array_to_tuple(c, x, y, z)
            
            print("xy plane", file=text_file)
            print(matrix[pos[2],:,:], file=text_file)
            
            if pos[2]<len(z)-1:
                print("z+1 plane", file=text_file)
                print(matrix[pos[2]+1,:,:], file=text_file)
            else:
                print("This is north boundary", file=text_file)
                
            if pos[2]>0:
                print("z-1 plane", file=text_file)
                print(matrix[pos[2]-1,:,:], file=text_file)
            else:
                print("this is south boundary", file=text_file)
            c+=1


L=np.array([5,5,5])
h=np.array([1,1,1])
hs=np.min(h)/5

init=np.array([2,0,0])
end=np.array([0,5,15])


Lx,Ly,Lz=L
#Constant vessel along the x axis
init=np.array([0,Ly/2,Lz/2])
end=np.array([Lx,Ly/2,Lz/2])


op=ac.Lap_cartesian(1, L,h)
op.assembly()
lap=sp.sparse.csc_matrix((op.data, (op.row, op.col)), shape=(op.total, op.total))


#Set Dirichlet BC
c=0
for i in op.boundary.T:
    code=i[1]
    pos=i[0]
    if code>10: #everything except north and south
        lap[pos,:]=0
        lap[pos,pos]=1
        print(pos)
        

visualize_laplacian(lap.toarray(), op.x, op.y, op.z)