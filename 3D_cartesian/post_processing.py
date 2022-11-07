#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:07:28 2021

@author: pdavid
"""

import numpy as np
import assembly_cartesian as ac
import matplotlib.pyplot as plt

#Post processing
class post_processing():
    def __init__(self, array,domain):
        self.solution=array
        self.lx=len(domain.x)
        self.ly=len(domain.y)
        self.lz=len(domain.z)
        self.x=domain.x
        self.y=domain.y
        self.z=domain.z
        self.f_step=self.lx
        s_step=self.lx*self.ly
        self.s_step=s_step
        
    
    def get_contour(self, coords, normal_axis,title):
        p=self.get_slice(coords, normal_axis)
        print(p)
        if normal_axis=="z":
            toreturn=self.solution[p].reshape(self.ly,self.lx)
            plott(toreturn, title, "x", "y")
            return(toreturn)
        
        if normal_axis=="x":
            toreturn=self.solution[p].reshape(self.lz,self.ly)
            plott(toreturn, title, "y", "z")
            return(toreturn)
        
        if normal_axis=="y":
            toreturn=self.solution[p].reshape(self.lz,self.lx)
            plott(toreturn, title, "x", "z")
            return(toreturn)
        
    def get_slice(self, coords, normal_axis_string):
        """
        l is the tuple with the length of each axis in positions 
        i is the position of the current cell
        normal_axis_string is the axis name in minuscule and string form
        """
        i=ac.coordinates_to_position(coords, self.x, self.y, self.z)
        a=normal_axis_string
        lx, ly, lz=self.lx, self.ly, self.lz
        f_step=lx
        s_step=lx*ly
        total=lx*ly*lz
        
        if a=="x":
            pos_x=i%lx
            return(np.arange(0,total, f_step)+pos_x)
        if a=="y":
            pos_y=((i%s_step)//f_step)*f_step
            array=np.array([]).astype(int)
            for i in range(lz):
                array=np.append(array, np.arange(f_step)+s_step*i)        
            return(array+pos_y)
        if a=="z":
            pos_z=(i//s_step)*s_step
            return(np.arange(s_step)+pos_z)
        
        
    def get_profile(self, coords, axis):
        i=ac.coordinates_to_position(coords, self.x, self.y, self.z)
        lx, ly, lz=self.lx, self.ly, self.lz
        f_step=lx
        s_step=lx*ly
        total=lx*ly*lz
        
        if axis=="x":
            pos_x=i-i%lx  
            toreturn=np.arange(f_step)+pos_x
            return(toreturn)
        if axis=="y":
            pos_y=i-((i%s_step)//f_step)*f_step
            array=np.arange(0,s_step, f_step)      
            toreturn=array+pos_y
            return(toreturn)
        if axis=="z":
            pos_z=i%s_step
            return(np.arange(0,total,s_step)+pos_z)
    
        
    
    def get_middle_plots(self,L, title):
        """L is the 3 term tuple with the length of each axis"""
        
        px=self.get_slice(0.5*L, "x")
        py=self.get_slice(0.5*L,"y")
        pz=self.get_slice(0.5*L, "z")
        
        fig, axs = plt.subplots(2, 2, figsize=(15,15))
        fig.suptitle(title)
        im=axs[0, 0].imshow(self.solution[px].reshape(self.lz,self.ly))
        axs[0, 0].set_title("x_slice") 
        axs[0,0].set_xlabel("y")
        axs[0,0].set_ylabel("z")
        
        im=axs[0, 1].imshow(self.solution[py].reshape(self.lz,self.lx))
        axs[0,1].set_title("y_slice") 
        axs[0,1].set_xlabel("x")
        axs[0,1].set_ylabel("z")
        
        im=axs[1, 0].imshow(self.solution[pz].reshape(self.ly,self.lx))
        axs[1, 0].set_title("z_slice") 
        axs[1,0].set_xlabel("x")
        axs[1,0].set_ylabel("y")
        
        axs[1,1].plot(self.y, self.solution[self.get_profile(0.5*L, "x")])
        axs[1,1].set_title("profile") 
        axs[1,1].set_xlabel("x")
        axs[1,1].set_ylabel("concentration")
        
        fig.colorbar(im, ax=axs.ravel().tolist())
    




def get_25_profiles(along_axis, l, coup_cells, solution, title):
    """l in position length"""
    [lx, ly, lz]=l
    f_step=lx
    s_step=lx*ly
    pos_array=np.array([0.1,0.3,0.5,0.7,0.9])
    if along_axis=="x":
        fig, axs = plt.subplots(3, 2, figsize=(15,15))
        fig.suptitle("profiles along x axis" + " "+title)
        for i in range(5): #y position
            y_pos=int(pos_array[i]*ly)
            array=np.empty((5, int(lx)),dtype=np.int32)
            print("array shape=", array.shape)
            for j in range(5): #along the z position
                z_pos=int(pos_array[j]*lz)
                array[j]=get_profile_from_position(z_pos*s_step+y_pos*f_step, "x", l)
                print("array[j].shape", array[j].shape)
                axs[i//2, i%2].plot(solution[array[j]], label="pz={}".format(pos_array[j]))
            axs[i//2, i%2].set_title("py={}".format(pos_array[i])) 
            axs[i//2, i%2].set_xlabel("x")
            axs[i//2, i%2].set_ylabel("$\phi$")
            axs[i//2, i%2].legend()
            
    if along_axis=="y":
        fig, axs = plt.subplots(3, 2, figsize=(15,15))
        fig.suptitle("profiles along y axis" + " "+title)
        for i in range(5): #z position
            z_pos=int(pos_array[i]*lz)
            array=np.empty((5, int(ly)),dtype=np.int32)
            for j in range(5): #along the x position
                x_pos=int(pos_array[j]*lx)
                array[j]=get_profile_from_position(z_pos*s_step+x_pos, "y", l)
                axs[i//2, i%2].plot(solution[array[j]], label="px={}".format(pos_array[j]))
            axs[i//2, i%2].set_title("pz={}".format(pos_array[i])) 
            axs[i//2, i%2].set_xlabel("y")
            axs[i//2, i%2].set_ylabel("$\phi$")
            axs[i//2, i%2].legend()        
        
    if along_axis=="z":
        fig, axs = plt.subplots(3, 2, figsize=(15,15))
        fig.suptitle("profiles along z axis" + " "+title)
        for i in range(5): #y position
            y_pos=int(pos_array[i]*ly)
            array=np.empty((5, int(lz)),dtype=np.int32)
            print("array shape=", array.shape)
            for j in range(5): #along the x position
                x_pos=int(pos_array[j]*lx)
                array[j]=get_profile_from_position(x_pos+y_pos*f_step, "z", l)
                print("array[j].shape", array[j].shape)
                axs[i//2, i%2].plot(solution[array[j]], label="px={}".format(pos_array[j]))
            axs[i//2, i%2].set_title("py={}".format(pos_array[i])) 
            axs[i//2, i%2].set_xlabel("z")
            axs[i//2, i%2].set_ylabel("$\phi$")
            axs[i//2, i%2].legend()
            
    axs[2,1].plot(solution[coup_cells])
    axs[2,1].set_title("Coupling_cells") 
    axs[2,1].set_xlabel("x")
    axs[2,1].set_ylabel("$\phi$")
    axs[2,1].legend()
    plt.savefig(title + ".pdf")


def get_profiles_comparison_25_plots(along_axis,  l1, solution1,l2, solution2, title, w1, w2):
    [l1x, l1y, l1z]=l1
    [l2x, l2y, l2z]=l2
    f_step1=l1x
    s_step1=l1x*l1y
    f_step2=l2x
    s_step2=l2x*l2y
    pos_array=np.array([0.1,0.3,0.5,0.7,0.9])
    if along_axis=="x":
        fig, axs = plt.subplots(5, 2, figsize=(20,20))
        fig.suptitle("profiles along x axis" + " "+title)
        for i in range(5): #y position
            y_pos1=int(pos_array[i]*l1y)
            y_pos2=int(pos_array[i]*l2y)
            array1=np.empty((5, int(l1x)),dtype=np.int32)
            array2=np.empty((5, int(l2x)),dtype=np.int32)
            for j in range(5): #along the z position
                z_pos1=int(pos_array[j]*l1z)
                z_pos2=int(pos_array[j]*l2z)
                array1[j]=get_profile_from_position(z_pos1*s_step1+y_pos1*f_step1, "x", l1)
                array2[j]=get_profile_from_position(z_pos2*s_step2+y_pos2*f_step2, "x", l2)
                axs[i, 0].plot(w1,solution1[array1[j]], label="pz={}".format(pos_array[j]))
                axs[i, 0].set_ylim((0,1))
                axs[i, 1].plot(w2,solution2[array2[j]], label="pz={}".format(pos_array[j]))
                axs[i, 1].set_ylim((0,1))
            axs[i, 0].set_title("py={}".format(pos_array[i])) 
            axs[i, 0].set_xlabel("x")
            axs[i, 0].set_ylabel("$\phi$")
            axs[i, 0].legend()
            axs[i, 1].set_title("py={}".format(pos_array[i])) 
            axs[i, 1].set_xlabel("x")
            axs[i, 1].set_ylabel("$\phi$")
            axs[i, 1].legend()
        plt.savefig(title + ".pdf")
        

def get_profile_from_position(position, axis, l):
    """This function admits any position along the axis that wants to be returned. 
    Its output will be the position along the axis (axis) that contains the position (position)
    
    Since we work with positions -> position will be int, l will be an array of 3 ints and axis will 
    be a string"""
    i=position
    lx, ly, lz=l
    f_step=lx
    s_step=lx*ly
    total=lx*ly*lz
    if axis=="x":
        pos_yz=i-i%lx  
        toreturn=np.arange(f_step)+pos_yz
        return(toreturn.astype(int))
    if axis=="y":
        pos_xz=i-((i%s_step)//f_step)*f_step
        array=np.arange(0,s_step, f_step)      
        toreturn=array+pos_xz
        return(toreturn.astype(int))
    if axis=="z":
        pos_xy=i%s_step
        return((np.arange(0,total,s_step)+pos_xy).astype(int))

def get_array_axis_plane(pos, along, perp_axis, l):
    """This function returns 5 array of positions of a plane (perpendicular to perp_axis) along a plane 
    described by along, that contains the position pos.
    perp_axis are strings which contain the name of the axis perpendicular ("x", "y", or "z")
    along contains as a string the axis along which the profiles are drawn 
    
    The pos contains the position along the perpendicular axis between 0 and 1 where the plots are located.
    lx,ly, and lz contain the length of the domain vectors
    """ 
    [lx, ly, lz]=l
    f_step=lx
    s_step=lx*ly
    if perp_axis=="y":
        #then pos defines the position along the y axis
        pos_y=np.around(pos*ly).astype(int)-1
        if along=="z":
            pos_x=np.around(np.array([0.1,0.3,0.5,0.7,0.9])*lx).astype(int)
            pos_array=np.empty((len(pos_x), lz))
            for i in range(5): #because of 5 profiles per plane:
                pos_array[i,:]=get_profile_from_position(pos_x[i]+pos_y*f_step, along, l)
                
        elif along=="x":
            pos_z=np.around(np.array([0.1,0.3,0.5,0.7,0.9])*lz).astype(int)
            pos_array=np.empty((len(pos_z), lx))
            for i in range(5): #because of 5 profiles per plane:
                pos_array[i,:]=get_profile_from_position(pos_z[i]*s_step+pos_y*f_step, along, l)

    if perp_axis=="z":
        #then pos defines the position along the y axis
        pos_z=np.around(pos*lz).astype(int)-1
        if along=="x":
            pos_y=np.around(np.array([0.1,0.3,0.5,0.7,0.9])*ly).astype(int)
            pos_array=np.empty((len(pos_y), lx))
            for i in range(5): #because of 5 profiles per plane:
                pos_array[i,:]=get_profile_from_position(pos_z*s_step+pos_y[i]*f_step, along, l)
                
        elif along=="y":
            pos_x=np.around(np.array([0.1,0.3,0.5,0.7,0.9])*lx).astype(int)
            pos_array=np.empty((len(pos_x), ly))
            for i in range(5): #because of 5 profiles per plane:
                pos_array[i,:]=get_profile_from_position(pos_z*s_step+pos_x[i], along, l)

    if perp_axis=="x":
        #then pos defines the position along the y axis
        pos_x=np.around(pos*lx).astype(int)-1
        if along=="z":
            pos_y=np.around(np.array([0.1,0.3,0.5,0.7,0.9])*ly).astype(int)
            pos_array=np.empty((len(pos_y), lz))
            for i in range(5): #because of 5 profiles per plane:
                pos_array[i,:]=get_profile_from_position(pos_x+pos_y[i]*f_step, along, l)
                
        elif along=="y":
            pos_z=np.around(np.array([0.1,0.3,0.5,0.7,0.9])*lz).astype(int)
            pos_array=np.empty((len(pos_z), ly))
            for i in range(5): #because of 5 profiles per plane:
                pos_array[i,:]=get_profile_from_position(pos_x+pos_z[i]*s_step, along, l)
    
    return(pos_array.astype(int))
    
def get_z_contours_comparison(post_processing_object1, post_processing_object2):
    a=post_processing_object1
    b=post_processing_object2
    pos_array=np.array([0.1,0.3,0.5,0.7,0.9])
    fig, axs = plt.subplots(2, 5, figsize=(40,20))
    fig.suptitle("z contours")
    for i in range(5):
        coords_a=np.array([0.5*a.x[-1],0.5*a.y[-1],a.z[int(pos_array[i]*a.lz)]])
        coords_b=np.array([0.5*b.x[-1],0.5*b.y[-1],b.z[int(pos_array[i]*b.lz)]])
        print(coords_a, coords_b)
        slice_a=a.get_slice(coords_a, "z")
        slice_b=b.get_slice(coords_b, "z")
        im=axs[0,i].imshow(a.solution[slice_a].reshape((a.ly, a.lx)), vmin=0, vmax=0.5)
        axs[0,i].set_title("pos {} x z".format(pos_array[i]))
        im=axs[1,i].imshow(b.solution[slice_b].reshape((b.ly, b.lx)), vmin=0, vmax=0.5)
        axs[1,i].set_title("pos {} x z".format(pos_array[i]))
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.savefig("comparison z profiles.pdf")
    
def get_contours_comparison(post_processing_object1, post_processing_object2, title):
    a=post_processing_object1
    b=post_processing_object2
    pos_array=np.array([0.1,0.3,0.5,0.7,0.9])
    fig, axs = plt.subplots(2, 5, figsize=(40,20))
    fig.suptitle("z contours")
    for i in range(5):
        coords_a=np.array([0.5*a.x[-1],0.5*a.y[-1],a.z[int(pos_array[i]*a.lz)]])
        coords_b=np.array([0.5*b.x[-1],0.5*b.y[-1],b.z[int(pos_array[i]*b.lz)]])
        print(coords_a, coords_b)
        slice_a=a.get_slice(coords_a, "z")
        slice_b=b.get_slice(coords_b, "z")
        im=axs[0,i].imshow(a.solution[slice_a].reshape((a.ly, a.lx)), vmin=0, vmax=0.5)
        axs[0,i].set_title("pos {} x z".format(pos_array[i]))
        im=axs[1,i].imshow(b.solution[slice_b].reshape((b.ly, b.lx)), vmin=0, vmax=0.5)
        axs[1,i].set_title("pos {} x z".format(pos_array[i]))
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.savefig(title + "comparison z profiles.pdf")
    
    fig, axs = plt.subplots(2, 5, figsize=(40,10))
    fig.suptitle("y contours")
    for i in range(5):
        coords_a=np.array([0.5*a.x[-1],a.y[int(pos_array[i]*a.ly)],0.5*a.z[-1]])
        coords_b=np.array([0.5*b.x[-1],b.y[int(pos_array[i]*b.ly)],0.5*b.z[-1]])
        slice_a=a.get_slice(coords_a, "y")
        slice_b=b.get_slice(coords_b, "y")
        im=axs[0,i].imshow(a.solution[slice_a].reshape((a.lz, a.lx)), vmin=0, vmax=0.5)
        axs[0,i].set_title("pos {} y".format(pos_array[i]))
        im=axs[1,i].imshow(b.solution[slice_b].reshape((b.lz, b.lx)), vmin=0, vmax=0.5)
        axs[1,i].set_title("pos {} y".format(pos_array[i]))
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.savefig(title + "comparison y profiles.pdf")
    
    fig, axs = plt.subplots(2, 5, figsize=(40,10))
    fig.suptitle("x contours")
    for i in range(5):
        coords_a=np.array([a.x[int(pos_array[i]*a.lx)],0.5*a.y[-1],0.5*a.z[-1]])
        coords_b=np.array([b.x[int(pos_array[i]*b.lx)],0.5*b.y[-1],0.5*b.z[-1]])
        slice_a=a.get_slice(coords_a, "y")
        slice_b=b.get_slice(coords_b, "y")
        im=axs[0,i].imshow(a.solution[slice_a].reshape((a.lz, a.ly)), vmin=0, vmax=0.5)
        axs[0,i].set_title("pos {} x".format(pos_array[i]))
        im=axs[1,i].imshow(b.solution[slice_b].reshape((b.lz, b.ly)), vmin=0, vmax=0.5)
        axs[1,i].set_title("pos {} x".format(pos_array[i]))
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.savefig(title + "comparison x profiles.pdf")
    
def plott(sol, title, xlabel, ylabel):
    plt.imshow(sol,interpolation='bilinear')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.colorbar()