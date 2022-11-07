#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 19:04:36 2021

@author: pdavid
"""

import numpy as np 
import matplotlib.pyplot as plt

xPoint=np.array([])

yPoint=np.array([])

#This piece of code could (and should) be eliminated. The information of the network is 
#introduced manually here

simplissime=False
nDoFPerCyl=10

if simplissime:
    #DEFINE THE NETWORK (could be read from h5 file)
    diameters=np.array([1.14149,1.03756,1.21498,1.53213,1.27028])*1e-6
    endVertex=np.array([1,2,3,4,5], dtype=int)
    startVertex=np.array([0,1,1,3,3], dtype=int)
    nbPointsPerEdge=np.array([27,32,7,30,28])*nDoFPerCyl
    
    #Necessary to introduce the bifurcation manually 
    bif_list=[[1,3],[[0,1,2],[2,3,4]]] #the number of the vertex and the edges it touches
    
    #BOUNDARY CONDITIONS
    uid_list=np.array([0,2,4,5], dtype=int)
    value_list=np.array([10000,2000,2000,2000])
    value_list_conc=np.array([1,0,0,0])
    
    
    xVertex=np.array([17,9,25,9,21,5])
    yVertex=np.array([2,17,23,23,46,50])
    
else:
    #DEFINE THE NETWORK (could be read from h5 file)
    diameters=np.array([20,10,10])*1e-3
    endVertex=np.array([1,2,3], dtype=int)
    startVertex=np.array([0,1,1], dtype=int)
    nbPointsPerEdge=np.array([20,20,20])*5
    #Necessary to introduce the bifurcation manually 
    bif_list=[[1],[[0,1,2]]] #the number of the vertex and the edges it touches
    
    #BOUNDARY CONDITIONS
    uid_list=np.array([0,2,3], dtype=int)
    value_list=np.array([1000,4000,6000])
    value_list_conc=np.array([0,0.5,1])    

#It is important to know how many vertices there are in the simulation
vertices=np.unique(np.concatenate([endVertex, startVertex]))

L=np.sqrt((xVertex[startVertex]-xVertex[endVertex])**2+(yVertex[startVertex]-yVertex[endVertex])**2)
h=L/nbPointsPerEdge


def plot_network(startVertex, endVertex, xVertex, yVertex):
    """Only purpose is to visualize the network if it is 2D"""
    networkx=np.array([])
    networky=np.array([])
    for i in range(len(startVertex)):
        networkx=np.append(networkx,xVertex[startVertex[i]])
        networkx=np.append(networkx,xVertex[endVertex[i]])
        
        networky=np.append(networky,yVertex[startVertex[i]])
        networky=np.append(networky,yVertex[endVertex[i]])
    plt.plot(networkx, networky)






# =============================================================================
# #Construction of the real cordinates of the network
# theta0=0
# theta1=np.pi/4
# theta2=-np.pi/4
# theta=np.array([theta0, theta1, theta2])
# xVertex=np.array([0, L[0], L[0]+ L[1]*np.cos(theta1), L[0]+ L[2]*np.cos(theta2)])
# yVertex=np.array([0, 0, L[1]*np.sin(theta1), -L[2]*np.sin(theta2)])
# 
# for i in range(len(startVertex)):
#     h=np.append(h,L[i]/nbPointsPerEdge[i])
# 
#     position=np.sum(nbPointsPerEdge[:i])
#     startPoint=np.append(startPoint, position)
#     px, py=(xVertex[startVertex[i]], yVertex[startVertex[i]])
# 
# 
#     
#     xPoint=np.append(xPoint, px+np.arange(h[i]/2, L[i], h[i])*np.cos(theta[i]))
#     yPoint=np.append(yPoint, py+np.arange(h[i]/2, L[i], h[i])*np.sin(theta[i]))
# =============================================================================
    


#PHYSICAL PROPERTY
viscosity=0.004

class flow():
    """This class acts as a flow solver, if the veocities (or flow) are imported from another simulation
    this class is not neccesary"""    
    def __init__(self, uid_list, value_list, L, diameters, startVertex, endVertex, viscosity):
        self.bc_uid=uid_list
        self.bc_value=value_list
        self.L=L
        self.d=diameters
        self.viscosity=viscosity
        self.start=startVertex
        self.end=endVertex
        self.n_vertices=np.max(np.array([np.max(self.start)+1, np.max(self.end)+1]))
        self.viscosity=viscosity
        
        
    def solver(self):
        A=np.zeros([self.n_vertices,self.n_vertices])
        P=np.zeros([self.n_vertices])
        for i in range(len(self.start)): #Loop that goes through each edge assembling the pressure matrix
        
            if self.start[i] not in self.bc_uid:
                A[self.start[i],self.start[i]]-=self.d[i]**4/self.L[i]
                A[self.start[i],self.end[i]]+=self.d[i]**4/self.L[i]
            if self.end[i] not in self.bc_uid:
                A[self.end[i],self.end[i]]-=self.d[i]**4/self.L[i]
                A[self.end[i],self.start[i]]+=self.d[i]**4/self.L[i]
        A[self.bc_uid,self.bc_uid]=1
        P[self.bc_uid]=self.bc_value
        
        self.A=A
        self.P=P
        
        return(A)
    
    def get_U(self):
        """Computes and returns the speed from the pressure values that have been previously computed"""
        pressures=np.linalg.solve(self.solver(), self.P)
        U=np.array([])
        for i in range(len(self.start)):
            vel=self.d[i]**2*(pressures[self.start[i]]-pressures[self.end[i]])/(32*self.viscosity*self.L[i])
            U=np.append(U,vel)
        return(U)
            
            
        


class massTransport():
    def __init__(self,  startVertex, endVertex, D, K, U, bif_list, nbPointsPerEdge, vertices, d, h, uid_list, value_list_conc,L):
        #advection scheme
        self.nbPointsPerEdge=nbPointsPerEdge
        self.n_edges=len(startVertex)
        self.D_eff=D #array 
        self.K_eff=K #array
        self.U_eff=U #array
        self.start=startVertex
        self.end=endVertex
        self.bif_list=bif_list
        self.c=np.zeros([len(vertices),np.sum(self.nbPointsPerEdge)+len(vertices)])
        self.A=np.zeros([np.sum(self.nbPointsPerEdge), np.sum(self.nbPointsPerEdge)+len(vertices)])
        self.vertices=vertices #Total amount of vertices present in the simulation
        
        self.L=L #Length of vessels
        self.d=d #diameter of vessels (should be an array since there are multiple vessels)
        self.h=h #discretization size of vessel
        self.leninner=np.sum(self.nbPointsPerEdge)
        self.bc=uid_list
        self.bc_value=value_list_conc
        
    def solver(self):
        """This function assembles the matrix for the effective mass transport equation"""
        A=self.A
        
        for i in range(self.n_edges): #Loop that goes through each of the edges/vessels
            D_eff=self.D_eff[i]
            K_eff=self.K_eff[i]
            U_eff=self.U_eff[i]
            h=self.h[i]
            if U_eff > 0:
                window=[0,1,-1,0,0]
                w=[1,-1,0]
            elif U_eff< 0:
                window=[0,0,1,-1,0]
                w=[0,1,-1]
            #UPWINDED ADVECTION FD SCHEME FOR 1D MASS TRANSPORT WITH REACTION
            s=self.nbPointsPerEdge[i] #amount of points the loop has to go through
            absolute=np.sum(self.nbPointsPerEdge[:i])  
            #For the first vertex of the edge, we work with the vertex for the incomming flux:
            p0, pf=self.start[i]+self.leninner, self.end[i]+self.leninner
            
            #OUTER DOFs -> The ones that neighbor the bifurcations/BCs
            #For the first:
            A[absolute,p0]+=(w[0]*U_eff+2*D_eff/h)/h
            A[absolute,absolute]+=(-K_eff*h-3*D_eff/h+w[1]*U_eff)/h
            A[absolute,absolute+1]+=(D_eff/h+w[2]*U_eff)/h
            
            #For the last:
            A[absolute+s-1,absolute+s-2]+=(w[0]*U_eff+D_eff/h)/h
            A[absolute+s-1,absolute+s-1]+=(-K_eff*h-3*D_eff/h+w[1]*U_eff)/h
            A[absolute+s-1,pf]+=(2*D_eff/h+w[2]*U_eff)/h
            

            for j in (np.arange(s-2)+1): #Goes through the inner vertices where there is no bif or BC
                k=j+absolute
                if i!=1 and i!=s-2: #Inner DoFs
                    A[k,k-2]+=(window[0]*U_eff)/h
                    A[k,k-1]+=(window[1]*U_eff+D_eff/h)/h
                    A[k,k]+=(-K_eff*h-2*D_eff/h+window[2]*U_eff)/h
                    A[k,k+1]+=(D_eff/h+window[3]*U_eff)/h
                    A[k,k+2]+=(window[4]*U_eff)/h

                if i==s-2 or i== 1: #still inner DoFs
                    #For this cases a 5 element window does not work, to preserve mass a 
                    #generic upwind is put in place
                    A[k,k-1]+=(w[0]*U_eff+D_eff/h)/h
                    A[k,k]+=(-K_eff*h-2*D_eff/h+w[1]*U_eff)/h
                    A[k,k+1]+=(D_eff/h+w[2]*U_eff)/h
            

        for i in range(len(self.bif_list[0])): #goes through each bifurcation
            #On each bifurcation we need the information of each edge, and the 
            #whether it is the final vertex or initial
            
            #We can see that the advection is upwinded
            
            vertex=self.bif_list[0][i]
            for j in self.bif_list[1][i]: #goes through each of the edges of the bif
            #All the sum of the flux going in is equal to zero
                D_eff=self.D_eff[j]
                K_eff=self.K_eff[j]
                U_eff=self.U_eff[j]
                h=self.h[j]
                d=self.d[j]
                print("bifurcation {k}, and edge {m}".format(k=i, m=j))
                
                if self.start[j]==vertex: #the bifurcation is the initial point of the edge
                    position_edge=np.sum(self.nbPointsPerEdge[:j])
                    self.c[vertex, position_edge]+=d**2*(np.max([-U_eff,0])+2*D_eff/h)
                    self.c[vertex, self.leninner+vertex]-=d**2*(np.max([U_eff,0])+D_eff*2/h)
                    
                elif self.end[j]==vertex: #the bifurcation is the ending point of the edge
                    position_edge=np.sum(self.nbPointsPerEdge[:j+1])-1
                    self.c[vertex, position_edge]+=d**2*(np.max([U_eff,0])+D_eff*2/h)
                    self.c[vertex, self.leninner+vertex]-=d**2*(np.max([-U_eff,0])+D_eff*2/h)
                    
                else:
                    print('error!!')
        
        for i in self.bc:
            self.c[i, i+self.leninner]=1
            
        #initialize the BCs:
        self.phi=np.zeros(A.shape[1])
        self.phi[self.leninner+self.bc]=self.bc_value
        return()
    
    def set_no_diff_BC(self,edges):
        """This function is necessary in case we have some boundary vertex where there is not a 
        Dirichlet BC being defined"""
        for i in edges:
            pos=self.get_edge(i)
            self.c[pos[-1]-kk.leninner,:]=0
            self.phi[pos[-1]]=0
            self.c[pos[-1]-self.leninner,pos[-1]]=1
            self.c[pos[-1]-self.leninner,pos[-2]]=-1
    
    def get_matrix(self):
        return(np.vstack([self.A, self.c]))
    
    def get_RHS(self):
        return(self.phi)
    
    def get_edge(self, edge_id):
        """Function to return the positions in the RHS of the vector of each edge 
        Including the ending vertices"""
        nbPointsPerEdge=self.nbPointsPerEdge
        startVertex=self.start
        endVertex=self.end
        leninner=np.sum(nbPointsPerEdge)
        ret=np.append(startVertex[edge_id]+leninner, np.arange(np.sum(nbPointsPerEdge[:edge_id]), np.sum(nbPointsPerEdge[:edge_id+1])))
        ret=np.append(ret, endVertex[edge_id]+leninner)
        return(ret)
    
    
    def solve_SS_plot(self):
        solution=np.linalg.solve(self.get_matrix(), self.get_RHS())
        self.plot_edge(solution)
        self.sol=solution
        return()
    
    def plot_edge(self, solution):
        """Usefull function that gives (in order) the position in the matrix (and the RHS) of each
        of the vertex belonging to an edge (including the bifurcation or boundary vertex at the 
                                            extremities of the edge (first and last value))"""
        for i in range(len(self.start)):
            """Loop that goes through each edge extracting the info from the solution array"""
            pos=self.get_edge(i)
            s=np.append(np.append(0, np.arange(self.h[i]/2, self.L[i], self.h[i])), L[i])
            plt.figure()
            plt.plot(s, solution[pos])
            plt.ylim((-0.1,1.1))
            plt.title("concentration steady state for edge {j}".format(j=i), fontsize=16)
            plt.show()
        
    def compute_implicit_matrix(self, inc_t):
        self.M=np.linalg.inv(np.eye(len(self.get_RHS()))-inc_t*self.get_matrix())
        return(self.M)
        
    def iterate_implicit(self, phi):
        """Performs one iteration in implicit mode"""
        M=self.M
        #This next two lines are to ensure the Dirichlet BC stay in place
        M[kk.leninner+self.bc,:]=0
        M[kk.leninner+self.bc,self.leninner+self.bc]=1
        self.M=M
        phi_plusone=M.dot(phi)
        
        return(phi_plusone)
            

    

k=flow(uid_list, value_list, L ,diameters, startVertex, endVertex, viscosity)
U_eff=k.get_U()

#Copied velocity from flow solver:
#U_eff=np.zeros(5)+1.95432e6*(np.pi*diameters**2)/4
D_v=np.zeros(5)+1e-9
D_eff=D_v
Da_m=1e-6
K_m=Da_m*D_v/(diameters*L)
K_eff=D_v*Da_m/(L*diameters)

U_eff= 2*h
D_eff= 3*h**2
K_eff= np.zeros(5)+0.1
kk=massTransport(startVertex, endVertex, D_eff, K_eff, U_eff, bif_list, nbPointsPerEdge, vertices, diameters, h, uid_list, value_list_conc,L)
kk.solver()


sol=np.linalg.solve(kk.get_matrix(), kk.get_RHS())
kk.plot_edge(sol)










#bmf solver massTransport massTransport.main.json | grep row | sed -e 's/row .*://' -e 's/ *$//' -e 's%)\(.\)%),\1%g' -e 's/^/\[/' -e 's/$/\]/' > file_qq


        
        
        
        
        