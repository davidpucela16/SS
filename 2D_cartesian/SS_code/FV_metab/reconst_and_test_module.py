
import os 
import numpy as np
from module_2D_coupling_FV_nogrid import * 

def get_errors(phi_SS, a_rec_final, noc_sol, p_sol,SS_phi_q, p_q, noc_q,ratio, phi_q):
    errors=[["coupling","SS" , ratio , get_L2(SS_phi_q, phi_q) , get_L2(phi_SS, a_rec_final) , get_MRE(SS_phi_q, phi_q) , get_MRE(phi_SS, a_rec_final)],
        ["coupling","Peaceman", ratio,get_L2(p_q, phi_q), get_L2(p_sol, np.ndarray.flatten(a_rec_final)), get_MRE(p_q, phi_q), get_MRE(p_sol, np.ndarray.flatten(a_rec_final))],
        ["FV","SS",1,get_L2(SS_phi_q, noc_q), get_L2(np.ndarray.flatten(phi_SS), noc_sol), get_MRE(SS_phi_q, phi_q), get_MRE(np.ndarray.flatten(phi_SS), noc_sol)],
        ["FV","Peaceman",1,get_L2(p_q, noc_q), get_L2(p_sol, noc_sol), get_MRE(p_q, phi_q), get_MRE(p_sol, noc_sol)],
        ["Peaceman","SS", 1,get_L2(SS_phi_q, p_q), get_L2(np.ndarray.flatten(phi_SS), p_sol), get_MRE(SS_phi_q, p_q), get_MRE(np.ndarray.flatten(phi_SS), p_sol)]]
    return(errors)

def separate_unk(ass_object, solution):
    """gets three arrays with the FV solution, reg terms, and sources"""
    xy=len(ass_object.x)*len(ass_object.y)
    s=len(ass_object.s_blocks)
    return(solution[:xy],solution[-s:]) 

def get_sub_x_y(orig_x, orig_y, orig_h, ratio):
    """returns the subgrid for that ratio and those originals"""
    h=orig_h/ratio 
    L=orig_x[-1]+orig_h/2
    num=int(L/h)
    h=L/num
    x=np.linspace(h/2, L-h/2, num)
    y=x
    return(x,y)

def modify_matrix(original, pos_x, pos_y, value_matrix):
    #pdb.set_trace()
    if not value_matrix.shape==(len(pos_y), len(pos_x)):
        raise TypeError("input shape not appropriate to include in the matrix")
    x_micro_len=original.shape[1]
    array_pos=np.array([], dtype=int)
    for j in pos_y:
        array_pos=np.concatenate((array_pos, pos_x+j*x_micro_len))
    original_flat=np.ndarray.flatten(original)
    original_flat[array_pos]=np.ndarray.flatten(value_matrix)
    return(original_flat.reshape(original.shape))


    
def bilinear_interpolation(corner_values, ratio):
    """The corner values must be given in the form of an np.array, in the following order:
        (0,0), (0,1), (1,0), (1,1)"""
    rec_block=np.zeros((ratio, ratio))
    A=np.array([[1,-1,-1,1],[0,0,1,-1],[0,1,0,-1],[0,0,0,1]])
    c=0
    for i in np.arange(0,1,1/ratio)+1/(2*ratio):
        d=0
        for j in np.arange(0,1,1/ratio)+1/(2*ratio):
            weights=A.dot(np.array([1,i,j,i*j]))
            rec_block[d, c]=weights.dot(corner_values)
            d+=1
        c+=1
# =============================================================================
#     for i in np.linspace(0,1,ratio):
#         d=0
#         for j in  np.linspace(0,1,ratio):
#             weights=A.dot(np.array([1,i,j,i*j]))
#             rec_block[d, c]=weights.dot(corner_values)
#             d+=1
#         c+=1
# =============================================================================
    return(rec_block)

def reduced_half_direction(phi_k, axis):
    """Function made to reduce by half the size of the matrix phi_k in one direction
    given by axis.
    A priori, axis=0 is for the y direction and axis=1 for the x direction ,just like np.concatenate"""
    ss=phi_k.shape
    if axis==1:
        #This would mean we reduce in the x direction therefore it is north or south boundary:
        st=int(ss[1]/2)
        aa=np.arange(st, dtype=int)
        to_return=(phi_k[:,2*aa]+phi_k[:,2*aa+1])/2
    elif axis==0:
        #This would mean we reduce in the y direction therefore it is north or south boundary:
        st=int(ss[0]/2)
        aa=np.arange(st, dtype=int)
        to_return=(phi_k[2*aa,:]+phi_k[2*aa+1,:])/2
    return(to_return)
    
def NN_interpolation(corner_values, ratio):
        """The corner values must be given in the form of an np.array, in the following order:
        (0,0), (0,1), (1,0), (1,1)"""
        rec_block=np.zeros((ratio, ratio))
        pos_corners=np.array([[0,0],[0,1],[1,0],[1,1]])
        c=0
        for i in np.arange(0,1,1/ratio)+1/(2*ratio):
            d=0
            for j in np.arange(0,1,1/ratio)+1/(2*ratio):
                dist=np.zeros(4)
                for k in range(4):
                    dist[k]=np.linalg.norm(pos_corners[k,:]-np.array([j,i]))
                
                rec_block[d, c]=corner_values[np.argmin(dist)] #takes the value of the closest one
                d+=1
            c+=1
        return(rec_block)

def get_Green_neigh_array(p_x, neigh, s_blocks, pos_s, Rv):
    """Returns the array to multiply the array of source_fluxes that will calculate
    the value of the singular term of the sources in the given neigh at the given point p_x
    
    $\sum_{j \in neigh} G(x_j, p_x)$"""
    array=np.zeros(len(s_blocks))
    sources=np.arange(len(s_blocks))[np.in1d(s_blocks, neigh)]
    for i in sources:
        value=Green(pos_s[i], p_x, Rv)
        array[i]=value
    return(array)

def green_to_block(phi_q, pos_center, original_h, ratio, neigh_blocks, s_blocks, pos_s,Rv):
    """This function will add the contribution from the given sources to each of 
    the sub-discretized block"""
    rec_block=np.zeros((ratio, ratio))
    c=0
    h=original_h
    for i in h*(np.arange(0,1,1/ratio)+1/(2*ratio)-1/2):
        d=0
        for j in h*(np.arange(0,1,1/ratio)+1/(2*ratio)-1/2):
            w=get_Green_neigh_array(pos_center+np.array([i,j]), neigh_blocks, s_blocks, pos_s, Rv)
            rec_block[d, c]=w.dot(phi_q)
            d+=1
        c+=1
    return(rec_block)
    
    

def get_block_reference(block_ID, ref_neighbourhood, directness, xlen, s_blocks, 
                        uni_s_blocks,phi_FV, phi_q, pos_s, Rv, x, y):
    """gets a single block in reference to the singular term in ref_neighourhood"""
    phi=phi_FV
    #get the uncommon blocks
    a=get_uncommon(ref_neighbourhood, get_neighbourhood(directness, xlen, block_ID))
    #get the value of the uncommon singular term at the given block
    unc_sing_array=get_Green_neigh_array(pos_to_coords(x, y, block_ID), a, s_blocks, pos_s, Rv)
    value=unc_sing_array.dot(phi_q)
    
    #substract the uncommon sources from the unknown
    return(phi[block_ID]-value)
    
def get_unk_same_reference(blocks, directness, xlen, s_blocks, uni_s_blocks, phi_FV, phi_q, pos_s, Rv, x, y):
    """Designed to get a given set the unknowns in the same frame of reference
    regarding the singular term.
    The frame of reference will the that one accounting for influence of all the 
    sources in the ensemble of all the neighbourhoods of the blocks given to the function"""
    #pdb.set_trace()
    ens_neigh=get_multiple_neigh(directness, xlen, blocks)
    total_sources=np.in1d(s_blocks, ens_neigh)
    
    unk_extended=np.zeros((len(blocks)))
    c=0
    for i in blocks:
        unk_extended[c]=get_block_reference(i, ens_neigh, directness, xlen, s_blocks, 
                                            uni_s_blocks,phi_FV,  phi_q, pos_s, Rv, x, y)
        c+=1
    return(unk_extended)
        
def get_4_blocks(position, tx, ty, th):
    #pdb.set_trace()
    blocks_x=np.where(np.abs(tx-position[0]) < th)[0]
    blocks_y=np.where(np.abs(ty-position[1]) < th)[0]*len(tx)
    blocks=np.array([blocks_y[0]+blocks_x[0], blocks_y[1]+blocks_x[0],blocks_y[0]+blocks_x[1],blocks_y[1]+blocks_x[1]])
    return(blocks)
    


class reconstruction_sans_flux():
    def __init__(self, solution, ass_object, L, ratio, directness):
        self.phi_FV, self.phi_q=separate_unk(ass_object, solution)
        self.t=ass_object
        t=self.t
        self.L=L
        self.dual_x=np.arange(0, L+0.01*t.h, t.h)
        self.dual_y=np.arange(0, L+0.01*t.h, t.h)
        self.rec_final=np.zeros((len(t.x)*ratio, len(t.x)*ratio))
        self.ratio=ratio
        self.directness=directness
        self.dual_boundary=get_boundary_vector(len(self.dual_x), len(self.dual_y))
        
    def get_bilinear_interpolation(self, position):
        #pdb.set_trace()
        blocks=get_4_blocks(position, self.t.x, self.t.y, self.t.h) #gets the ID of each of the 4 blocks 
        corner_values=get_unk_same_reference(blocks, self.directness, len(self.t.x), 
                               self.t.s_blocks, self.t.uni_s_blocks, self.phi_FV, 
                               self.phi_q, self.t.pos_s, self.t.Rv, self.t.x, self.t.y)
        
        ens_neigh=get_multiple_neigh(self.directness, len(self.t.x), blocks)
        total_sources=np.arange(len(self.t.s_blocks))[np.in1d(self.t.s_blocks, ens_neigh)]
        rec=bilinear_interpolation(corner_values, self.ratio)
        
        if np.any(np.in1d(ens_neigh, self.t.uni_s_blocks)):
            rec+=green_to_block(self.phi_q, position,self.t.h,self.ratio, ens_neigh, 
                                self.t.s_blocks,self.t.pos_s,self.t.Rv)
        

        return(rec)
    
    def get_NN_interpolation(self, position):
        #pdb.set_trace()
        blocks=get_4_blocks(position, self.t.x, self.t.y, self.t.h)
        corner_values=get_unk_same_reference(blocks, self.directness, len(self.t.x), 
                               self.t.s_blocks, self.t.uni_s_blocks, self.phi_FV,
                               self.phi_q, self.t.pos_s, self.t.Rv, self.t.x, self.t.y)
        
        ens_neigh=get_multiple_neigh(self.directness, len(self.t.x), blocks)
        total_sources=np.arange(len(self.t.s_blocks))[np.in1d(self.t.s_blocks, ens_neigh)]
        
        rec=NN_interpolation(corner_values, self.ratio)

        if np.any(np.in1d(ens_neigh, self.t.uni_s_blocks)):
            rec+=green_to_block(self.phi_q, position,self.t.h,self.ratio, ens_neigh, 
                                self.t.s_blocks,self.t.pos_s,self.t.Rv)
            
        return(rec) 
        
    def reconstruction(self, *rec_type):
        t=self.t
        ratio=self.ratio
        x,y=get_sub_x_y(t.x, t.y, t.h, ratio)
        self.x, self.y=x,y
        c=0
        if rec_type:
            typ="NN"
        else:
            typ="bil"
        for i in self.dual_x[1:-1]:
            d=0
            for j in self.dual_y[1:-1]:
                pos_x=((x>=i-t.h/2) & (x<i+t.h/2))
                pos_y=((y>=j-t.h/2) & (y<j+t.h/2))
                
                if typ=="NN":
                    local_sol=self.get_NN_interpolation(np.array([i,j]))
                    print("Nearest neighbourg interpolation")
                else:
                    local_sol=self.get_bilinear_interpolation(np.array([i,j]))
                    
                self.rec_final=modify_matrix(self.rec_final, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], local_sol)
                d+=1
            c+=1
        return(self.rec_final)
    
    def reconstruction_boundaries(self, boundary_values):
        dual_bound=get_boundary_vector(len(self.dual_x), len(self.dual_y))
        phi_q=self.phi_q
        ratio=self.ratio
        s_blocks=self.t.s_blocks
        pos_s=self.t.pos_s
        Rv=self.t.Rv
        t=self.t
        x,y=get_sub_x_y(t.x, t.y, t.h, ratio)
        h=t.h
        self.boundary_values=boundary_values
        
        Bn, Bs, Be, Bw=boundary_values   
        for c in range(4):
            for b in self.dual_boundary[c,1:-1]:

                normal=np.array([[0,1],[0,-1],[1,0],[-1,0]])[c]
                tau=np.array([[0,1],[-1,0]]).dot(normal)
                
                #position at the center point of the boundary of this block:
                #Also can be described as the center of hte dual block 
                p_dual=pos_to_coords(self.dual_x,self.dual_y, b)
                
                #the unknowns that are included here
                first_block=coord_to_pos(t.x, t.y,p_dual -(normal+tau)*h/2)
                second_block=coord_to_pos(t.x, t.y, p_dual+(-normal+tau)*h/2)
                
                blocks=np.array([first_block, second_block])
                ens_neigh=get_multiple_neigh(self.directness, len(t.x), blocks)
                unk_extended=get_unk_same_reference(blocks, self.directness, len(t.x), t.s_blocks, 
                                                    t.uni_s_blocks, self.phi_FV, phi_q, t.pos_s, Rv, t.x, t.y)
                
                unk_boundaries=np.zeros(2)
                position=np.zeros((2,2))
                for k in range(2):
                    #position of the boundary unknown
                    position[k,:]=pos_to_coords(t.x, t.y, blocks[k])+normal*t.h/2
                    unk_boundaries[k]=boundary_values[c]-get_Green_neigh_array(position[k,:], ens_neigh, s_blocks, pos_s, Rv).dot(phi_q)
                
                rec=bilinear_interpolation(np.array([unk_extended[0], unk_boundaries[0], unk_extended[1], unk_boundaries[1]]), ratio)
                
                pos_y=((y>=p_dual[1]-t.h/2) & (y<p_dual[1]+t.h/2))
                pos_x=((x>=p_dual[0]-t.h/2) & (x<p_dual[0]+t.h/2))
                
                if c==1:
                    rec=rotate_180(rec)
                elif c==3:
                    rec=rotate_clockwise(rec)
                elif c==2:
                    rec=rotate_counterclock(rec)
                
                local_sol=reduced_half_direction(rec,int(np.abs(normal[0])))
                for i in np.arange(np.sum(pos_y)):
                    for j in np.arange(np.sum(pos_x)):
                        #position of the point
                        p_pos=np.array([x[pos_x][j], y[pos_y][i]])
                        kernel=get_Green_neigh_array(p_pos, ens_neigh, t.s_blocks, t.pos_s, t.Rv)
                        local_sol[i,j]+=kernel.dot(phi_q)
                
                self.rec_final=modify_matrix(self.rec_final, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], local_sol)
                
                
        return()
    
    def rec_corners(self):
        
        phi_q=self.phi_q
        ratio=self.ratio
        s_blocks=self.t.s_blocks
        pos_s=self.t.pos_s
        Rv=self.t.Rv
        t=self.t
        bound=get_boundary_vector(len(t.x), len(t.y))
        
        x,y=get_sub_x_y(t.x, t.y, t.h, ratio)
        h=t.h
        Bn, Bs, Be, Bw=self.boundary_values 
        
        
        for i in range(4):
            if i==0:
                #south-west
                p_dual=np.array([self.dual_x[0], self.dual_y[0]])
                block=bound[1,0]
                pos_y=(y<t.y[0])
                pos_x=(x<t.x[0])
                
                neigh=get_neighbourhood(self.directness, len(t.x), block)
                value=self.phi_FV[block]+get_Green_neigh_array(pos_to_coords(t.x, t.y, block), neigh, t.s_blocks, t.pos_s, t.Rv).dot(phi_q)
                array_bil=np.array([(Bs+Bw)/2, Bw, Bs, value])
            
            if i==1:
                #north-west
                p_dual=np.array([self.dual_x[0], self.dual_y[-1]])
                block=bound[0,0]
                pos_y=(y>t.y[-1])
                pos_x=(x<t.x[0])
                
                neigh=get_neighbourhood(self.directness, len(t.x), block)
                value=self.phi_FV[block]+get_Green_neigh_array(pos_to_coords(t.x, t.y, block), neigh, t.s_blocks, t.pos_s, t.Rv).dot(phi_q)
                array_bil=np.array([Bw, (Bw+Bn)/2, value, Bn])
                
            if i==2:
                #south-east
                p_dual=np.array([self.dual_x[-1], self.dual_y[0]])
                block=bound[1,-1]       
                pos_y=(y<t.y[0])
                pos_x=(x>t.x[-1])
                neigh=get_neighbourhood(self.directness, len(t.x), block)
                value=self.phi_FV[block]+get_Green_neigh_array(pos_to_coords(t.x, t.y, block), neigh, t.s_blocks, t.pos_s, t.Rv).dot(phi_q)
                array_bil=np.array([Bs, value, (Bs+Be)/2, Be])
                
                
            if i==3:
                #north-east
                p_dual=np.array([self.dual_x[0], self.dual_y[-1]])
                block=bound[0,-1]
                pos_y=(y>t.y[-1])
                pos_x=(x>t.x[-1])
                neigh=get_neighbourhood(self.directness, len(t.x), block)
                value=self.phi_FV[block]+get_Green_neigh_array(pos_to_coords(t.x, t.y, block), neigh, t.s_blocks, t.pos_s, t.Rv).dot(phi_q)
                array_bil=np.array([value,Bn, Be, (Bn+Be)/2])


            local_sol=bilinear_interpolation(array_bil, int(ratio/2))
            
            #I am very tired, so the reconstruction in the corners will be done with the absolute value of the concentration
            self.rec_final=modify_matrix(self.rec_final, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], local_sol)
                        
@njit
def rotate_clockwise(matrix):
    sh=matrix.shape
    new_matrix=np.zeros((sh[1], sh[0]))
    for i in range(sh[0]):        
        for j in range(sh[1]):
            row=j
            col=sh[0]-i-1
            new_matrix[row, col]=matrix[i,j]
    return(new_matrix)

@njit
def rotate_180(matrix):
    new=rotate_clockwise(matrix)
    new=rotate_clockwise(new)
    return(new)
@njit
def rotate_counterclock(matrix):
    new=rotate_clockwise(matrix)
    new=rotate_clockwise(new)
    new=rotate_clockwise(new)
    return(new)



def coarse_NN_rec(x, y, phi_FV, pos_s, s_blocks, phi_q, ratio, h, directness, Rv):
        """phi_FV is given in matrix form
        This funcion does a coarse mesh reconstruction with average u values on the cells plus
        singular term values at the cell's center"""
        rec=np.zeros((len(y), len(x)))
        
        for i in range(len(y)):
            for j in range(len(x)):
                neigh=get_neighbourhood(directness, len(x), i*len(x)+j)
                Ens=np.arange(len(s_blocks))[np.in1d(s_blocks, neigh)]#sources in the neighbourhood
                rec[i,j]=phi_FV[i, j]
                for k in Ens:
                    rec[i,j]+=Green(pos_s[k], np.array([x[j],y[i]]), Rv)*phi_q[k]
                
        return(rec)