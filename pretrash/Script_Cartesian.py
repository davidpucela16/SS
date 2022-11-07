#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:18:28 2021

@author: pdavid
"""

import sys
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg
import time
sys.path.insert(1, '/home/pdavid/Bureau/Code/Solution_splitting')

from assembly_cartesian import Lap_cartesian
#straight forwardly code a new function

K_eff=1
D_eff=1
L=np.array([0.5,0.5,0.5])
h=np.array([0.1,0.1,0.1])


a=Lap_cartesian(1,1,L,h)
a.assembly()
A=sp.sparse.csc_matrix((a.data, (a.col, a.row)), shape=(a.total, a.total))
            
    
    
