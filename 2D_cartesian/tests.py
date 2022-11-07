#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:22:03 2021

@author: pdavid
"""

import numpy as np

def test_conservativeness(solution, coup_cells, total_FV,cell_ID):
    n_sources=len(coup_cells)
    pos=np.where(coup_cells==cell_ID)[0][0]
    v_values=solution[total_FV+n_sources+4*pos:total_FV+n_sources+4*(pos+1)]
    
    q_value=solution[total_FV+pos]
    
    
    