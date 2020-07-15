# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:17:13 2020

@author: Anton
"""

#!usr/bin/python

import numpy as np
import main_script
from timeit import default_timer as timer 

def find_cov(wm, gm, k, d0, n, ep, g0_list, g2_list):
    cov_array = np.zeros([len(g2_list),len(g0_list)],object)
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_obj = main_script.Little_r(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            cov_array[l,m] = temp_obj.solve_for_cov_from_initial()
    return cov_array