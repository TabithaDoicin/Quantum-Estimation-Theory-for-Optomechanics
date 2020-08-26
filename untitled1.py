# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 03:33:37 2020

@author: Anton
"""

#!usr/bin/python


import numpy as np
import matplotlib.pyplot as plt
import wrapper as w
import time
import main_script
import warnings
warnings.filterwarnings('ignore')
i = complex(0,1)

wm = 1.1e7
gm = 32
k = 2e5
d0 = 1.1e7
g0_list = np.linspace(200,200.001,2)
g2_list = np.linspace(0.01,0.00001,2)
n = 237.54
min_ep = 3.88e9
max_ep = 1.64e11
ep_list = np.linspace(1e8,1e11,2)

a_sq, qfi = w.find_alpha_and_qfi_over_ep(wm, gm, k, d0, n, ep_list, g0_list, g2_list)

print(a_sq)
print(qfi)
def get_qfi_elem_from_arr(qfi_array, elem):
    res = np.zeros([qfi.size])
    for ii in range(qfi.size):
       res[ii] = qfi_array[ii][elem[0], elem[1]] 
    return res

print(get_qfi_elem_from_arr(qfi,[1,1]))