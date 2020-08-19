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

i = complex(0,1)

wm = 1.1e7
gm = 32
k = 2e5
d0 = 1.1e7
g0_list = np.linspace(200,300,2)
g2_list = np.linspace(0,100,2)
n = 237.54
ep = 6e8

r_arr, cov_arr, a_sq_arr = w.prep_qfi_efficient(wm,gm,k,d0,n,ep,g0_list,g2_list)
print(r_arr)
print(cov_arr)
qfi_out, r_diff_a = w.single_qfi(r_arr, cov_arr, g0_list)
print(qfi_out)
qfi_out_b = w.multi_qfi(r_arr, cov_arr, g0_list, g2_list)
print(qfi_out_b)