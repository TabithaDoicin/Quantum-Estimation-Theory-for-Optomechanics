# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:53:47 2020

@author: Anton
"""

#!usr/bin/python


import numpy as np
import matplotlib.pyplot as plt
import wrapper as w
import time


i = complex(0,1)

wm = 1.1e7
gm = 32
k = 2e5
d0 = 1.1e7
g0_list = np.linspace(0.1,10000,100)
g2_list = np.linspace(0,0,1)
n = 237.5
ep = 6e8
##
start = time.time()
##
r_arr, cov_arr, a_sq = w.prep_qfi(wm, gm, k, d0, n, ep, g0_list, g2_list)
qfi_output = w.single_qfi(r_arr, cov_arr, g0_list)
rel_output = w.rel_error(qfi_output,g0_list)
##

print(qfi_output)
print('Done in ' + str(time.time()-start) + ' seconds.')

plt.scatter(a_sq,rel_output)