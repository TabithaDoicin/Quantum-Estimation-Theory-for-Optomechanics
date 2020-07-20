# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:53:47 2020

@author: Anton
"""

#!usr/bin/python


import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import main_script as m
import wrapper as w
import time
import blockwiseview

i = complex(0,1)

wm = 1.1e7
gm = 32
k = 2e5
d0 = 1.1e7
g0_list = np.linspace(200,1000,5)
g2_list = np.linspace(0,0,1)
n = 237.5
ep = 6e8
##
start = time.time()
##
r_arr, cov_arr = w.prep_qfi(wm, gm, k, d0, n, ep, g0_list, g2_list)
#qfi_output = w.single_qfi(r_arr, cov_arr, g0_list)
print(np.around(cov_arr[0,:][0],4))
print(np.around(r_arr[0,:][0],4))
##



print(np.around([r_arr[0,:][ii] for ii in range(len(g0_list))],6))
print(np.around(np.gradient([r_arr[0,:][ii] for ii in range(len(g0_list))],1,axis=0),6))

#print(qfi_output)
print('Done in ' + str(time.time()-start) + ' seconds.')
##use np.block() function to bring together then trace, also this is becoming overcomplicatedly slow :(
test = np.reshape(np.arange(4*4),(4,4))
print(test)
print(blockwiseview.blockwise_view(test,(2,2)))
print(np.reshape(np.ones(4),(2,2)))
print(np.matmul(np.reshape(np.ones(4),(2,2)), blockwiseview.blockwise_view(test,(2,2))))
print(np.matmul(blockwiseview.blockwise_view(test,(2,2)), np.reshape(np.ones(4),(2,2))))