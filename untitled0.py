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
g0_list = np.linspace(200,201,10)
g2_list = np.linspace(0,0,1)
n = 237.5
ep_list = np.linspace(1e8,1e12,1000)

##

start = time.time()

##
#r_arr, cov_arr, a_sq = w.prep_qfi(wm, gm, k, d0, n, ep_list[0], g0_list, g2_list)
#qfi_output = w.single_qfi(r_arr, cov_arr, g0_list)
#rel_output = w.rel_error(qfi_output,g0_list)
##
#plt.scatter(a_sq, rel_output)
###

r_arr_arr = np.zeros([len(ep_list)],np.ndarray)
cov_arr_arr = np.zeros([len(ep_list)],np.ndarray)
a_sq_arr = np.zeros([len(ep_list)],np.ndarray)
qfi_output_arr = np.zeros([len(ep_list)],np.ndarray)
rel_output_arr = np.zeros([len(ep_list)],np.ndarray)

for ii in range(len(ep_list)):#implement something like this in wrapper
    r_arr_arr[ii], cov_arr_arr[ii], a_sq_arr[ii] = w.prep_qfi(wm, gm, k, d0, n, ep_list[ii], g0_list, g2_list)
    qfi_output_arr[ii] = w.single_qfi(r_arr_arr[ii], cov_arr_arr[ii], g0_list)
    rel_output_arr[ii] = w.rel_error(qfi_output_arr[ii],g0_list)
    
a_sq_list_epsilon = np.zeros([len(ep_list)])
qfi_list_epsilon = np.zeros([len(ep_list)])
rel_list_epsilon = np.zeros([len(ep_list)])
print(a_sq_arr)

for ii in range(len(ep_list)):
    a_sq_list_epsilon[ii] = a_sq_arr[ii][0][4] #because this outputs for g2 too
    qfi_list_epsilon[ii] = qfi_output_arr[ii][4]
    rel_list_epsilon[ii] = rel_output_arr[ii][4]
    
print(qfi_list_epsilon)
plt.scatter(np.log(a_sq_list_epsilon), np.log(qfi_list_epsilon))
print('This took ' + str(time.time() - start) + ' seconds.')