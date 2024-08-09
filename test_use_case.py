# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:53:47 2020

@author: Tabitha
"""

#!usr/bin/python


import numpy as np
import matplotlib.pyplot as plt
import wrapper as w
import time

##MODEL PARAMETERS##

wm = 1.1e7
gm = 32
k = 2e5
d0 = 1.1e7
g0_list = np.linspace(200,200.1,2) #2 values of g0 and g2 required to find difference between covariance matrices, for derivatives., hence small difference between
g2_list = np.linspace(0.1,0.11,2)  #only 2 are really required 
t_list = 1e-3 #1mK
ep_list = np.linspace(1e8,1e10,200)

##
start = time.time()
##

###########

n = w.temp_to_n(t_list,wm) #changing from temp in kelvin to n

a_sq_ep, qfi_list_ep = w.find_alpha_and_qfi_over_ep(wm, gm, k, d0, n, ep_list, g0_list, g2_list) # finding alpha squared list and qfi list

figs, axs = plt.subplots(1,2) # initiating figs and axes to have 2

#log-log plots

list_of_qfi_values_g0_g0 = w.get_qfi_elem_from_arr(qfi_list_ep,[0,0])

rel_error_g0_g0 = w.rel_error(list_of_qfi_values_g0_g0, g0_list[0], wm) #wm and g0's required in rel_error equation

axs[0].scatter(np.log10(a_sq_ep), np.log10(rel_error_g0_g0))

#or more simply, all in one line for g2g2 element
axs[1].scatter(np.log10(a_sq_ep), np.log10(w.rel_error(w.get_qfi_elem_from_arr(qfi_list_ep,[1,1]), g2_list[0], wm)))

##########

##
print('This took ' + str(time.time() - start) + ' seconds.')
##
