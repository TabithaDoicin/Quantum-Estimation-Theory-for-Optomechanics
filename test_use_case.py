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

##MODEL PARAMETERS##

wm = 1.1e7
gm = 32
k = 2e5
d0 = 1.1e7
g0_list = np.linspace(200,200.1,2)
g2_list = np.linspace(0.1,0.11,2)
t_list = np.linspace(0, 10000,20)
min_ep = 3.88e9
max_ep = 1.64e11
ep_list = np.linspace(1e8,1e10,200)


##
start = time.time()
##

###########

n_list = w.temp_to_n(t_list,wm)
results = np.zeros([len(n_list),len(ep_list)])

a_sq_ep, qfi_list_ep = w.find_alpha_and_qfi_over_ep(wm, gm, k, d0, n_list[1], ep_list, g0_list, g2_list)

figs, axs = plt.subplots(1,2)

axs[0].scatter(np.log10(a_sq_ep), np.log10(w.rel_error(w.get_qfi_elem_from_arr(qfi_list_ep,[0,0]), g0_list[0], wm)))
axs[1].scatter(np.log10(a_sq_ep), np.log10(w.rel_error(w.get_qfi_elem_from_arr(qfi_list_ep,[1,1]), g2_list[0], wm)))

###########

##
print('This took ' + str(time.time() - start) + ' seconds.')
##
