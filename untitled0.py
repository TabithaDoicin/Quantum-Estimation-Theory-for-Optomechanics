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
import main_script as m

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
ep_list = np.linspace(1e8,1e11,1000)

##

start = time.time()



a_sq, qfi = w.find_alpha_and_qfi_over_ep(wm, gm, k, d0, n, ep_list, g0_list, g2_list)




fig, ax = plt.subplots()
#ax.scatter(np.log(a_sq_list_epsilon), np.log(qfi_list_epsilon))
ax.scatter(np.log(a_sq), np.log(w.get_qfi_elem_from_arr(qfi, [0,0])))
plt.xlabel(r'$ \ln(\mid\alpha\mid ^2)$', fontsize=14) 
plt.ylabel(r'$\ln(QFI)$', fontsize=14)
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)
plt.show()

#plt.scatter(a_sq_list_epsilon, qfi_list_epsilon)
print('This took ' + str(time.time() - start) + ' seconds.')
