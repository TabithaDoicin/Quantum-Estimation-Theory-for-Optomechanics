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

i = complex(0,1)
W = np.array([[0,i,0,0],
              [-1*i,0,0,0],
              [0,0,0,i],
              [0,0,-1*i,0]])
wm = 1.1e7
gm = 32
k = 2e5
d0 = 1.1e7
g0_list = np.linspace(200,200,1)
n = 237.5
#for finding suitable ranges of epsilon, no value of epsilon given, will be finding ranges for different g0's
list_of_epsilon_trials = []
lower_ep = []
upper_ep = []
discriminant_value_list = []
for l in range(len(g0_list)):
    list_of_epsilon_trials.append(m.Little_r(wm,gm,k,d0,g0_list[l]))
    temp = list_of_epsilon_trials[l].range_of_epsilon(eval_dis = True)
    discriminant_value_list.append(list_of_epsilon_trials[l].dis_val)
    lower_ep.append(temp[0])
    upper_ep.append(temp[1])
    
plt.figure(1)
plt.scatter(g0_list, lower_ep)
plt.scatter(g0_list, upper_ep)
print(lower_ep)
ep = 6e8 #middle_ep from g0 = 1000, works for all g0's
#ep = 8889047118
list_of_x0_trials = []
x0_list = []
#now finding x0 for the different g0's
for l in range(len(g0_list)):
    list_of_x0_trials.append(m.Little_r(wm,gm,k,d0,g0_list[l],ep,0,237.5))
    x0_list.append(list_of_x0_trials[l].solve_x())
#print(x0_list)

plt.figure(2)
x0_list_1 = []
x0_list_2 = []
x0_list_3 = []
for l in range(len(x0_list)):
    x0_list_1.append(x0_list[l][2])
    x0_list_2.append(x0_list[l][1])
    x0_list_3.append(x0_list[l][0])
print(x0_list_1)
plt.scatter(g0_list, x0_list_1)
plt.scatter(g0_list, x0_list_2)
plt.scatter(g0_list, x0_list_3)

r_list = []
for l in range(len(x0_list_1)):
    r_list.append(list_of_x0_trials[l].solve_r(x = x0_list_1[l]))
print(r_list)

cov_list = []
for l in range(len(x0_list_1)):
    cov_list.append(list_of_x0_trials[l].solve_cov(r = r_list[l]))
print(np.around(cov_list[0],5))

#print(2*cov_list[0] + W)