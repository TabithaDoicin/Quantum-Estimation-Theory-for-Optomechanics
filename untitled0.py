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

i = complex(0,1)
W = np.array([[0,i,0,0],
              [-1*i,0,0,0],
              [0,0,0,i],
              [0,0,-1*i,0]])
wm = 1.1e7
gm = 32
k = 2e5
d0 = 1.1e7
g0_list = np.linspace(200,1000,100)
g2_list = np.linspace(0,1,100)
n = 237.5
ep = 10000
##
start = time.time()
##
print(w.find_cov(wm, gm, k, d0, n, ep, g0_list, g2_list))
##
print('took ' + str(time.time()-start) + ' seconds.')
##