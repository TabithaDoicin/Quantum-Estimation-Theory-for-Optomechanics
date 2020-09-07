# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:28:44 2020

@author: Anton
"""

import numpy as np
import matplotlib.pyplot as plt
import wrapper as w
import time
import main_script as m
from scipy import linalg

i = complex(0,1)

wm = 1.1e7
gm = 32
k = 2e5
d0 = 1.1e7
g0_list = np.linspace(200,200.01,2)
g2_list = np.linspace(1e-5,1.01e-5,2)
n = 237.54
min_ep = 3.88e9
max_ep = 1.64e11
ep_list = np.linspace(1e8,1e8,1)


r0, cov0, A0, B0, C0= m.efficient_solver(wm, gm, k, d0, g0_list[0], ep_list[0], g2_list[0], n)



A0k = [[-100000,1.099999939899912e7,33.05507335791648,0],
      [-1.099999939899912e7,-100000,-3636.05787070953,0],
      [0,0,-16,1.1e7],
      [-3636.05787070953,-33.05507335791648,-1.10000165275597e7,-16]]
B0k = np.transpose(A0k)
cov0k = [[0.5256015658030877,0.0002327221709910232,0.00643892909927377,1.408261365806591],
         [0.0002327221709910232,0.5256121578719648,-1.408437299495501,0.01941710364010985],
         [0.00643892909927377,-1.408437299495501,78.00376089999515,-0.0002327799841454618],
         [1.408261365806591,0.01941710364010985,-0.0002327799841454618,78.00387599642127]]
C0k = C0 #they are the same, basically

print(cov0-cov0k)