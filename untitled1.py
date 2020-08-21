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
import warnings
warnings.filterwarnings('ignore')
i = complex(0,1)

wm = 1.1e7
gm = 32
k = 2e5
d0 = 1.1e7
g0_list = np.linspace(200,300,2)
g2_list = np.linspace(0,100,2)
n = 237.54
ep = 6e8



print('derivative TESTING:::')


