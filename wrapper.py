# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:17:13 2020

@author: Anton
"""

#!usr/bin/python

import numpy as np
from numpy import linalg
import main_script
from timeit import default_timer as timer 
from numba import jit
import blockwiseview


def find_cov(wm, gm, k, d0, n, ep, g0_list, g2_list):
    cov_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_obj = main_script.Little_r(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            cov_array[l,m] = temp_obj.solve_for_cov_from_initial()
    return cov_array

def find_r(wm, gm, k, d0, n, ep, g0_list, g2_list):
    r_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_obj = main_script.Little_r(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            r_array[l,m] = temp_obj.roots_x0
    return r_array

def find_x(wm, gm, k, d0, n, ep, g0_list, g2_list):
    x_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_obj = main_script.Little_r(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            x_array[l,m] = temp_obj.solve_x()
    return x_array

def prep_qfi(wm, gm, k, d0, n, ep, g0_list, g2_list):
    r_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    cov_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_obj = main_script.Little_r(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            cov_array[l,m] = temp_obj.solve_for_cov_from_initial()
            r_array[l,m] = temp_obj.r
    return r_array, cov_array

def single_qfi(r_arr, cov_arr, g0_list):
    i = complex(0,1)
    W = np.array([[0,i,0,0],
              [-1*i,0,0,0],
              [0,0,0,i],
              [0,0,-1*i,0]])
    L_w = np.kron(W,W)
    r_diff_arr = np.gradient([r_arr[0,:][ii] for ii in range(len(g0_list))], g0_list[1]-g0_list[0], axis=0)
    cov_diff_arr = np.gradient([cov_arr[0,:][ii] for ii in range(len(g0_list))], g0_list[1]-g0_list[0], axis=0)
    qfi_output_arr = np.zeros([len(g0_list)], np.complex128)
    for ii in range(len(g0_list)-1):
        temp_cov = cov_arr[0,:][ii]
        temp_L_cov = np.kron(temp_cov, temp_cov)
        middle_bit = np.linalg.pinv(4*temp_L_cov + L_w)
        part_a = np.dot(r_diff_arr[ii], np.dot(np.linalg.pinv(temp_cov), r_diff_arr[ii]))
        print(middle_bit.shape)
        print(temp_cov.shape)
        part_b = 2*np.trace(np.matmul(cov_diff_arr[ii],np.matmul(middle_bit,cov_diff_arr[ii])))
        qfi_output_arr[ii] = part_a + part_b
    return qfi_output_arr