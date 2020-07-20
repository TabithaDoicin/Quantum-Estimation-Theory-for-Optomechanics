# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:17:13 2020

@author: Anton
"""

#!usr/bin/python

import numpy as np
import main_script
import blockwiseview
from numba import njit

def block_mult(a, b):
    res = np.zeros([a.shape[0], b.shape[1]],np.ndarray)
    for l in range(a.shape[0]):
        for m in range(b.shape[1]):
            res[l][m] = sum([np.dot(a[l][ii], b[ii][m]) for ii in range(a.shape[1])])
    return res
                
def easyblock(arr):
    return np.block([[arr[0][0],arr[0][1],arr[0][2],arr[0][3]],
                     [arr[1][0],arr[1][1],arr[1][2],arr[1][3]],
                     [arr[2][0],arr[2][1],arr[2][2],arr[2][3]],
                     [arr[3][0],arr[3][1],arr[3][2],arr[3][3]]])

def find_cov(wm, gm, k, d0, n, ep, g0_list, g2_list):
    cov_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_obj = main_script.Little_r(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            cov_array[l,m] = temp_obj.solve_for_cov_from_initial()
    return cov_array.copy()

def find_r(wm, gm, k, d0, n, ep, g0_list, g2_list):
    r_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_obj = main_script.Little_r(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            r_array[l,m] = temp_obj.roots_x0
    return r_array.copy()

def find_x(wm, gm, k, d0, n, ep, g0_list, g2_list):
    x_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_obj = main_script.Little_r(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            x_array[l,m] = temp_obj.solve_x()
    return x_array.copy()

def prep_qfi(wm, gm, k, d0, n, ep, g0_list, g2_list):
    r_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    cov_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    alpha_squared_array = np.zeros([len(g2_list),len(g0_list)])
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_obj = main_script.Little_r(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            cov_array[l,m] = temp_obj.solve_for_cov_from_initial()
            r_array[l,m] = temp_obj.r
            alpha_squared_array[l,m] = 0.5*(np.real(r_array[l,m][2])**2 + np.real(r_array[l,m][3])**2)
    return r_array.copy(), cov_array.copy(), alpha_squared_array.copy()

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
        part_b = 2*np.trace(easyblock(block_mult(cov_diff_arr[ii],block_mult(blockwiseview.blockwise_view(middle_bit,(4,4)),cov_diff_arr[ii]))))
        qfi_output_arr[ii] = part_a + part_b
    return qfi_output_arr

def rel_error(qfi_arr, g0_array):
    rel_arr = np.zeros(len(g0_array))
    for ii in range(len(g0_array)):
        rel_arr[ii] = qfi_arr[ii]**-0.5 * g0_array[ii]**-1
    return rel_arr