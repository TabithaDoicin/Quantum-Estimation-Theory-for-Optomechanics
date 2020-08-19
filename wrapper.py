# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:17:13 2020

@author: Anton
"""

#!usr/bin/python

import numpy as np
import main_script

i = complex(0,1)
W = np.array([[0,i,0,0],
              [-1*i,0,0,0],
              [0,0,0,i],
              [0,0,-1*i,0]])
L_w = np.kron(W,W)

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

def prep_qfi_efficient(wm, gm, k, d0, n, ep, g0_list, g2_list):
    r_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    cov_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    alpha_squared_array = np.zeros([len(g2_list),len(g0_list)])
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            r_array[l,m], cov_array[l,m] = main_script.efficient_solver(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            alpha_squared_array[l,m] = 0.5*(np.real(r_array[l,m][2])**2 + np.real(r_array[l,m][3])**2)
    return r_array.copy(), cov_array.copy(), alpha_squared_array.copy()

def single_qfi(r_arr, cov_arr, g0_list):
    r_diff_arr = np.gradient([r_arr[0,:][ii] for ii in range(len(g0_list))], g0_list[1]-g0_list[0], axis=0)
    cov_diff_arr = np.gradient([cov_arr[0,:][ii] for ii in range(len(g0_list))], g0_list[1]-g0_list[0], axis=0)
    qfi_output_arr = np.zeros([len(g0_list)], np.float64)
    for ii in range(len(g0_list)-1):
        temp_cov = cov_arr[0,ii]
        temp_L_cov = np.kron(temp_cov, temp_cov)
        middle_bit = np.linalg.pinv(4*temp_L_cov + L_w)
        part_a = np.dot(r_diff_arr[ii], np.dot(np.linalg.pinv(temp_cov), r_diff_arr[ii]))
        part_b = 2*np.dot(np.ravel(cov_diff_arr[ii]),np.matmul(middle_bit,np.ravel(cov_diff_arr[ii])))
        qfi_output_arr[ii] = part_a + part_b
    return qfi_output_arr

def multi_qfi(r_arr, cov_arr, g0_list, g2_list):
    qfi_output_arr = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    r_diff_arr_1 = np.zeros([len(g2_list)],np.ndarray)
    r_diff_arr_2 = np.zeros([len(g0_list)],np.ndarray)
    cov_diff_arr_1 = np.zeros([len(g2_list)],np.ndarray)
    cov_diff_arr_2 = np.zeros([len(g0_list)],np.ndarray)
    for k in range(len(g2_list-1)): #g0 derivatives
        r_diff_arr_1[k] = np.gradient([r_arr[k,:][ii] for ii in range(len(g0_list))], g0_list[1]-g0_list[0], axis=0)
        cov_diff_arr_1[k] = np.gradient([cov_arr[k,:][ii] for ii in range(len(g0_list))], g0_list[1]-g0_list[0], axis=0)
    for k in range(len(g0_list-1)): #g2 derivatives
        r_diff_arr_2[k] = np.gradient([r_arr[:,k][ii] for ii in range(len(g2_list))], g2_list[1]-g2_list[0], axis=0)
        cov_diff_arr_2[k] = np.gradient([cov_arr[:,k][ii] for ii in range(len(g2_list))], g2_list[1]-g2_list[0], axis=0)
    part_a = np.zeros([2,2], dtype = np.float64)
    part_b = np.zeros([2,2], dtype = np.float64)
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_cov = cov_arr[l,m]
            temp_L_cov = np.kron(temp_cov,temp_cov)
            middle_bit = np.linalg.pinv(4*temp_L_cov + L_w)
            #g0g0
            part_a[0,0] = np.dot(r_diff_arr_1[l][m], np.dot(np.linalg.pinv(temp_cov), r_diff_arr_1[l][m]))
            part_b[0,0] = 2*np.dot(np.ravel(cov_diff_arr_1[l][m]),np.matmul(middle_bit,np.ravel(cov_diff_arr_1[l][m])))
            #g0g2?
            part_a[0,1] = np.dot(r_diff_arr_1[l][m], np.dot(np.linalg.pinv(temp_cov), r_diff_arr_2[m][l]))
            part_b[0,1] = 2*np.dot(np.ravel(cov_diff_arr_1[l][m]),np.matmul(middle_bit,np.ravel(cov_diff_arr_2[m][l])))
            #g2g0? might have to switch idk
            part_a[1,0] = np.dot(r_diff_arr_2[m][l], np.dot(np.linalg.pinv(temp_cov), r_diff_arr_1[l][m]))
            part_b[1,0] = 2*np.dot(np.ravel(cov_diff_arr_2[m][l]),np.matmul(middle_bit,np.ravel(cov_diff_arr_1[l][m])))
            #g2g2
            part_a[1,1] = np.dot(r_diff_arr_2[m][l], np.dot(np.linalg.pinv(temp_cov), r_diff_arr_2[m][l]))
            part_b[1,1] = 2*np.dot(np.ravel(cov_diff_arr_2[m][l]),np.matmul(middle_bit,np.ravel(cov_diff_arr_2[m][l])))
            qfi_output_arr[l,m] = np.array([[part_a[0,0] + part_b[0,0], part_a[0,1] + part_b[0,1]],
                                              [part_a[1,0] + part_b[1,0], part_a[1,1] + part_b[1,1]]], dtype=np.float64)
    return qfi_output_arr

def rel_error(qfi_arr, g0_array):
    rel_arr = np.zeros(len(g0_array))
    for ii in range(len(g0_array)):
        rel_arr[ii] = qfi_arr[ii]**-0.5 * g0_array[ii]**-1
    return rel_arr