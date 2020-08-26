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

def unarray(arr):#probably the best part of this monster tbqh
    return np.array([e.tolist() for e in arr.flatten()]).reshape(arr.shape[0],arr.shape[1],-1)

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
            r_array[l,m] = temp_obj.solve_r()
    return r_array

def find_x(wm, gm, k, d0, n, ep, g0_list, g2_list):
    x_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_obj = main_script.Little_r(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            x_array[l,m] = temp_obj.solve_x()
    return x_array

def prep_qfi_w_class(wm, gm, k, d0, n, ep, g0_list, g2_list):
    r_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    cov_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    alpha_squared_array = np.zeros([len(g2_list),len(g0_list)])
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            temp_obj = main_script.Little_r(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n)
            cov_array[l,m] = temp_obj.solve_for_cov_from_initial()
            r_array[l,m] = temp_obj.r
            alpha_squared_array[l,m] = 0.5*(np.real(r_array[l,m][2])**2 + np.real(r_array[l,m][3])**2)
    return r_array, cov_array, alpha_squared_array

def prep_qfi_no_class(wm, gm, k, d0, n, ep, g0_list, g2_list, testmode = False):
    r_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    cov_array = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    alpha_squared_array = np.zeros([len(g2_list),len(g0_list)])
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            r_array[l,m], cov_array[l,m] = main_script.efficient_solver(wm, gm, k, d0, g0_list[m], ep, g2_list[l], n, checks = testmode)
            alpha_squared_array[l,m] = 0.5*(np.real(r_array[l,m][2])**2 + np.real(r_array[l,m][3])**2)
    return r_array, cov_array, alpha_squared_array

def prep_qfi(wm, gm, k, d0, n, ep, g0_list, g2_list, use_class = False, testmode = False):
    if use_class == False:
        return prep_qfi_no_class(wm, gm, k, d0, n, ep, g0_list, g2_list, testmode)
    else:
        return prep_qfi_w_class(wm, gm, k, d0, n, ep, g0_list, g2_list)

def single_qfi(r_arr, cov_arr, g0_list):
    r_diff_arr = np.gradient(unarray(r_arr),g0_list[1]-g0_list[0],axis=1)
    cov_diff_arr = np.gradient(unarray(cov_arr),g0_list[1]-g0_list[0],axis=1)
    qfi_output_arr = np.zeros([len(g0_list)], np.float64)
    for ii in range(len(g0_list)-1):
        temp_cov = cov_arr[0,ii]
        temp_L_cov = np.kron(temp_cov, temp_cov)
        middle_bit = np.linalg.pinv(4*temp_L_cov + L_w)
        part_a = np.dot(r_diff_arr[ii], np.dot(np.linalg.pinv(temp_cov), r_diff_arr[ii]))
        part_b = 2*np.dot(np.ravel(cov_diff_arr[ii]),np.matmul(middle_bit,np.ravel(cov_diff_arr[ii])))
        qfi_output_arr[ii] = part_a + part_b
    return qfi_output_arr

def multi_qfi(r_arr, cov_arr, g0_list, g2_list):#bigger and better version of what is above, works for g2 
    qfi_output_arr = np.zeros([len(g2_list),len(g0_list)],np.ndarray)
    r_diff_arr_1 = np.gradient(unarray(r_arr),g0_list[1]-g0_list[0],axis=1)#need to maybe implement this across the board
    r_diff_arr_2 = np.gradient(unarray(r_arr),g2_list[1]-g2_list[0],axis=0)
    cov_diff_arr_1 = np.gradient(unarray(cov_arr),g0_list[1]-g0_list[0],axis=1)
    cov_diff_arr_2 = np.gradient(unarray(cov_arr),g2_list[1]-g2_list[0],axis=0)
    for l in range(len(g2_list)):
        for m in range(len(g0_list)):
            part_a = np.zeros([2,2], dtype = np.float64)
            part_b = np.zeros([2,2], dtype = np.float64)
            temp_cov = cov_arr[l,m]
            temp_L_cov = np.kron(temp_cov,temp_cov)
            middle_bit = np.linalg.pinv(4*temp_L_cov + L_w)
            #g0g0
            part_a[0,0] = np.dot(r_diff_arr_1[l][m], np.dot(np.linalg.pinv(temp_cov), r_diff_arr_1[l][m]))
            part_b[0,0] = 2*np.dot(np.ravel(cov_diff_arr_1[l][m]),np.matmul(middle_bit,np.ravel(cov_diff_arr_1[l][m])))
            #g0g2
            part_a[0,1] = np.dot(r_diff_arr_1[l][m], np.dot(np.linalg.pinv(temp_cov), r_diff_arr_2[l][m]))#these arent symmetrical yet
            part_b[0,1] = 2*np.dot(np.ravel(cov_diff_arr_1[l][m]),np.matmul(middle_bit,np.ravel(cov_diff_arr_2[l][m])))
            #g2g0
            part_a[1,0] = np.dot(r_diff_arr_2[l][m], np.dot(np.linalg.pinv(temp_cov), r_diff_arr_1[l][m]))
            part_b[1,0] = 2*np.dot(np.ravel(cov_diff_arr_2[l][m]),np.matmul(middle_bit,np.ravel(cov_diff_arr_1[l][m])))
            #g2g2
            part_a[1,1] = np.dot(r_diff_arr_2[l][m], np.dot(np.linalg.pinv(temp_cov), r_diff_arr_2[l][m]))
            part_b[1,1] = 2*np.dot(np.ravel(cov_diff_arr_2[l][m]),np.matmul(middle_bit,np.ravel(cov_diff_arr_2[l][m])))
            qfi_output_arr[l,m] = np.array([[part_a[0,0] + part_b[0,0], part_a[0,1] + part_b[0,1]],
                                              [part_a[1,0] + part_b[1,0], part_a[1,1] + part_b[1,1]]], dtype=np.float64)
    return qfi_output_arr

def find_alpha_and_qfi_over_ep(wm, gm, k, d0, n, ep_list, g0_list, g2_list):
    r_arr_arr = np.zeros([len(ep_list)],np.ndarray)
    cov_arr_arr = np.zeros([len(ep_list)],np.ndarray)
    a_sq_arr = np.zeros([len(ep_list)],np.ndarray)
    qfi_output_arr = np.zeros([len(ep_list)],np.ndarray)
    for ii in range(len(ep_list)):#iterates over epsilons, ...arr_arr's are over epsilons, i dont know how to explain it but it *definitely* works
        r_arr_arr[ii], cov_arr_arr[ii], a_sq_arr[ii] = prep_qfi(wm, gm, k, d0, n, ep_list[ii], g0_list, g2_list, use_class = False)
        qfi_output_arr[ii] = multi_qfi(r_arr_arr[ii], cov_arr_arr[ii], g0_list, g2_list)[0,0] #which deri
    a_sq_list_epsilon = np.zeros([len(ep_list)], np.float64)
    qfi_list_epsilon = np.zeros([len(ep_list)], np.ndarray)
    for ii in range(len(ep_list)):
        a_sq_list_epsilon[ii] = a_sq_arr[ii][0][0] #because this outputs for g2 too, [][] needed, all these are close together unless misuse, 
        #like which derivative am i taking? should only be an issue if you make g2's and g0's too far from each other but then you get accuracy drop anyway
        qfi_list_epsilon[ii] = wm**2 * qfi_output_arr[ii]
    return a_sq_list_epsilon, qfi_list_epsilon

def get_qfi_elem_from_arr(qfi_array, elem): #cbb to write this out everytime, just made a function for it
    res = np.zeros([qfi_array.size])
    for ii in range(qfi_array.size):
       res[ii] = qfi_array[ii][elem[0], elem[1]] 
    return res

def rel_error(qfi_arr, g0_array):
    rel_arr = np.zeros(len(g0_array))
    for ii in range(len(g0_array)):
        rel_arr[ii] = qfi_arr[ii]**-0.5 * g0_array[ii]**-1
    return rel_arr