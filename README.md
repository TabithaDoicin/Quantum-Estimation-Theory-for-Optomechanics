# Optomechanics

Python toolbox implementation of QET(Quantum Estimation Theory) for a driven dissipative optomechanical system. Theoretical discussion and foundation for the code can be found in the paper: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.033508

Features:

-Finds average position, momentum, of light and matter in the system.

-Finds steady state covariance matrix for specific or ranges of g0 and g2*, in addition to epsilon, given some model parameters.

-Finds QFI matrix for specific or range of epsilon, again given some model parameters.

*Most cases will only require 2 values of g0 and g2, i.e. just enough variation to get a numerical derivative for the covariance matrix (derivatives of covmat are used in finding QFI).

Ver = Python3.7

Dependencies: Numpy, Scipy, matplotlib(for easy plotting)

Installation:

Download all python files in the repository and put them in your working directory. Importing the wrapper is required.

To see the simplest example of how to use the code to find QFI values over a range of epsilon, take a look at 'test_use_case.py'

-Discussion of parameters and  how they relate to the paper:

wm - omega_m - frequency of mechanical oscillator

gm - gamma_m - mechanical damping rate of mechanical oscillator

k - kappa - cavity damping

d0 - delta_0 - detuning between frequencies

n - nonlinear measure of temperature

g0 and g2 - parameters we are trying to estimate. more info in paper

-Functions:

Wrapper:

find_cov(wm, gm, k, d0, n, ep, g0_list, g2_list) - finds covariance matrices given parameters and outputs them in 2d array of matrices. out_shape = [length(g0_list),length(g2_list)]

find_r(wm, gm, k, d0, n, ep, g0_list, g2_list) - finds vectors [p0,q0,P1,X1] given parameters and outputs them in 2d array of vectors.

find_x(wm, gm, k, d0, n, ep, g0_list, g2_list) - finds 2d array of X1's.

prep_qfi_no_class(wm, gm, k, d0, n, ep, g0_list, g2_list) - finds qfi efficiently given parameters.

multi_qfi(r_arr, cov_arr, g0_list, g2_list) - given 2d arrays of r, and cov, finds qfi matrix.

find_alpha_and_qfi_over_ep(wm, gm, k, d0, n, ep_list, g0_list, g2_list) - wraps finding qfi matrix for multiple epsilons. Outputs a 1d array of QFI matrices of length length(ep_list)

get_qfi_elem_from_arr(qfi_array, elem) - used to nicely pull out specific QFI elements from above function output array.

rel_error(qfi_arr, g, wm) - used to find relative error from a list of qfi values.

temp_to_n(temp,wm) - temperature(in K) to n given wm

