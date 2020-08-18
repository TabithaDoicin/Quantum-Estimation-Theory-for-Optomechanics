# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:27:36 2020

@author: Anton
"""

#!usr/bin/python

import numpy as np
from scipy import linalg
from numba import jit

##Constants
i = complex(0,1) #for simplicity in writing later
W = np.array([[0,i,0,0],
              [-1*i,0,0,0],
              [0,0,0,i],
              [0,0,-1*i,0]])
hb = 1.0545718e-34

class Little_r: #Everything about finding epsilon bounds and X0 for a specific g0
    
    def __init__(self, wm, gm, k, d0, g0, ep=0, g2=0, n=0):
        self.wm = wm #omega_m
        self.gm = gm #gamma_m
        self.k = k #kappa
        self.d0 = d0 #del_0
        self.g0 = g0 #self-explanatory
        self.ep = ep
        self.g2 = g2
        self.n = n
    
    def range_of_epsilon(self, eval_dis = False): #function that finds epsilon bounds and also value of discriminant between epsilon values
        if self.g2 != 0:
            print('Cannot find determinant due to quintic.')
            pass
        elif self.g0 == 0:
            print('g0 cannot be 0, choose a different range.')
            pass
        else:
            a = 2 * self.g0**2 * (self.wm**2 + 0.25 * self.gm**2) #a
            b = -2 * 2**0.5 * self.g0 * self.d0 * (self.wm**2 + 0.25 * self.gm**2) #b
            c = (self.wm**2 + 0.25 * self.gm**2) * (self.d0**2 + 0.25*self.k**2) #c
            d = -2**0.5 * self.g0 * self.wm #really d/epsilon^2
                
            self.roots_of_epsilon = np.roots([-27*a**2*d**2, 0, 
                                              18*a*b*c*d-4*b**3*d, 
                                              0, b**2*c**2-4*a*c**3])
            ##important_ouputs
            self.r_of_e_sorted = np.sort(self.roots_of_epsilon) #discriminant=0
            if eval_dis == True: #outputs the value of the discriminant between epsilon roots if set true
                def discriminant_value(ep):
                    return 18*a*b*c*d*ep**2 - 4*b**3*d*ep**2 + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2*ep**4
                self.dis_val = []
                for l in range(len(self.r_of_e_sorted)+1):
                    if l == 0:
                        val = discriminant_value(self.r_of_e_sorted[0] - 1000)
                    elif l == 1 or l == 2 or l == 3:
                        val = discriminant_value((self.r_of_e_sorted[l-1] + self.r_of_e_sorted[l])/2)
                    elif l == 4:
                        val = discriminant_value(self.r_of_e_sorted[0] + 1000)
                    self.dis_val.append(val)
            return [self.r_of_e_sorted[2], self.r_of_e_sorted[3]]
    
    def solve_x(self): #finds all roots of polynomial for x
        a = 0.25 * self.g2**2 * (self.gm**2 + 4 * self.wm**2)
        b = -((self.g0*self.g2*(self.gm**2 + 4*self.wm**2))/2**0.5)
        c = (1/2)*self.g0**2*(self.gm**2 + 4*self.wm**2) + (1/2)*self.d0*self.g2*(self.gm**2 + 4*self.wm**2)
        d = -((self.d0*self.g0*(self.gm**2 + 4*self.wm**2))/2**0.5)
        e = 2*self.ep**2*self.g2*self.wm + (1/4)*self.d0**2*(self.gm**2 + 4*self.wm**2) + (1/16)*self.k**2*(self.gm**2 + 4*self.wm**2)
        f = (-2**0.5)*self.ep**2*self.g0*self.wm
        self.roots_x0 = np.roots([a,b,c,d,e,f])#numpy roots function
        self.root = self.roots_x0[len(self.roots_x0)-1] #last root in the array given is always the real root
        return self.root
    
    def solve_r(self, x): #uses the chosen root to return [x0,p0,x1,p1]
        def d_eff(y): #delta_eff
            return (self.d0 - 2**0.5 * self.g0 * y + self.g2 * y**2)
        self.r = np.zeros([4], dtype = np.complex128)
        self.r[0] = x #X0
        self.r[1] = (0.5*self.gm * (self.wm)**-1 * x) #p0
        self.r[2] = (2**-0.5 * -2 * d_eff(x) * self.ep * (d_eff(x)**2 + self.k**2 * 0.25)**-1) #x1 (Q0)
        self.r[3] = (2**-0.5 * -1 * self.k * self.ep * (d_eff(x)**2 + self.k**2 * 0.25)**-1) # p1 (P1)
        return self.r

    def solve_cov(self, r):
        def d_eff(y): #delta_eff
            return (self.d0 - 2**0.5 * self.g0 * y + self.g2 * y**2)
        self.deff = d_eff(self.r[0])
        self.gamma = 0.5*np.array([[self.k,-self.k*i,0,0],
             [self.k*i,self.k,0,0],
             [0,0,self.gm*(2*self.n+1),-1*i*self.gm],
             [0,0,self.gm*i,self.gm*(2*self.n+1)]])
        self.gamma_A = 0.5*(self.gamma-np.transpose(self.gamma))
        self.gamma_S = 0.5*(self.gamma+np.transpose(self.gamma))
        self.H = np.array([[hb * d_eff(self.r[0]), 0, -2**0.5 * self.g0 * hb * self.r[2], 0],
                           [0, hb * d_eff(self.r[0]), -2**0.5 * self.g0 * hb * self.r[3], 0],
                           [-2**0.5 * self.g0 * hb * self.r[2], -2**0.5 * self.g0 * hb * self.r[3], hb*self.wm, 0],
                           [0,0,0, hb*self.wm]])
        self.A = -1*i/hb*np.dot(W,self.H) + np.dot(W,self.gamma_A)
        self.C = -1*np.dot(W,np.dot(self.gamma_S,W))
        self.cov = linalg.solve_continuous_lyapunov(self.A,self.C)
        return self.cov

    def solve_for_cov_from_initial(self):
        self.solve_x()
        return(self.solve_cov(self.solve_r(self.root)))
    
#@jit(nopython = False, fastmath=True, nogil=True, cache=True)
def efficient_solver(wm, gm, k, d0, g0, ep, g2, n, recalib = False, checks = False):
    ##initialisation of some data arrays for speed##
    r = np.zeros((4), dtype = np.complex128)
    ##finding x0##
    a = 0.25 * g2**2 * (gm**2 + 4 * wm**2)
    b = -((g0*g2*(gm**2 + 4*wm**2))/2**0.5)
    c = (1/2)*g0**2*(gm**2 + 4*wm**2) + (1/2)*d0*g2*(gm**2 + 4*wm**2)
    d = -((d0*g0*(gm**2 + 4*wm**2))/2**0.5)
    e = 2*ep**2*g2*wm + (1/4)*d0**2*(gm**2 + 4*wm**2) + (1/16)*k**2*(gm**2 + 4*wm**2)
    f = (-2**0.5)*ep**2*g0*wm
    roots_x0 = np.roots(np.array([a,b,c,d,e,f], dtype = np.complex128))#numpy roots function
    r[0] = roots_x0[len(roots_x0)-1] #last root in the array given is always the real root, also this is x0
    x = r[0]#as far as i know this utilises data casting, so no extra memory use or speed loss
    ##finding p0,x1,p1, and putting it all into a numpy array##
    r[1] = (0.5*gm * (wm)**-1 * x) #p0
    ##
    d_eff = d0 - 2**0.5 * g0 *x + g2 * x**2
    ##recalibration if needed-idk about how this works but ill implement for the moment and turn it off##
    if recalib == True: #only for g2=0 at the moment, need to get recalibrations from kamila sometime
        d0 = d_eff + 2**0.5 * g0 * x
        d_eff = wm
    else:
        pass
    ##
    r[2] = (2**-0.5 * -2 * d_eff * ep * (d_eff**2 + k**2 * 0.25)**-1) #x1 (Q0)
    r[3] = (2**-0.5 * -1 * k * ep * (d_eff**2 + k**2 * 0.25)**-1) # p1 (P1)
    ##solving covariance matrix
    gamma = 0.5*np.array([[k,-k*i,0,0],
             [k*i,k,0,0],
             [0,0,gm*(2*n+1),-1*i*gm],
             [0,0,gm*i,gm*(2*n+1)]])
    gamma_A = 0.5*(gamma-np.transpose(gamma))
    gamma_S = 0.5*(gamma+np.transpose(gamma))
    H = np.array([[hb * d_eff, 0, -2**0.5 * g0 * hb * r[2], 0],
                           [0, hb * d_eff, -2**0.5 * g0 * hb * r[3], 0],
                           [-2**0.5 * g0 * hb * r[2], -2**0.5 * g0 * hb * r[3], hb*wm, 0],
                           [0,0,0, hb*wm]])
    A = -1*i/hb*np.dot(W,H) + np.dot(W,gamma_A)
    C = -1*np.dot(W,np.dot(gamma_S,W))
    cov = linalg.solve_continuous_lyapunov(A,C)
    if checks == True:
        rs_test = 2*cov + W
        print('RS-Matrix:')
        print(np.around(rs_test,5))
        print('Eigenvectors and eigenvalues of B:')
        print(np.linalg.eig(np.transpose(A)))
    return r, cov