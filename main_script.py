# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:27:36 2020

@author: Tabitha
"""

#!usr/bin/python

import numpy as np
from scipy import linalg

##Constants
i = complex(0,1) #for simplicity in writing later
W = np.array([[0,i,0,0],
              [-1*i,0,0,0],
              [0,0,0,i],
              [0,0,-1*i,0]])
hb = 1.0545718e-34

class Sim: #For outputting graphs, this isnt exactly required, but ive left it in case it serves a niche purpose later, 
#i.e. having an object for each set of base parameters seemed like a good idea at the start and still might be
    
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
    
    def solve_r(self, x = None, recalib = True): #uses the chosen root to return [x0,p0,x1,p1]
        if x is None:
            x = self.solve_x()
        else:
            pass
        self.r = np.zeros([4], dtype = np.complex128)
        self.r[2] = x #X0
        self.deff = self.d0 - 2**0.5 * self.g0 * x + self.g2 * x**2
        if recalib == True: 
            self.d0 = self.wm
            self.w_eff = self.wm + (self.wm**2 + 0.25*self.k**2)**-1 * 2*self.g2*self.ep**2
            self.g_eff = 2*self.g2*x - 2**0.5 * self.g0
        else:
            pass
        self.r[0] = (0.5*self.gm * (self.wm)**-1 * x) #p0
        self.r[1] = (2**-0.5 * -2 * self.deff * self.ep * (self.deff**2 + self.k**2 * 0.25)**-1) #x1 (q0)
        self.r[3] = (2**-0.5 * -1 * self.k * self.ep * (self.deff**2 + self.k**2 * 0.25)**-1) # p1 (P1)
        return self.r

    def solve_cov(self, r):
        self.gamma = 0.5*np.array([[self.k,-self.k*i,0,0],
                                   [self.k*i,self.k,0,0],
                                   [0,0,self.gm*(2*self.n+1),-1*i*self.gm],
                                   [0,0,self.gm*i,self.gm*(2*self.n+1)]])
        self.gamma_A = 0.5*(self.gamma-np.transpose(self.gamma))
        self.gamma_S = 0.5*(self.gamma+np.transpose(self.gamma))
        self.H = np.array([[hb * self.deff, 0, hb * self.r[0] * self.g_eff, 0],
                      [0, hb * self.deff, hb * self.r[1] * self.g_eff, 0],
                      [hb*self.r[0]*self.g_eff, hb*self.g_eff*self.r[1], hb*self.w_eff, 0],
                      [0,0,0, hb*self.wm]])
        self.A = -1*i/hb*np.dot(W,self.H) + np.dot(W,self.gamma_A)
        self.C = -1*np.dot(W,np.dot(self.gamma_S,W))
        self.cov = linalg.solve_continuous_lyapunov(self.A,self.C)
        return self.cov

    def solve_for_cov_from_initial(self):
        self.solve_x()
        return(self.solve_cov(self.solve_r(self.root)))
    
def efficient_solver(wm, gm, k, d0, g0, ep, g2, n, recalib = True, checks = False):
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
    r[2] = roots_x0[len(roots_x0)-1] #last root in the array given is always the real root, also this is x0
    x = r[2]#as far as i know this utilises data casting, so no extra memory use or speed loss
    ##finding p0,x1,p1, and putting it all into a numpy array##
    r[3] = (0.5*gm * (wm)**-1 * x) #p0
    ##
    ##recalibration if needed-idk about how this works but ill implement for the moment and turn it off##
    d_eff = wm + 2**0.5 * g0 * x - g2 * x**2
    if recalib == True: 
        d0 = wm
        w_eff = wm + (wm**2 + 0.25*k**2)**-1 * 2*g2*ep**2
        g_eff = 2*g2*x - 2**0.5 * g0
    else:
        pass
    ##
    r[0] = (2**-0.5 * -2 * d_eff * ep * (d_eff**2 + k**2 * 0.25)**-1) #x1 (Q0)
    r[1] = (2**-0.5 * -1 * k * ep * (d_eff**2 + k**2 * 0.25)**-1) # p1 (P1)
    ##solving covariance matrix
    H = np.array([[hb * d_eff, 0, hb * r[0] * g_eff, 0],
                 [0, hb * d_eff, hb * r[1] * g_eff, 0],
                 [hb*r[0]*g_eff, hb*g_eff*r[1], hb*w_eff, 0],
                 [0,0,0, hb*wm]])
    gamma = 0.5*np.array([[k,-k*i,0,0],
                          [k*i,k,0,0],
                          [0,0,gm*(2*n+1),-1*i*gm],
                          [0,0,gm*i,gm*(2*n+1)]])
    gamma_A = 0.5*(gamma-np.transpose(gamma))
    gamma_S = 0.5*(gamma+np.transpose(gamma))
    A = -1*i/hb*np.dot(W,H) + np.dot(W,gamma_A)
    C = -1*np.dot(W,np.dot(gamma_S,W))
    cov = linalg.solve_continuous_lyapunov(A,C)
    if checks == True:
        rs_test = 2*cov + W
        print('RS-Matrix-eigenvalues:')
        print(np.linalg.eig(rs_test)[0])
        print('B-Matrix-eigenvalues:')
        print(np.linalg.eig(np.transpose(A))[0])
    else:
        pass
    return np.real(r), np.real(cov)
