# Optomechanics

Python toolbox implementation of QET(Quantum Estimation Theory) for a driven dissipative optomechanical system. Theoretical discussion and foundation for the code can be found in the paper:

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

