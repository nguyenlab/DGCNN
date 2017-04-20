import numpy as np
import copy

def dummyReLU(dummy, x):
    return x*(x>0)

def dummyReLUPrime(dummy, z):
    return np.array(z != 0, dtype=np.float)
    
############################################
# Identity

def dummyIdentity(x):
    return x
def dummyEye(dummy, x):
    return np.ones_like(x)

############################################
# sigmoid

def dummySigmoid(dummy, x):
    #e = 2.718281828459045

    return 1/ (1 + np.e ** (-x) )

def sigmoid(x):
    return 1 / (1 + np.e ** -x )
def sigmoidPrime(y):
    return y*(1-y)
def dummySigmoidPrime(dummy, y):

    return y * (1 - y)
############################################
# hyperbolic tangent

def dummyTanh(dummy, x):
    return 2 / (1+np.e**(-x)) - 1
    
def dummyTanhPrime(dummy, y):
    ysigmoid = (y+1.0)/2
    dsigmoid = ysigmoid * ( 1- ysigmoid)
    return 2 * dsigmoid
    
############################################
# softmax
# N.B. softmax is often used in the output layer
# partial J w.r.t. z = y - t

def softmax(z):
    maxbycol = np.max(z, axis = 0) 
    z = z - maxbycol # extract the maximal value for each dataset to prevent numerical overflow
    z = np.e**z
    sumbycol = np.sum(z, axis = 0)
    
    return z / sumbycol

def dummySoftMax( z):
    # x: <nUnits> by <nData>
    maxbycol = np.max(z, axis = 0) 
    z = z - maxbycol # extract the maximal value for each dataset to prevent numerical overflow
    z = np.e**z
    sumbycol = np.sum(z, axis = 0)
    
    return z / sumbycol

#########
# test 
'''
x = .9
delta = .005
xplus = x + delta
xminus = x - delta

numericalGradient = (dummyTanh(0, xplus) - dummyTanh(0, xminus) ) / 2/ delta
y = dummyTanh(0, x)
analyticalGradient = dummyTanhPrime(0, y)
print numericalGradient, analyticalGradient
'''
#
