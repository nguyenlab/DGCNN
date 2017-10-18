import os, sys, numpy as np, copy

sys.path.append('../')

# from InitTreeConvolution import *
import gl
import cPickle as p
# import test as test
# import FFNN

from InitParam import *
import write_param

sys.setrecursionlimit(100000000)

# tmptokenMap = dict()
# tmpCnt = 0

'''
The folder with suffix '1' contains parameters for experiment B, in which
learning rate is 0.0025 initially, while the ohter folder contains parameters
for experiment A, in which learning rate is 0.005 initially, and now is set to
be 0.001

'''

numDis = gl.numDis
numOut = gl.numOut
numRecur = gl.numCon

numFea = gl.numFea

# numSen = 10
tokenMap = p.load(open(gl.tokenMap))
tokenNum = len(tokenMap)


np.random.seed(314)

preWeights = np.array([])
preBiases = np.array([])
preWeights, preWleft = InitParam(preWeights, num=numFea * numFea)
preWeights, preWright = InitParam(preWeights, num=numFea * numFea)
preBiases, preBtoken = InitParam(preBiases, num=numFea * tokenNum, upper=0.4, lower=0.6)
preBiases, preBconstruct = InitParam(preBiases, num=numFea)

preparam = p.load(open('../preparam'))
preW = preparam[:len(preWeights)]
preB = preparam[len(preWeights):]

preWleft = preW[preWleft]
preWright = preW[preWright]
preBtoken = preB[preBtoken]
preBconstruct = preB[preBconstruct]

#################################
##################################
# Initialize weights

Weights = np.array([])
Biases = np.array([])

# def InitParam(OldWeights, num=None, newWeights=None, upper=None, lower=None):
#     oldlen = len(OldWeights)
#     if newWeights != None:
#         newWeights = np.array(newWeights)
#         num = len(newWeights)
#         OldWeights = np.concatenate((OldWeights, newWeights.reshape(-1)))
#
#     else:
#         if upper == None:
#             upper = 0.02
#             lower = -.02
#         tmpWeights = np.random.uniform(lower, upper, num)
#         OldWeights = np.concatenate((OldWeights, tmpWeights.reshape(-1)))
#     return OldWeights, range(oldlen, oldlen + num)
# word/symbol/token embedding
Biases, BwordIdx = InitParam(Biases, newWeights=preBtoken)
print 'len biases =', len(Biases), 'BwordIdx= ', len(BwordIdx)
# print preW
# Biases, BwordIdx =  InitParam(Biases, len(preBtoken))
# left/right constructing and its biases
Weights, Wleft = InitParam(Weights, newWeights=preWleft)
Weights, Wright = InitParam(Weights, newWeights=preWright)
Biases, Bconstruct = InitParam(Biases, newWeights=preBconstruct)

'''

Biases, BwordIdx = InitParam(Biases, num=numFea*tokenNum, upper=0.4,lower=0.6)

        #Biases, BwordIdx =  InitParam(Biases, len(preBtoken))
        # left/right constructing and its biases
Weights, Wleft  = InitParam(Weights, num=numFea*numFea)
Weights, Wright = InitParam(Weights, num=numFea*numFea)
Biases,Bconstruct=InitParam(Biases,  num =numFea)

'''

# Weights, Wleft = InitParam(Weights,  len(preWleft) )
# Weights, Wright= InitParam(Biases,   len(preWright) )
# Biases, Bconstruct = InitParam(Biases, len(preBconstruct))

# combinition of the encoded vector and the vector of the symbol
w1 = (np.eye(numFea) / 2).reshape(-1)
w2 = (np.eye(numFea) / 2).reshape(-1)

Weights, Wcomb_ae = InitParam(Weights, newWeights=w1)
print 'len W=', len(Weights)
Weights, Wcomb_orig = InitParam(Weights, newWeights=w2)
print 'Begin Conv Wid=', len(Weights)
print 'Begin Conv Bias=', len(Biases)
# convolution
Weights, Wrecur_root = InitParam(Weights, num=numFea * numRecur)
Weights, Wrecur_left = InitParam(Weights, num=numRecur * numRecur)
Weights, Wrecur_right = InitParam(Weights, num=numRecur * numRecur)

Biases, Brecur = InitParam(Biases, num=numRecur)

# discriminative layer
Weights, Wdis = InitParam(Weights, num=numRecur * numDis)
Biases, Bdis = InitParam(Biases, num=numDis)

# output layer
Weights, Wout = InitParam(Weights, num=numDis * numOut, upper=.0002, lower=-.0002)
Biases, Bout = InitParam(Biases, newWeights=np.zeros((numOut, 1)))

Weights = Weights.reshape((-1, 1))
Biases = Biases.reshape((-1, 1))
print Weights[1], Weights[2], Weights[3], Weights[4], Weights[5]
print 'numDis', numDis
print 'Weights', len(Weights)
print 'Bias', len(Biases)
# dwadwad
# 17940
# 1544
#
write_param.write_binary('../paramTest', Weights, Biases)
print 'Done!'