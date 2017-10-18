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

numFea = gl.numFea
numCon = gl.numCon
numRecur = gl.numCon

tokenMap = p.load(open(gl.tokenMap))
tokenNum = len(tokenMap)
numWords = len(tokenMap)
numDis = gl.numDis
numOut = gl.numOut

numPool = 3

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

# word/symbol/token embedding
Biases, BwordIdx = InitParam(Biases, newWeights=preBtoken)

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
Weights, Wcomb_orig = InitParam(Weights, newWeights=w2)

# convolution
Weights, Wconv_root = InitParam(Weights, num=numFea * numCon)
Weights, Wconv_left = InitParam(Weights, num=numFea * numCon)
Weights, Wconv_right = InitParam(Weights, num=numFea * numCon)

Biases, Bconv = InitParam(Biases, num=numCon)
#recursive
Weights, Wrecur_root = InitParam(Weights, num=numFea * numRecur)
Weights, Wrecur_left = InitParam(Weights, num=numRecur * numRecur)
Weights, Wrecur_right = InitParam(Weights, num=numRecur * numRecur)

Biases, Brecur = InitParam(Biases, num=numRecur)

# discriminative layer
Weights, Wconv_dis = InitParam(Weights, num= numCon * numDis)
Weights, Wrecur_dis = InitParam(Weights, num=numRecur * numDis)
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