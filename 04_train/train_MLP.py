import os, sys, numpy as np, copy

import write_param

sys.path.append('../')

import MLP_DataIO
import cPickle as p
from nn import serialize

import sys, os

sys.path.append('../nn')
datapath ='C:/Users/anhpv/Desktop/Long_Data/'

from InitParam import *

import gl
from nn import Token
import numpy as np

sys.setrecursionlimit(1000000)

tokenMap = p.load(open(gl.tokenMap))
tokenNum = len(tokenMap)
numWords = len(tokenMap)

numLeft = numRight = 100
numJoint = 100
numDis = gl.numDis
numOut = gl.numOut

word_dict, vectors, numFea = MLP_DataIO.LoadVocab(vocabfile=datapath + 'w2v_random.txt')
print 'Vocab length =', len(word_dict)
print 'Embedding size =', numFea

np.random.seed(314)

preBword = vectors # numFea*len(Vocab) # word vectors
print len(preBword)
# Initialize weights and biases
Weights = np.array([])
Biases = np.array([])

# word/symbol/token embedding
Biases, BwordIdx = InitParam(Biases, newWeights=preBword)
# left - right layers
Weights, Wleft = InitParam(Weights, num=numFea * numLeft)
Weights, Wright = InitParam(Weights, num=numFea * numRight)

Biases, Bleft = InitParam(Biases, num=numLeft)
Biases, Bright = InitParam(Biases, num=numRight)
# joint layer
Weights, Wjoint_left = InitParam(Weights, num=numLeft * numJoint)
Weights, Wjoint_right = InitParam(Weights, num=numRight * numJoint)

Biases, Bjoint = InitParam(Biases, num=numJoint)

# discriminative layer
Weights, Wdis = InitParam(Weights, num=numJoint * numDis)
Biases, Bdis = InitParam(Biases, num=numDis)

# output layer
Weights, Wout = InitParam(Weights, num=numDis * numOut, upper=.0002, lower=-.0002)
Biases, Bout = InitParam(Biases, newWeights=np.zeros((numOut, 1)))

Weights = Weights.reshape((-1, 1))
Biases = Biases.reshape((-1, 1))


print 'Weights', len(Weights)
print 'Bias', len(Biases)
# dwadwad
# 17940
# 1544
#
write_param.write_binary('../MLP_paramTest', Weights, Biases)
print 'Done!'