import commonFunctions
import common_params
import constructNetwork_TBCNN_Sib as TC
import cPickle as p

import sys

import write_param

sys.path.append('../nn')

from InitParam import *

import gl
from nn import Token
import numpy as np
sys.setrecursionlimit(1000000)

netstructure ='sibstcnn'
numFea = gl.numFea
numCon = gl.numCon
    
tokenMap = p.load(open(common_params.tokenMapFile))
common_params.tokenMap = tokenMap
print 'Load token map from:',common_params.tokenMapFile
tokenNum = len(tokenMap)
numWords = len(tokenMap)
numDis = gl.numDis
numOut = gl.numOut
    

numPool = 3

     
np.random.seed(314)

preWeights = np.array([])
preBiases =  np.array([])
preWeights, preWleft = InitParam(preWeights, num=numFea*numFea)
preWeights, preWright= InitParam(preWeights, num=numFea*numFea)
preBiases,  preBtoken = InitParam(preBiases,  num=numFea*tokenNum, upper=0.4,lower=0.6)
preBiases,  preBconstruct  = InitParam(preBiases, num =numFea)

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
Biases, BwordIdx = InitParam(Biases, newWeights = preBtoken)

        #Biases, BwordIdx =  InitParam(Biases, len(preBtoken))
        # left/right constructing and its biases
Weights, Wleft  = InitParam(Weights, newWeights = preWleft)
Weights, Wright = InitParam(Weights, newWeights = preWright)
Biases,Bconstruct=InitParam(Biases,  newWeights = preBconstruct)

'''

Biases, BwordIdx = InitParam(Biases, num=numFea*tokenNum, upper=0.4,lower=0.6)

        #Biases, BwordIdx =  InitParam(Biases, len(preBtoken))
        # left/right constructing and its biases
Weights, Wleft  = InitParam(Weights, num=numFea*numFea)
Weights, Wright = InitParam(Weights, num=numFea*numFea)
Biases,Bconstruct=InitParam(Biases,  num =numFea)


'''


        #Weights, Wleft = InitParam(Weights,  len(preWleft) )
        #Weights, Wright= InitParam(Biases,   len(preWright) )
        #Biases, Bconstruct = InitParam(Biases, len(preBconstruct))        
        
        # combinition of the encoded vector and the vector of the symbol
w1 = (np.eye(numFea) / 2).reshape(-1)
w2 = (np.eye(numFea) / 2).reshape(-1)
       
Weights, Wcomb_ae = InitParam(Weights, newWeights = w1)
Weights, Wcomb_orig=InitParam(Weights, newWeights = w2)
        
        # convolution
Weights, Wconv_root = InitParam(Weights, num = numFea*numCon)
Weights, Wconv_left = InitParam(Weights, num = numFea*numCon)
Weights, Wconv_right= InitParam(Weights, num = numFea*numCon)
Weights, Wconv_sib= InitParam(Weights, num = numFea*numCon)

Biases,  Bconv      = InitParam(Biases,  num = numCon)
        
        # discriminative layer
Weights, Wdis = InitParam(Weights, num = numPool*numCon*numDis)
Biases,  Bdis = InitParam(Biases,  num = numDis)
        
        # output layer
Weights, Wout = InitParam(Weights, num = numDis*numOut, upper = .0002, lower = -.0002)
Biases,  Bout = InitParam(Biases,  newWeights = np.zeros( (numOut,1) ) )
        
        
Weights = Weights.reshape((-1,1))
Biases = Biases.reshape((-1,1))
# initial the gradients

gradWeights = np.zeros_like(Weights)
gradBiases = np.zeros_like(Biases)

print 'numDis', numDis
print 'numCon', numCon
print 'numOut', numOut
print 'Weights', len(Weights)
print 'Bias', len(Biases)

paramsFile = 'paramTest_SibTBCNN_Conv'+ str(gl.numCon)+'_Dis'+str(gl.numDis)
write_param.write_binary(common_params.xypath+'../'+paramsFile, Weights, Biases)
print 'Parameters have been saved at: ', common_params.xypath+'../'+paramsFile

def InitByNodes(nodes):
# add more infomation
    for nidx in xrange(len(nodes)):
        if nodes[nidx].parent != None:
            nodes[nodes[nidx].parent].children.append(nidx)

    #add infor of sibling
    for nidx in xrange(len(nodes)):
        if nodes[nidx].parent != None:
            nodes[nidx].siblings.extend(nodes[nodes[nidx].parent].children)  # appends all its children
            if nidx in nodes[nidx].siblings:
                nodes[nidx].siblings.remove(nidx)  # remove itself
    # add position info
    for nidx in xrange(len(nodes)):
        node = nodes[nidx]

        lenchildren = len(node.children)
        if lenchildren >= 0:
            for child in node.children:
                if lenchildren == 1:
                    nodes[child].leftRate = .5
                    nodes[child].rightRate = .5
                else:
                    nodes[child].rightRate = nodes[child].pos/(lenchildren-1.0)
                    nodes[child].leftRate = 1.0 - nodes[child].rightRate


    dummy, dummy, avgdepth = commonFunctions.computeLeafNum(nodes[-1], nodes)
    #print 'avgdepth', avgdepth
    avgdepth *= .6
    if avgdepth < 1:
        avgdepth = 1
    layers = TC.ConstructTreeConvolution(nodes, numFea, numCon, numDis, numOut,\
                           Wleft, Wright, Bconstruct,\
                           Wcomb_ae, Wcomb_orig,\
                           Wconv_root, Wconv_left, Wconv_right,Wconv_sib, Bconv,\
                           Wdis, Wout, Bdis, Bout,\
                           poolCutoff = avgdepth
        )
    return layers