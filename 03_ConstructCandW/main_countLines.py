import constructNetwork_TBCNN_Sib as TC
import cPickle as p
import serialize
import FFNN 
import sys,os
sys.path.append('../nn')
from pycparser import c_parser, c_ast, parse_file

from InitParam import *

import gl
import Token
import numpy as np
sys.setrecursionlimit(1000000)


numFea = gl.numFea
numCon = gl.numCon
    
tokenMap = p.load(open('../tokenMap.txt'))
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
 
sta = open('StatisticsLines104','w')
sta.write('lines\n')
datadir =   '../../../data/BigData_Choose104/'
#savedir =   '../../../data/BigData_Choose104/'
targetdir = '../../../data/nouse/'
procount = 0;
for subdir in os.listdir(datadir):
    if not subdir.endswith('/'):
        subdir = subdir + '/'
    #print '111111111111111111111'

    count = 0
    procount+=1
    print '!!!!!!!!!!!!!!!!!!  procount = ',procount
    #if procount >3:
    #    break
    for onefile in os.listdir(datadir + subdir):

        #print 'oneoneoneoneone!!!!!!!!!! '
        filename = onefile
              
        onefile = datadir + subdir + onefile
        #print savefile
        #onefile = 'test.c'
        txtread=file(onefile,"r")
        rr=1
        lines = 0
        #s=txtread.next()
        while rr:
            s=txtread.readline()
            if s=='':
                break
            lines = lines +1
        sta.write(str(lines))
        sta.write('\n')
            
        count +=1
        #print 'count = ',count
        if count >500:
            break
        
    
        
        
        #print 'processed ', onefile, len(nodes), 'nodes,', len(layers), 'layers'
        ##############look!!
        #break
    #print '222222222222 '
sta.close()   
print 'Done!!!!!!'
print 'procount = ',procount
