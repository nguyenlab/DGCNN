import struct
import numpy as np
import sys
sys.path.append('../')
from InitParam import *
import write_param
import gl
fileData = open('../param_pretrain1','rb')


numCon = gl.numCon
numWords = gl.numWords
numDis = gl.numDis
numOut = gl.numOut
numPool = gl.numPool
numFea = gl.numFea
    
numW = struct.unpack('i', fileData.read(4) )
numB = struct.unpack('i', fileData.read(4) )


print numW
print numB
Weights = np.array([])
Biases = np.array([])
Weights_pretrain = [0.0] * 180000
Biases_pretrain = [0.0] * 62817900

for i in xrange( 182000 ):
    tmp = struct.unpack('f', fileData.read(4) )
    if i >= 180000:
        continue
    Weights_pretrain[i] = tmp
    
    


for i in xrange( 62817905):
    
    if i >= 62817900:
        continue
    tmp = struct.unpack('f', fileData.read(4) ) 
    
    #if i >= 62817900:
    #    continue
    Biases_pretrain[i] = tmp
    
    
print Biases_pretrain[0]
print Biases_pretrain[1]

Biases, Bpretrain = InitParam(Biases, newWeights = Biases_pretrain)
Weights, Wpretrain = InitParam(Weights, newWeights = Weights_pretrain)  


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

write_param.write_binary('../param', Weights, Biases)

print len(Weights)
print len(Biases)
