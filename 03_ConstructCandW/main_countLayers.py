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
 
sta = open('StatisticsMaxDepth104','w')
#sta.write('NodesNum  NumLeaf  Avgdepth  lenOfLayers\n')
sta.write('maxDepth')

def computeLeafNum(root, nodes, depth = 0):
    if len(root.children) == 0:
        root.leafNum = 1
        root.childrenNum = 1
        return 1, 1, depth # leafNum, childrenNum
    root.allLeafNum = 0
    avgdepth = 0.0
    maxdepth = depth
    for child in root.children:
        #leafNum, childrenNum, childAvgDepth, tmpmdepth = computeLeafNum(nodes[child], nodes, depth+1)
        leafNum, childrenNum, tmpmdepth = computeLeafNum(nodes[child], nodes, depth+1)
        if tmpmdepth >maxdepth:
            maxdepth = tmpmdepth        
        root.leafNum += leafNum
        root.childrenNum += childrenNum
        
    
    root.childrenNum += 1
    return root.leafNum, root.childrenNum, maxdepth

def InitByNodes(nodes):
# add more infomation
    for nidx in xrange(len(nodes)):
        if nodes[nidx].parent != None:
            nodes[nodes[nidx].parent].children.append(nidx)
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


    RootLeafNum, dummy, maxdepth = computeLeafNum(nodes[-1], nodes)
    #print 'avgdepth', avgdept
    '''
    sta.write(str(len(nodes)))
    sta.write('    ')
    sta.write(str(RootLeafNum))
    sta.write('    ')
    sta.write(str(avgdepth))
    sta.write('    ')
    '''
    #print maxdepth
    #print RootLeafNum #avgdepth
    #print len(nodes)
    '''
    avgdepth *= .6
    if avgdepth < 1:
        avgdepth = 1
    layers = TC.ConstructTreeConvolution(nodes, numFea, numCon, numDis, numOut,\
                           Wleft, Wright, Bconstruct,\
                           Wcomb_ae, Wcomb_orig,\
                           Wconv_root, Wconv_left, Wconv_right, Bconv,\
                           Wdis, Wout, Bdis, Bout,\
                           poolCutoff = avgdepth
        )
    '''
    layers = ''
    #sta.write(str(len(layers)))
    sta.write(str(maxdepth))    
    sta.write('\n')
    return layers


def ConstructNodes(ast, name, parent, pos, nodes, leafs):
    #global tmpCnt
    if name == None:
        name = ast.__class__.__name__
    #    if name not in tmptokenMap.keys():
    #        tmptokenMap[name] = tmpCnt
    #        tmpCnt += 1
    Node = Token.token(name , gl.numFea*tokenMap[name],\
                 parent, pos)
    if len(ast.children()) == 0:
        leafs.append(Node)
    else:
        nodes.append(Node)
    #print nodes[0].word
    curid = len(nodes)
    for idx, (name, child) in enumerate(ast.children()):
        ConstructNodes(child, None, curid, idx, nodes, leafs)
    #def __init__(self, word, bidx, parent, pos = 0):

def AdjustOrder(nodes):
    nodes.reverse()
    length = len(nodes)
    for node in nodes:
        if node.parent != None:
            node.parent = length - node.parent
            
         
            


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
        try:
            #print '-----------------', onefile, '-----------------'
            ast = parse_file(onefile, use_cpp=True)
        except:
            print 'ooooops, parsing error for', onefile
            continue
        nodes = []
        leafs = []
        #print 'constructing the network'
        ConstructNodes(ast, 'root', None, None, nodes, leafs)
        #print len(nodes)
        #print nodes[0].word
        nodes.extend(leafs)
        #print len(nodes)
        AdjustOrder(nodes)
        layers = InitByNodes(nodes)

        #fdump = open(targetdir + subdir + filename, 'w')
        #p.dump(layers, fdump)
        #fdump.close()
        #print 'len of layers is ',len(layers)

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
