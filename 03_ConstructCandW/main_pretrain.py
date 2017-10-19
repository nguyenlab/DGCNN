import constructPretrainNetwork as TC
import cPickle as CP
import serialize
import FFNN 
import sys
sys.path.append('../nn')

from InitParam import *
import cPickle as p
import gl
import Token
import numpy as np
sys.setrecursionlimit(1000000)



numFea = gl.numFea
numCon = gl.numCon
numWords = gl.numWords
numDis = gl.numDis
numOut = gl.numOut
numPool = gl.numPool


#####################
# Weights:
#----------------
# Wconstruct_left
# Wconstruct_right
# Wcombinition_ae
# Wcombinition_orig
# Wconvolution_root
# Wconvolution_left
# Wconvolution_right
#####################
# Biases:
#-------------------
# token features
# Bconstruct
# BConvolution
#####################

np.random.seed(1)


########## Initialise Biases and dict ############


f = file('dic_hash_cache.pkl', 'rb')  
dic = CP.load(f) 
f.close()

flag = False
#########################################
# initialize weights
Weights = np.array([])
Biases = np.array([])
#f = file('word_vector_cache.pkl', 'rb')  
#wordsFea = p.load(f)
# Biases, Bword = InitParam(Biases, newWeights = wordsFea)

Biases, Bword = InitParam(Biases, num = 209392 * 300)
Bword = np.array(Bword)
f.close()
#wordsFea = np.array(wordsFea)

Weights, Wleft  = InitParam(Weights, num = numFea*numFea)
Weights, Wright = InitParam(Weights, num = numFea*numFea)
Biases,Bconstruct=InitParam(Biases,  num = numFea)



# output layer
Weights, Wout = InitParam(Weights, num = numFea * numOut)
Biases,  Bout = InitParam(Biases,  num = numOut)

#InitParam(OldWeights, num = None, newWeights = None, upper = None, lower = None):




#Wdis, Wout, Bdis, Bout

# initial the gradients

Weights = Weights.reshape((-1,1))
Biases = Biases.reshape((-1,1))
gradWeights = np.zeros_like(Weights)
gradBiases = np.zeros_like(Biases)





bidx_encoding = len(dic) * numFea   
def computeLeafNum(root, nodes, depth = 0):
    if len(root.children) == 0:
        root.leafNum = 1
        root.childrenNum = 1
        return 1, 1, depth # leafNum, childrenNum
    root.allLeafNum = 0
    avgdepth = 0.0
    for child in root.children:
        leafNum, childrenNum, childAvgDepth = computeLeafNum(nodes[child],nodes, depth+1)
        root.leafNum += leafNum
        root.childrenNum += childrenNum
        avgdepth += childAvgDepth * leafNum
    avgdepth /= root.leafNum
    root.childrenNum += 1
    return root.leafNum, root.childrenNum, avgdepth
layers = [None] * 200
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


    dummy, dummy, avgdepth = computeLeafNum(nodes[-1], nodes)
    #print 'avgdepth', avgdepth
    avgdepth *= .6
    if avgdepth < 1:
        avgdepth = 1

    layers = TC.ConstructTreeConvolution(nodes, numFea, numOut,\
                           Wleft, Wright, Bconstruct,\
                           Wout, Bout
        )
    #if flag:
    #FFNN.displayNetwork(layers,  Weights, Biases)
      #  dsadwadadw
        
    #FFNN.cleanActivation(layers[0])
    #FFNN.cleanDerivatives(layers[-1])
    
    #truncate = -1
    #layers[ truncate ].successiveUpper = None
    #FFNN.forwardpropagation(layers[0], None, Weights, Biases)
   
    
    return layers


config = 'test'
f1 = open('TreeStructure/SOStr_' + config + '.txt')
f2 = open('TreeStructure/STree_' + config + '.txt')
#f1 = open('TreeStructure/SOS.txt')
#f2 = open('TreeStructure/STree.txt') 
n = 0
line1 = f1.readline()
line2 = f2.readline()

while line1 != None and line2 != None:
    line1 = line1.strip('\n')
    line2 = line2.strip('\n') 
    if line1 == '' or line2 == '':
        break
    SOStr = line1.split('|')
    lSOStr = len(SOStr)

    STree = line2.split('|')
    lSTree = len(STree) 
    
    node = [None] * (lSTree) 
    flag = [0] * (lSTree)
    if n % 100 == 0:
        print n
    for nidx in xrange(lSOStr):
        SOStr[nidx] = SOStr[nidx].lower()
       
        
        if dic.has_key(SOStr[nidx]): 
            idx = dic[ SOStr[nidx] ]
        else:
            idx = 0
        
        
    
        node[nidx] = Token.token(SOStr[nidx], idx * numFea, int (STree[nidx]) - 1 , flag[ int (STree[nidx])  - 1])
        flag[ int (STree[nidx]) - 1] = 1
        
    for nidx in xrange(lSTree - lSOStr):
        node[nidx + lSOStr] = Token.token('n' + str(nidx), bidx_encoding, int (STree[nidx + lSOStr]) - 1, flag[ int (STree[nidx + lSOStr]) - 1])
        flag[ int (STree[nidx + lSOStr]) - 1] = 1
    node[lSTree - 1].parent = None
    
    #for nidx in xrange(len(node)):
    #	print 'word: ' + node[nidx].word + ' bidx: ' +  str (node[nidx].bidx) + ' parent: ' + str(node[nidx].parent) + ' pos: ' + str(node[nidx].pos)
    
    p_nodes = node
    Layers = InitByNodes(p_nodes) 
    
    #fout = open('../../network1_' + config + '/' + str(n), 'wb')
    serialize.serialize( Layers, '../../network_pretrain_' + config + 
            '/' + str(n))
            
    #fout.close()
    n = n + 1
    line1 = f1.readline()
    line2 = f2.readline()



f1.close()
f2.close()



config = 'CV'
f1 = open('TreeStructure/SOStr_' + config + '.txt')
f2 = open('TreeStructure/STree_' + config + '.txt')
#f1 = open('TreeStructure/SOS.txt')
#f2 = open('TreeStructure/STree.txt') 
n = 0
line1 = f1.readline()
line2 = f2.readline()

while line1 != None and line2 != None:
    line1 = line1.strip('\n')
    line2 = line2.strip('\n') 
    if line1 == '' or line2 == '':
        break
    SOStr = line1.split('|')
    lSOStr = len(SOStr)

    STree = line2.split('|')
    lSTree = len(STree) 
    
    node = [None] * (lSTree) 
    flag = [0] * (lSTree)
    if n % 100 == 0:
        print n
    for nidx in xrange(lSOStr):
        SOStr[nidx] = SOStr[nidx].lower()
        
        if dic.has_key(SOStr[nidx]): 
            idx = dic[ SOStr[nidx] ]
        else:
            idx = 0
            
        node[nidx] = Token.token(SOStr[nidx], idx * numFea, int (STree[nidx]) - 1 , flag[ int (STree[nidx])  - 1])
        flag[ int (STree[nidx]) - 1] = 1
        
    for nidx in xrange(lSTree - lSOStr):
        node[nidx + lSOStr] = Token.token('n' + str(nidx), bidx_encoding, int (STree[nidx + lSOStr]) - 1, flag[ int (STree[nidx + lSOStr]) - 1])
        flag[ int (STree[nidx + lSOStr]) - 1] = 1
    node[lSTree - 1].parent = None
    
    #for nidx in xrange(len(node)):
    #	print 'word: ' + node[nidx].word + ' bidx: ' +  str (node[nidx].bidx) + ' parent: ' + str(node[nidx].parent) + ' pos: ' + str(node[nidx].pos)
    
    p_nodes = node
    Layers = InitByNodes(p_nodes) 
    
    #fout = open('../../network1_' + config + '/' + str(n), 'wb')
    serialize.serialize( Layers, '../../network_pretrain_' + config + 
            '/' + str(n))
            
    #fout.close()
    n = n + 1
    line1 = f1.readline()
    line2 = f2.readline()



f1.close()
f2.close()


config = 'train'
f1 = open('TreeStructure/SOStr_' + config + '.txt')
f2 = open('TreeStructure/STree_' + config + '.txt')
#f1 = open('TreeStructure/SOS.txt')
#f2 = open('TreeStructure/STree.txt') 
n = 0
line1 = f1.readline()
line2 = f2.readline()

while line1 != None and line2 != None:
    line1 = line1.strip('\n')
    line2 = line2.strip('\n') 
    if line1 == '' or line2 == '':
        break
    SOStr = line1.split('|')
    lSOStr = len(SOStr)

    STree = line2.split('|')
    lSTree = len(STree) 
    
    node = [None] * (lSTree) 
    flag = [0] * (lSTree)
    if n % 100 == 0:
        print n
    for nidx in xrange(lSOStr):
        SOStr[nidx] = SOStr[nidx].lower()
        
        if dic.has_key(SOStr[nidx]): 
            idx = dic[ SOStr[nidx] ]
        else:
            idx = 0
            
        node[nidx] = Token.token(SOStr[nidx], idx * numFea, int (STree[nidx]) - 1 , flag[ int (STree[nidx])  - 1])
        flag[ int (STree[nidx]) - 1] = 1
        
    for nidx in xrange(lSTree - lSOStr):
        node[nidx + lSOStr] = Token.token('n' + str(nidx), bidx_encoding, int (STree[nidx + lSOStr]) - 1, flag[ int (STree[nidx + lSOStr]) - 1])
        flag[ int (STree[nidx + lSOStr]) - 1] = 1
    node[lSTree - 1].parent = None
    
    #for nidx in xrange(len(node)):
    #	print 'word: ' + node[nidx].word + ' bidx: ' +  str (node[nidx].bidx) + ' parent: ' + str(node[nidx].parent) + ' pos: ' + str(node[nidx].pos)
    
    p_nodes = node
    Layers = InitByNodes(p_nodes) 
    
    #fout = open('../../network1_' + config + '/' + str(n), 'wb')
    serialize.serialize( Layers, '../../network_pretrain_' + config + 
            '/' + str(n))
            
    #fout.close()
    n = n + 1
    line1 = f1.readline()
    line2 = f2.readline()



f1.close()
f2.close()