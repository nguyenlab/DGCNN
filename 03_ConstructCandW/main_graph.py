import constructNetwork_TBCNN_Sib as TC
import cPickle as p
from nn import serialize
import treeNode
import nn
import pycparser
from nn import FFNN
import sys, os

sys.path.append('../nn')
from pycparser import c_parser, c_ast, parse_file

from InitParam import *

import gl
from nn import Token
import numpy as np

sys.setrecursionlimit(1000000)

numFea = gl.numFea
numCon = gl.numCon

tokenMap = treeNode.LoadTokenMap('D:/GraphData/graphs.txt') #p.load(open('../tokenMap_Loop.txt'))
tokenNum = len(tokenMap)
numWords = len(tokenMap)
numDis = gl.numDis
numOut = gl.numOut

numPool = 3

np.random.seed(314)

preWeights = np.array([])
preBiases = np.array([])
#InitParam return: random value, indices
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

# discriminative layer
Weights, Wdis = InitParam(Weights, num=numPool * numCon * numDis)
Biases, Bdis = InitParam(Biases, num=numDis)

# output layer
Weights, Wout = InitParam(Weights, num=numDis * numOut, upper=.0002, lower=-.0002)
Biases, Bout = InitParam(Biases, newWeights=np.zeros((numOut, 1)))

Weights = Weights.reshape((-1, 1))
Biases = Biases.reshape((-1, 1))
# initial the gradients

gradWeights = np.zeros_like(Weights)
gradBiases = np.zeros_like(Biases)


def computeLeafNum(root, nodes, depth=0):
    if len(root.children) == 0:
        root.leafNum = 1
        root.childrenNum = 1
        return 1, 1, depth  # leafNum, childrenNum
    root.allLeafNum = 0
    avgdepth = 0.0
    for child in root.children:
        leafNum, childrenNum, childAvgDepth = computeLeafNum(nodes[child], nodes, depth + 1)
        root.leafNum += leafNum
        root.childrenNum += childrenNum
        avgdepth += childAvgDepth * leafNum
    avgdepth /= root.leafNum
    root.childrenNum += 1
    return root.leafNum, root.childrenNum, avgdepth


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
                    nodes[child].rightRate = nodes[child].pos / (lenchildren - 1.0)
                    nodes[child].leftRate = 1.0 - nodes[child].rightRate

    dummy, dummy, avgdepth = computeLeafNum(nodes[-1], nodes)
    # print 'avgdepth', avgdepth
    avgdepth *= .6
    if avgdepth < 1:
        avgdepth = 1
    layers = TC.ConstructTreeConvolution(nodes, numFea, numCon, numDis, numOut, \
                                         Wleft, Wright, Bconstruct, \
                                         Wcomb_ae, Wcomb_orig, \
                                         Wconv_root, Wconv_left, Wconv_right, Bconv, \
                                         Wdis, Wout, Bdis, Bout, \
                                         poolCutoff=avgdepth
                                         )
    return layers


def ConstructNodes(treenode, name, parent, pos, nodes, leafs):
    # global tmpCnt
    if name == None:
        name = treenode.__class__.__name__
    # if name not in tmptokenMap.keys():
    #        tmptokenMap[name] = tmpCnt
    #        tmpCnt += 1
    # if name == 'For' or name == 'While' or name == 'DoWhile':
    #     name = 'Loop'
    Node = Token.token(name, gl.numFea * tokenMap[name], \
                       parent, pos)
    if len(treenode.children) == 0:
        leafs.append(Node)
    else:
        nodes.append(Node)
    # print nodes[0].word
    curid = len(nodes)
    for idx, child in enumerate(treenode.children):
        ConstructNodes(child, child.content, curid, idx, nodes, leafs)
        # def __init__(self, word, bidx, parent, pos = 0):


def AdjustOrder(nodes):
    nodes.reverse()
    length = len(nodes)
    for node in nodes:
        if node.parent != None:
            node.parent = length - node.parent


datadir = 'D:/GraphData/data/'
targetdir = 'D:/GraphData/networks/'
procount = 0;
parser = pycparser.c_parser.CParser()
for subdir in os.listdir(datadir):
    if not subdir.endswith('/'):
        subdir = subdir + '/'
    count = 0
    procount += 1
    print '!!!!!!!!!!!!!!!!!!  procount = ', procount
    # if procount >3:
    #    break
    for onefile in os.listdir(datadir + subdir):

        # print 'oneoneoneoneone!!!!!!!!!! '
        filename = onefile
        onefile = datadir + subdir + onefile

        try:
            # print '-----------------', onefile, '-----------------'
            # ast = parse_file(onefile, use_cpp=True)
            if onefile.endswith(".txt"):
                tree = treeNode.LoadTree(onefile)

                # ast.show()
                # print(ast)
                # f.write(onefile)
                # f.write("\t\tAST\t\t")
                # f.write(cat + '\t\t')
                # ast.exporttofile(offset=0, attrnames=False, nodenames=False, _my_node_name='PVA', f=f)
                # (self, buf=sys.stdout, offset=0, attrnames=False, nodenames=False, showcoord=False, _my_node_name=None)
                # ast.traverse2paragraph(buf = f,offset=0, attrnames=False, nodenames=False, _my_node_name='PVA')
                # f.write('\n')
                # print 'ooooops, parsing error for', onefile
                # print ast
        except:
            print 'ooooops, parsing error for', onefile
            continue
        nodes = []
        leafs = []
        # print 'constructing the network'
        ConstructNodes(tree, 'root', None, None, nodes, leafs)
        # print len(nodes)
        # print nodes[0].word
        nodes.extend(leafs)
        # print len(nodes)
        AdjustOrder(nodes)

        # print '---------------------------------------'
        for ii in xrange(len(nodes)):
            # print ii
            inode = nodes[ii]
            # print ii,'  ',inode.word,'  ',inode.parent,'  ',nodes[inode.parent].word,'  ',inode.pos
        layers = InitByNodes(nodes)

        count += 1
        # print 'count = ',count

        directory = targetdir + subdir
        if not os.path.exists(directory):
            os.makedirs(directory)
        serialize.serialize(layers, directory + '/seri_' + filename)

        # print 'processed ', onefile, len(nodes), 'nodes,', len(layers), 'layers'
        ##############look!!
        # break
        # print '222222222222 '

print 'Done!!!!!!'
print 'procount = ', procount


