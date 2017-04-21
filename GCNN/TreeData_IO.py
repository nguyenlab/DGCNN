import sys
sys.path.append('../')

import json
import os
import random

import numpy as np
import gl
import pycparser
from Graph import Graph, GVertex
import GraphData_IO
import gcnn_params

gl.reConstruct= False # reconstruct For, While, DoWhile
gl.ignoreDecl = False # Ignore declaration branches
parser = pycparser.c_parser.CParser()

# convert AST ---> Graph
def getGraphFromSourceCode(filename):
    instream = open(filename, 'r')
    text = instream.read()
    instream.close()
    ast = parser.parse(text=text)  # Parse code to AST
    if gl.reConstruct:  # reconstruct braches of For, While, DoWhile
        ast.reConstruct()
    g = tree2Graph(ast)

    return g

def tree2Graph(root):
    # convert from AST ---> Graph
    vertexes ={}
    edges =[]
    tokdict ={}
    vertex_dict={}
    traverseTree(node= root, vertexes=vertexes, edges=edges,
                     tokdict=tokdict, vertex_dict=vertex_dict, parent_name = '')
    return Graph(vertexes, edges)
def traverseTree( node, vertexes, edges, tokdict, vertex_dict, parent_name=''):
    # traverse tree nodes to create vertexes and edges for graph
    node_name = node.__class__.__name__
    # check and add the token to token dictionary if not exist
    if node_name in tokdict:
        tokdict[node_name]+=1
    else:
        tokdict[node_name] = 0
    # add current node to vertexes
    v = GVertex(id=0, name=node_name + '_' + str(tokdict[node_name]), token= node_name, toktype='ASM', content='')
    v.id = len(vertexes)
    vertexes[v.id] = v
    vertex_dict[v.name] = len(vertex_dict)
    # add edge from parent ---> current node
    if parent_name !='': # not root node
        edges.append((vertex_dict[parent_name], v.id))
    for (child_name, child) in node.children():
        traverseTree(node= child, vertexes=vertexes, edges=edges,
                     tokdict=tokdict, vertex_dict=vertex_dict, parent_name = v.name)

def generateGraphJson():
    # generate train - CV - test sets ( graph jSon format)
    pronum = 104
    procount = 0
    datadir ='D:/data/original_data/'
    desdir ='D:/JsonAST_Graph/'
    config ='data'
    network =[]
    for pi in xrange(1, pronum + 1):
        subdir = str(pi) + '/'
        procount += 1
        print 'procount = ', procount
        for onefile in os.listdir(datadir + subdir):
            # print 'oneoneoneoneone!!!!!!!!!! '
            filename = onefile
            onefile = datadir + subdir + onefile
            network.append((onefile, procount - 1))

    np.random.seed(314159)
    np.random.shuffle(network)

    print len(network)
    numTrain = int(.6 * len(network))
    numCV = int(.8 * len(network))
    print 'numTrain : ', numTrain
    print 'numCV : ', numCV - numTrain
    print 'numTest', len(network) - numCV
    print 'final procount = ', procount

    json_graphs=[]
    for i in xrange(0, numTrain):
        (tf, ti) = network[i]
        g = getGraphFromSourceCode(tf)
        g.label = ti

        json_graphs.append(g.dump())
    # write training graphs
    f = file(desdir + config + '_train.json', 'w')
    json.dump(json_graphs, f)
    f.close()


    json_graphs =[]
    for i in xrange(numTrain, numCV):
        (tf, ti) = network[i]
        g = getGraphFromSourceCode(tf)
        g.label = ti

        json_graphs.append(g.dump())

    f = file(desdir+config + '_CV.json', 'w')
    json.dump(json_graphs, f)
    f.close()


    json_graphs =[]
    for i in xrange(numCV, len(network)):
        (tf, ti) = network[i]
        g = getGraphFromSourceCode(tf)
        g.label = ti

        json_graphs.append(g.dump())
    f = file(desdir+config + '_test.json', 'w')
    json.dump(json_graphs, f)
    f.close()
from InitParam import *
def createTokGroupVec():
    import cPickle as p
    import gl
    import numpy as np

    numFea = gl.numFea
    numCon = gl.numCon

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

    preBtoken = preBtoken.flatten()
    tok_id = []
    for tok in tokenMap:
        tok_id.append((tok, tokenMap[tok]))

    tok_id = sorted(tok_id, key=lambda x: x[1])

    for tok, id in tok_id:
        bidx = id * numFea
        vec = preBtoken[bidx:bidx + numFea]
        vec = [str(i) for i in vec]
        print tok  # , ' '.join(vec)
    # grop
    groups = ['g_others', 'g_decl', 'g_select',
              'g_value', 'g_op', 'g_loop', 'g_jump']
    for g in groups:
        vec = [random.uniform(-1, 1) for v in xrange(numFea)]
        vec = [str(i) for i in vec]
        print g, ' '.join(vec)
def testAST_GraphNet():
    text ='''
 int main()
 {
    int x = y + 3;
 }
    '''
    parser = pycparser.c_parser.CParser()
    ast = parser.parse(text=text)  # Parse code to AST
    ast.show()
    if gl.reConstruct:  # reconstruct braches of For, While, DoWhile
        ast.reConstruct()
    g =tree2Graph(ast)
    import main_MultiChannelGCNN as GCNN
    word_dict, vectors, numFea = GraphData_IO.LoadVocab(vocabfile=gcnn_params.datapath + 'tokvec.txt')
    layers = GCNN.InitByNodes(graph=g, word_dict=word_dict)
    print "num node =", len(g.Vs)
    print 'Totally:', len(layers), 'layer(s)'
    num_con =0
    for i, layer in enumerate(layers):
        layer.idx = i
        num_con += len(layer.connectUp)
        for (icon, con) in enumerate(layer.connectDown):
            con.ydownid = icon

    print 'connection num:',num_con

    for l in layers:
        if hasattr(l, 'bidx') and l.bidx is not None:
            print l.name, '\tBidx', l.bidx[0], '\tlen biases=', len(l.bidx)
        else:
            print l.name

        print "    Down:"
        for c in l.connectDown:
            if hasattr(c, 'Widx'):
                print "        ", c.xlayer.name, " -> ", '|', \
                    '(xnum= ', c.xnum, ', ynum= ', c.ynum, '), Wid = ', c.Widx[0], '(Woef=', c.Wcoef, ')'
            else:
                print "        ", c.xlayer.name, " -> ", '|', \
                    '(xnum= ', c.xnum, ', ynum= ', c.ynum, ')'
if __name__ == '__main__':
    testAST_GraphNet()
    print 'Done'

