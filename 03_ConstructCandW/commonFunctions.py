import sys

import common_params

sys.path.append('../')
import struct

import gl
from nn import Token

def readParameters(binIn,out):
    fin = open(binIn, 'rb')
    content = fin.read()
    fin.close()

    numW = struct.unpack('i', content[0:4])[0]
    numB = struct.unpack('i', content[4:8])[0]

    bidx = 8

    fout = open(out,'w')
    fout.writelines('numWeights: ' + str(numW)+'\n')
    fout.writelines('numBiases: '+str(numB)+'\n')
    fout.writelines('Weights\n')
    for idx in range(0,numW):
        w = struct.unpack('f', content[bidx:bidx + 4])[0]
        bidx +=4
        # fout.writelines(str(w)+'\n')

    fout.writelines('Biases')
    for idx in range(0, numB):
        b = struct.unpack('f', content[bidx:bidx + 4])[0]
        bidx +=4
        fout.writelines(str(b)+'\n')

    fout.close()

    # def write_binary(fname, W, B):
    #     f = file(fname, 'wb')
    #     numW = struct.pack('i', len(W))
    #     numB = struct.pack('i', len(B))
    #
    #     f.write(numW)
    #     f.write(numB)
    #     for i in xrange(len(W)):
    #         tmp = struct.pack('f', W[i, 0])
    #         f.write(tmp)
    #
    #     for i in xrange(len(B)):
    #         tmp = struct.pack('f', B[i, 0])
    #         f.write(tmp)
    #
    #     f.close()

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

def ConstructNodes(ast, name, parent, pos, nodes, leafs, tokenMap):
    # global tmpCnt
    if name is None:
        name = ast.content
    if common_params.reConstruct:
        if name =='While' or name == 'DoWhile':
            name ='For'
    Node = Token.token(name, gl.numFea * tokenMap[name], \
                       parent, pos)
    if len(ast.children) == 0:
        leafs.append(Node)
    else:
        nodes.append(Node)
    # print nodes[0].word
    curid = len(nodes)
    for idx, child in enumerate(ast.children):
        ConstructNodes(child, None, curid, idx, nodes, leafs, tokenMap)

def AdjustOrder(nodes):
    nodes.reverse()
    length = len(nodes)
    for node in nodes:
        if node.parent != None:
            node.parent = length - node.parent

def WriteNet(f =None, layers=None):
    num_lay = struct.pack('i', len(layers))
    if num_lay <= 2:
        print 'error'
    f.write(num_lay)

    num_con = 0

    #################################
    # preprocessing, compute some indexes
    for i, layer in enumerate(layers):
        layer.idx = i
        num_con += len(layer.connectUp)
        for (icon, con) in enumerate(layer.connectDown):
            con.ydownid = icon

    # print len(layers),' ' , num_con
    num_con = struct.pack('i', num_con)
    f.write(num_con)

    #################################
    # layers

    for layer in layers:
        # name
        # = struct.pack('s', layer.name )
        # numUnit
        tmp = struct.pack('i', layer.numUnit)
        f.write(tmp)
        # numUp
        tmp = struct.pack('i', len(layer.connectUp))
        f.write(tmp)
        # numDown
        tmp = struct.pack('i', len(layer.connectDown))
        f.write(tmp)

        if layer.layer_type == 'p':  # pooling
            if layer.poolType == 'max':
                tlayer = 'x'
            elif layer.poolType == 'sum':
                tlayer = 'u'
            tmp = struct.pack('c', tlayer)
            f.write(tmp)

        elif layer.layer_type == 'o':  # ordinary nodes

            if layer.act == 'embedding':
                tlayer = 'e'
            elif layer.act == 'autoencoding':
                tlayer = 'a'
            elif layer.act == 'convolution':
                tlayer = 'c'
            elif layer.act == 'combination':
                tlayer = 'b'
            elif layer.act == "ReLU":
                tlayer = 'r'
            elif layer.act == 'softmax':
                tlayer = 's'
            elif layer.act == 'hidden':
                tlayer = 'h'
            elif layer.act == 'recursive':
                tlayer = 'v'
            else:
                print "error"
                return layer
            tmp = struct.pack('c', tlayer)

            f.write(tmp)
            bidx = -1
            if layer.bidx != None:
                bidx = layer.bidx
                bidx = bidx[0]

            tmp = struct.pack('i', bidx)
            f.write(tmp)

    #########################
    # connections
    for layer in layers:
        for xupid, con in enumerate(layer.connectUp):
            # xlayer idx
            tmp = struct.pack('i', layer.idx)

            f.write(tmp)
            # ylayer idx
            tmp = struct.pack('i', con.ylayer.idx)
            f.write(tmp)
            # idx in x's connectUp
            tmp = struct.pack('i', xupid)
            f.write(tmp)
            # idx in y's connectDown
            tmp = struct.pack('i', con.ydownid)
            f.write(tmp)
            if con.ylayer.layer_type == 'p':
                Widx = -1
            else:
                Widx = con.Widx
                Widx = Widx[0]

            tmp = struct.pack('i', Widx)
            f.write(tmp)
            if Widx >= 0:
                tmp = struct.pack('f', con.Wcoef)
                f.write(tmp)
def generateSettingContent(filename, params):
    template='''batch
10
begin
1
num of epoch
60
mark point to write output
60
mode (type of output data)
0
num_train
<numtrain>
num_cv
<numcv>
num_test
<numtest>
output
<output>
parameter file
<paramFile>
fx_train  (x -training file)
<xtrain>
fx_CV     (x - validation file)
<xcv>
fx_test   (x - test file)
<xtest>
fy_train  (y - training file)
<ytrain>
fy_CV     (y- validation file)
<ycv>
fy_test   (y- test file)
<ytest>
alpha
0.1
beta
0.7
active function(ReLU, tanh)
tanh
database
<database>
// p1: epoch, p2: mark - export from this round, mode: 0 - export probabilities, 1- predicting results, 2-vector
'''
    for pname in params:
        key ='<'+pname+'>'
        value = str(params[pname])
        template = template.replace(key, value)
    #print template

    f = open(filename,'w')
    f.write(template)
    f.close()
    print 'setting parameters are saved at '+ filename
# generateSettingContent({'numtrain':10, 'numcv':20, 'numtest':30, 'output':2,
#                         'paramFile':'paramFile','xtrain': 'traindata', 'xcv':'cvdata', 'xtest':'testdata',
#                         'ytrain':'ytrain', 'ycv':'ycv','ytest': 'ytest'})