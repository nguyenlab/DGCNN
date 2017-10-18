import random
import struct

import constructNetWork_MLP as TC
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

print 'num of biases = ', len(Biases)
print 'num of weights = ', len(Weights)

# initial the gradients

gradWeights = np.zeros_like(Weights)
gradBiases = np.zeros_like(Biases)



def InitNetbyText(text=''):

    phrases = text.split('|||')
    if len(phrases) !=2:
        print '\ndata sample error '+ text
    else:
        phrase_1 = phrases[0].lstrip().rstrip().split(' ')
        phrase_2 = phrases[1].lstrip().rstrip().split(' ')

        layers = InitByNodes(phrase_1, phrase_2)

    return layers
def InitByNodes(phrase_1, phrase_2):
    # phrase_1, phrase_2, word_dict, numFea, numLeft, numRight, numJoint, numDis, numOut, \
    # Wleft, Wright, Bleft, Bright,
    # Wjoint_left, Wjoint_right, Bjoint,
    # Wdis, Wout, Bdis, Bout                          ):
    # w1 - ---> |
    # w2 - ---> | Left  ---> |
    # w3 - ---> |            |
    #                        | joint ---> Fully - Connected ---> out
    # w1 - ---> |            |
    # w2 - ---> | right ---> |
    # w3 - ---> |


    layers = TC.ConstructTreeConvolution( phrase_1, phrase_2, word_dict, numFea,numLeft, numRight, numJoint, numDis, numOut, \
                             Wleft, Wright, Bleft, Bright,
                             Wjoint_left, Wjoint_right, Bjoint,
                             Wdis, Wout, Bdis, Bout)


    return layers

def ConstructNetworksFromFile(datafile='',targetdir='', prefix ='', classlabel =1, networks=[]):
    file = open(datafile, "r")
    idx =0
    for line in file:

        layers = InitNetbyText(text=line)
        networks.append((layers, classlabel-1))
        idx = idx +1
        if idx % 1000==0:
            print idx
        # if (idx>10):
        #     break

        # netfile =str(idx)
        # directory = targetdir + str(classlabel)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # serialize.serialize(layers, directory + '/'+prefix+'_' + netfile)
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
def testNet():
    # test net
    text = "in favour motion ||| health system"
    layers = InitNetbyText(text=text)
    print 'Totally:', len(layers), 'layer(s)'
    for l in layers:
        if hasattr(l, 'bidx') and l.bidx is not None:
            print l.name, '\tBidx', l.bidx[0], '\tlen biases=', len(l.bidx)
        else:
            print l.name


        print "    Down:"
        for c in l.connectDown:
            if hasattr(c, 'Widx'):
                print "        ", c.xlayer.name, " -> ", '|', \
                    '(xnum= ', c.xnum, ', ynum= ', c.ynum, '), Wid = ', c.Widx[0]
            else:
                print "        ", c.xlayer.name, " -> ", '|', \
                    '(xnum= ', c.xnum, ', ynum= ', c.ynum, ')'
def ConstructNetworksForTest(testfile='', targetdir ='',netfile='')  :
    networks =[]
    ConstructNetworksFromFile(datafile=testfile, targetdir='', prefix='',
                              classlabel=1, networks=networks)

    # write networks
    f = file(targetdir+netfile+'_x', 'wb')
    f_y = file(targetdir+netfile+'_y', 'w')
    for i in xrange(0, len(networks)):
        (net, ti) = networks[i]
        #write net
        WriteNet(f, net)
        # print ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()


if __name__ == "__main__":
    #testNet()
    # targetdir = datapath+'networks/'
    # networks =[]
    # # #positive file
    # ConstructNetworksFromFile(datafile=datapath+'positive.training-data.txt', targetdir=targetdir, prefix='pos', classlabel=1, networks=networks)
    # # #negative file
    # ConstructNetworksFromFile(datafile=datapath + 'negative.training-data.txt', targetdir=targetdir, prefix='neg', classlabel=2, networks=networks)
    #
    # # test

    for onefile in os.listdir(datapath+'testdata/'):
        ConstructNetworksForTest(testfile=datapath+'testdata/'+onefile, targetdir=datapath+'testnet/', netfile=onefile)

    # # write network
    # np.random.seed(314159)
    # np.random.shuffle(networks)
    #
    # print 'networks =',len(networks)
    # numTrain = int(.7 * len(networks))
    # print 'numTrain : ', numTrain
    # print 'numTest : ', len(networks) - numTrain
    #
    # f = file(datapath+'xy/' + 'data_train', 'wb')
    # f_y = file(datapath+'xy/' + 'data_ytrain.txt', 'w')
    # for i in xrange(0, numTrain):
    #     (net, ti) = networks[i]
    #     #write net
    #     WriteNet(f, net)
    #     # print ti
    #     f_y.write(str(ti) + '\n')
    # f.close()
    # f_y.close()
    #
    # f = file(datapath+'xy/' +  'data_CV', 'wb')
    # f_y = file(datapath+'xy/' + 'data_yCV.txt', 'w')
    #
    # for i in xrange(numTrain, len(networks)):
    #     (net, ti) = networks[i]
    #     #write net
    #     WriteNet(f, net)
    #     #write ti
    #     f_y.write(str(ti) + '\n')
    # f.close()
    # f_y.close()
    #
    # f = file(datapath+'xy/' + 'data_test', 'wb')
    # f_y = file(datapath+'xy/' + 'data_ytest.txt', 'w')
    #
    # for i in xrange(numTrain, len(networks)):
    #     (net, ti) = networks[i]
    #     # write net
    #     WriteNet(f, net)
    #     # write ti
    #     f_y.write(str(ti) + '\n')
    # f.close()
    # f_y.close()
    #
    #
    #
    # print 'Done!!'
    #
    #
    # # filename =str(idx)
    # # directory = gl.targetdir + str(probcount)
    # # if not os.path.exists(directory):
    # #     os.makedirs(directory)
    # # serialize.serialize(layers, directory + '/seri_' + filename)




