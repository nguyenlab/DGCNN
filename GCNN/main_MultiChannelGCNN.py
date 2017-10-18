import sys

sys.path.append('../nn')
sys.path.append('../03_ConstructCandW')

import json
import commonFunctions
import constructNetWork_MultiChannelGCNN as TC
import GraphData_IO

import write_param
from Graph import Graph
from database import CodeChef
import gcnn_params as params

datapath = params.datapath

from InitParam import *

import numpy as np

sys.setrecursionlimit(1000000)

word_dict, vectors, numFea = GraphData_IO.LoadVocab(vocabfile=datapath + params.tokvecFile)
toktypeDict = GraphData_IO.LoadTokenTypeDict(filename=datapath + params.toktype)

print 'Load token embedding from: ', datapath + params.tokvecFile
print 'Vocab length =', len(word_dict)
print 'Embedding size =', numFea

numDis = params.numDis
numOut = params.numOut

numCon = params.numCon
numView = params.numView

np.random.seed(314)

preBword = vectors # numFea*len(Vocab) # word vectors
print '(Words = ', len(word_dict), ') * (Size =', numFea,') = ', len(preBword)
# Initialize weights and biases
Weights = np.array([])
Biases = np.array([])

# word/symbol/token embedding
Biases, BwordIdx = InitParam(Biases, newWeights=preBword)
#parameters include
# word_dict: [dict_view1, dict_view2]
# numView, numFea, numCon, numDis, numOut, \
# Wconv_root    [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
# Wconv_in  [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
# Wconv_out     [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
# Bconv         [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
# Wdis[pool_view1, pool_view2, ...---> Dis]

Wmap =[]
Bmap =[]

Wconv_root=[]
Wconv_in =[]
Wconv_out =[]
Bconv =[]

# convolution layers
num_Pre = numFea

for c in range(len(numCon)):
    if c==0:
        view_wroot =[None]*numView
        view_win = [None]*numView
        view_wout = [None] * numView

        for v in range(0, numView):
            Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
            view_wroot[v]= w

            Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
            view_win[v] = w

            Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
            view_wout[v] = w

        Wconv_root.append(view_wroot)
        Wconv_in.append(view_win)
        Wconv_out.append(view_wout)
    else:
        Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
        Wconv_root.append(w)

        Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
        Wconv_in.append(w)

        Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
        Wconv_out.append(w)

    Biases, b = InitParam(Biases, num=numCon[c])
    Bconv.append(b)

    num_Pre = numCon[c]

# discriminative layer
Weights, Wdis = InitParam(Weights, num=num_Pre * numDis)

Biases, Bdis = InitParam(Biases, num=numDis)

# output layer
Weights, Woutput = InitParam(Weights, num=numDis * numOut, upper=.0002, lower=-.0002)
Biases, Boutput = InitParam(Biases, newWeights=np.zeros((numOut, 1)))

Weights = Weights.reshape((-1, 1))
Biases = Biases.reshape((-1, 1))

print 'num of biases = ', len(Biases)
print 'num of weights = ', len(Weights)

# initial the gradients

gradWeights = np.zeros_like(Weights)
gradBiases = np.zeros_like(Biases)

print 'numCon', numCon
print 'numDis', numDis
print 'numOut', numOut
print 'Weights', len(Weights)
print 'Bias', len(Biases)
# dwadwad
# 17940
# 1544
#
paramFile ='paramTest_ASTGraph_V'+str(numView)
write_param.write_binary(params.xypath+'../'+paramFile, Weights, Biases)
print 'Parameters have been saved at: ', params.xypath+'../'+paramFile
problem =''
def InitByNodes(graph, word_dict,toktypeDict):
    # Embedding ---> Mapping ---> Conv1 ---> .... ---> Convn ---> Pooling ---> Fully-Connected ---> Output
    # def ConstructGraphConvolution(graph, word_dict, toktypeDict, numView, numFea, numCon, numDis, numOut, \
    #                               Wconv_root, Wconv_in, Wconv_out, Bconv, \
    #                               Wdis, Woutput, Bdis, Boutput
    #                               ):
    # word_dict: [dict_view1, dict_view2]
    # numView, numFea, numCon, numDis, numOut, \
    # Wconv_root    [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Wconv_income  [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Wconv_out     [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Bconv         [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Wdis[pool_view1, pool_view2, ...---> Dis]
    layers = TC.ConstructGraphConvolution(graph, word_dict,toktypeDict, numView, numFea, numCon, numDis, numOut, \
                                  Wconv_root, Wconv_in, Wconv_out, Bconv, \
                                  Wdis, Woutput, Bdis, Boutput
                                 )


    return layers

def getLabel(label):
    if label >0 and numOut==2:
            return 1
    return label
def constructNetFromJson(jsonFile, f_x, f_y):
    # open files to write
    # f_x = file(xfile, 'wb')
    # f_y = file(yfile, 'w')
    count =0
    #numinst = random.randint(20,30)
    with open(jsonFile, 'r') as f:
        jsonObjs = json.load(f)
        for obj in jsonObjs:
            count +=1
            # if count< 26086:
            #     continue
            graph = Graph.load(obj)

            # for id, v in graph.Vs.items():
            #     print v.getData(toktypeDict, withOps = True)
            # return 1

            g_net = InitByNodes(graph=graph, word_dict=word_dict, toktypeDict=toktypeDict)
            # write net
            commonFunctions.WriteNet(f_x, g_net)
            # print y
            f_y.write(str(getLabel(graph.label)) + '\n')
            if count %10:
                f_x.flush()
                f_y.flush()

    # close files
    # f_x.close()
    # f_y.close()

    return count

def CodeChefExperiment(problem = 'MNMX'):
    # tokenvec.txt and toktypeDict.txt must be put in the data directory
    datafiles = {}
    xypath = params.xypath

    # problem: MNMX, FLOW016, SUBINC
    dtb = CodeChef(problem=problem)
    jsontrain = dtb.getTrainName()
    jsonCV = dtb.getCVName()
    jsontest = dtb.getTestName()

    print 'Construct net from:', jsontrain, jsonCV, jsontest
    datafiles['train'] = [datapath + jsontrain, xypath + dtb.problem + '_gcnn_train_Xnet',
                          xypath + dtb.problem + '_gcnn_train_Y.txt']
    datafiles['CV'] = [datapath + jsonCV, xypath + dtb.problem + '_gcnn_CV_Xnet', xypath + dtb.problem + '_gcnn_CV_Y.txt']
    datafiles['test'] = [datapath + jsontest, xypath + dtb.problem + '_gcnn_test_Xnet', xypath + dtb.problem + '_gcnn_test_Y.txt']

    # jsonfile, xfile, yfile = datafiles['train']
    # constructNetFromJson(jsonFile=jsonfile, xfile=xfile, yfile=yfile)
    numInst = {}
    for fold in datafiles:
        jsonfile, xfile, yfile = datafiles[fold]
        # write X, Y to file
        f_x = file(xfile, 'wb')
        f_y = file(yfile, 'w')
        numInst[fold] = constructNetFromJson(jsonFile=jsonfile, f_x=f_x, f_y=f_y)
        f_x.close()
        f_y.close()

    # write setting content
    # write setting content
    print 'setting parameters:'
    commonFunctions.generateSettingContent(xypath+'../settings_'+dtb.problem+'.txt',
        {'numtrain': numInst['train'], 'numcv': numInst['CV'], 'numtest': numInst['test'], 'output': numOut,
         'paramFile': paramFile, 'xtrain': dtb.problem + '_gcnn_train_Xnet', 'xcv': dtb.problem + '_gcnn_CV_Xnet',
         'xtest': dtb.problem + '_gcnn_test_Xnet',
         'ytrain': dtb.problem + '_gcnn_train_Y.txt', 'ycv': dtb.problem + '_gcnn_CV_Y.txt',
         'ytest': dtb.problem + '_gcnn_test_Y.txt', 'database': problem})

def GraphJsonKFold(K = 5, path=''):
    for idx in range(5, K+1):
        print 'Begin Fold', idx
        jsonObjs = []
        #get graphs from Nonvirus Dir
        gjson = GraphData_IO.getGraphsFromDataDir(path + 'Fold' + str(idx) + '/NonVirus/', classlabel=0)
        jsonObjs.extend(gjson)
        # get graphs from Virus Dir
        gjson = GraphData_IO.getGraphsFromDataDir(path+'Fold'+str(idx)+'/Virus/', classlabel=1)
        jsonObjs.extend(gjson)

        # shuffle the data
        np.random.seed(314159)
        np.random.shuffle(jsonObjs)
        # Write to file
        with open(path+'Fold'+str(idx)+'/dataFold'+str(idx),'w') as f:
            json.dump(jsonObjs,f)
def saveXYForKFold(K=5, datapath='', xypath=''):
    print 'datapath:', datapath
    print 'xypath:', xypath
    for idx in range(1, K+1):
        # write X, Y test from current folds
        f_x = open(xypath+ 'Fold'+str(idx)+'_Xtest','wb')
        f_y = open(xypath+ 'Fold'+str(idx)+'_Ytest.txt','w')
        numtest = constructNetFromJson(datapath+'Fold'+str(idx)+'/dataFold'+str(idx), f_x=f_x, f_y = f_y)

        # return

        f_x.close()
        f_y.close()
        # Merge other folds, Write X, Y for training
        f_x = open(xypath+ 'Fold'+str(idx)+'_Xtrain','wb')
        f_y = open(xypath+ 'Fold'+str(idx)+'_Ytrain.txt','w')
        numtrain =0
        for idx_train in range(1, K+1):
            if idx_train == idx:
                continue
            numtrain += constructNetFromJson(datapath+'Fold'+str(idx_train)+'/dataFold'+str(idx_train), f_x=f_x, f_y = f_y)
        # write setting file
        commonFunctions.generateSettingContent(xypath + '../settings_Fold'+str(idx)+'.txt',
                                               {'numtrain': numtrain, 'numcv': numtrain, 'numtest': numtest, 'output': numOut,
                                                'paramFile': paramFile, 'xtrain': 'Fold'+str(idx)+'_Xtrain', 'xcv': 'Fold'+str(idx)+'_Xtrain',
                                                'xtest': 'Fold'+str(idx)+'_Xtest',
                                                'ytrain': 'Fold'+str(idx)+'_Ytrain.txt', 'ycv': 'Fold'+str(idx)+'_Ytrain.txt',
                                                'ytest': 'Fold'+str(idx)+'_Ytest.txt', 'database': 'Virus'})
        f_x.close()
        f_y.close()

def saveXYForKFoldTrainCVTest(K=5, datapath='', xypath=''):
    print 'datapath:', datapath
    print 'xypath:', xypath
    for idx in range(1, K+1):
        f_x = open(xypath+ 'Fold'+str(idx)+'_Xtrain','wb')
        f_y = open(xypath+ 'Fold'+str(idx)+'_Ytrain.txt','w')
        numtrain = constructNetFromJson(datapath+'Fold'+str(idx)+'_cfg_train', f_x=f_x, f_y = f_y)
        f_x.close()
        f_y.close()

        f_x = open(xypath + 'Fold' + str(idx) + '_XCV', 'wb')
        f_y = open(xypath + 'Fold' + str(idx) + '_YCV.txt', 'w')
        numCV = constructNetFromJson(datapath + 'Fold' + str(idx) + '_cfg_CV', f_x=f_x, f_y=f_y)
        f_x.close()
        f_y.close()

        f_x = open(xypath + 'Fold' + str(idx) + '_Xtest', 'wb')
        f_y = open(xypath + 'Fold' + str(idx) + '_Ytest.txt', 'w')
        numtest = constructNetFromJson(datapath + 'Fold' + str(idx) + '_cfg_test', f_x=f_x, f_y=f_y)
        f_x.close()
        f_y.close()
        # write setting file
        commonFunctions.generateSettingContent(xypath + '../settings_Fold'+str(idx)+'.txt',
                                               {'numtrain': numtrain, 'numcv': numCV, 'numtest': numtest, 'output': numOut,
                                                'paramFile': paramFile, 'xtrain': 'Fold'+str(idx)+'_Xtrain', 'xcv': 'Fold'+str(idx)+'_XCV',
                                                'xtest': 'Fold'+str(idx)+'_Xtest',
                                                'ytrain': 'Fold'+str(idx)+'_Ytrain.txt', 'ycv': 'Fold'+str(idx)+'_YCV.txt',
                                                'ytest': 'Fold'+str(idx)+'_Ytest.txt', 'database': 'F'+str(idx)})
        f_x.close()
        f_y.close()

if __name__ == "__main__":
    # tokenvec.txt and toktypeDict.txt must be put in the data directory
    # Write X, Y for CodeChef Problem
    # problem: MNMX, FLOW016, SUBINC, SUMTRIAN
    # if common_params.reConstruct:
    #     word_dict['While'] = word_dict['DoWhile'] = word_dict['For']
    # # print word_dict
    problems = ['SUMTRIAN', 'FLOW016', 'MNMX', 'SUBINC']
    for pro in problems:
        CodeChefExperiment(problem=pro)

    # Write graph data for K Fold
    # GraphJsonKFold(K=5, path=datapath)
    # Write X, Y for Virus Problem
    # saveXYForKFold(K=5, datapath=datapath, xypath = params.xypath)
    # saveXYForKFoldTrainCVTest(K=5, datapath=datapath, xypath = params.xypath)
    print 'Done!!'
