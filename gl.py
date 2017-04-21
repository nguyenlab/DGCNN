# -*- coding: utf-8 -*-
"""
Created on Tue May 20 16:01:35 2014

@author: mou
"""
# # settings
# ignoreDecl = False # prune declaration branches
# reConstruct = False # rename While, DoWhile, For ==> Loop
# tokenMap ='../tokenMap.txt'
# # datadir =  'D:/data/original_data/'
# # targetdir = 'D:/data/prun_semantic/TBCNN_RvNN/network/'
# #targetdir ='C:/Users/anhpv/Desktop/RNN_Networks/'
# # data dir for stack overflow data
# datadir = 'D:/StackOverFlowData/Data/Experiments/data/'
# targetdir = 'D:/StackOverFlowData/Data/Experiments/TBCNN/network/'

margin = 1
learnRate = 0.0025 # 0.0001
beta = .0001
momentum = 0.1

batchsize = 100

decay = momentum/(1-momentum)

alpha = learnRate * (1-momentum) / batchsize
 


encounterNAN = False
# for numerical gradient checking
# numFea = 30#200
# numCon = 30 # 50
# numRecur = 600 #num recursive
# numDis = 100#600 # 50
# numOut = 104
# original parameters for TBCNN
numFea = 30#200
numCon = 600 # 50
numDis = 600 # 50
numOut = 2
#numWords = 209392 #400000#209392#100232# 209392
numPool = 3


