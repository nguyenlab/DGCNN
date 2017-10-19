import cPickle as p
# settings
import os

ignoreDecl = False # prune declaration branches
reConstruct = False # rename While, DoWhile, For ==> Loop

datadir =  '' # source code directory
# jsondir = '/home/s1520015/Experiment/CodeChef/OriginalTrees/' #
# xypath = '/home/s1520015/Experiment/CodeChef_TwoClass/SibStCNN_2C/xy/'

# jsondir = 'Z:/Experiment/CodeChef/OriginalTrees/' #
# xypath = 'Z:/Experiment/CodeChef_TwoClass/SibStCNN_2C/xy/'

jsondir = '/home/s1520015/Experiment/ASMCFG/Nets/' #
xypath = '/home/s1520015/Experiment/ASMCFG/Nets/BOW/'

tokenMapFile = '../tokenMap.txt'
tokenMap = None#p.load(open(tokenMapFile))

dataname ='SUMTRIAN' #'SUMTRIAN, FLOW016, MNMX, SUBINC

jsonfold={}
jsonfold['train'] = '_AST_train.json'#'_train_AST.json'
jsonfold['CV'] = '_AST_CV.json' #'_CV_AST.json'
jsonfold['test'] = '_AST_test.json' #'_test_AST.json'
