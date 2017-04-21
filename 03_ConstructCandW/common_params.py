import cPickle as p
# settings
import os

ignoreDecl = False # prune declaration branches
reConstruct = False # rename While, DoWhile, For ==> Loop

datadir =  '' # source code directory
jsondir = '/home/s1520015/Experiment/CodeChef/PrunedTrees/' #
xypath = '/home/s1520015/Experiment/CodeChef_PruningTree/'#jsondir + 'xy/'

tokenMapFile = '../tokenMap.txt'
tokenMap = None#p.load(open(tokenMapFile))

dataname ='MNMX' #'SUMTRIAN, FLOW016, MNMX, SUBINC

jsonfold={}
jsonfold['train'] = '_train_AST.json'
jsonfold['CV'] = '_CV_AST.json'
jsonfold['test'] = '_test_AST.json'
