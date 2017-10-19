import json
import sys
import numpy as np
sys.path.append('../GCNN')
sys.path.append('../Data_IO')

from Graph import GVertex, Graph
from ReadData import createWordDict,saveArray

def getAdMatrix(graph, tokdict):
    # convert a graph to a adjacency matrix
    N = len(tokdict)
    ad_matrix = np.zeros((N,N), dtype=np.int)

    Vs = graph.Vs
    Es = graph.Es

    for (v1id, v2id) in Es:
        tok1 = Vs[v1id].token
        tok2 = Vs[v2id].token
        if tok1 not in tokdict:
            print tok1, ' not found'
            continue
        if tok2 not in tokdict:
            print tok2, ' not found'
            continue
        ad_matrix[tokdict[tok1]][tokdict[tok2]] +=1

    return ad_matrix

def CFG2AdMatrices(cfgJson,tokdict, admatrixJson):
    # read all CFGs
    admJson=[]
    with open(cfgJson, 'r') as f:
        jsonObjs = json.load(f)

        # graph = Graph.load(jsonObjs[0])
        # adm = getAdMatrix(graph = graph, tokdict = tokdict)
        for obj in jsonObjs:
            graph = Graph.load(obj)
            adm = getAdMatrix(graph, tokdict)
            admJson.append({'label': graph.label, 'matrix': adm.tolist()})

    with open(admatrixJson, 'w') as outfile:
        json.dump(admJson, outfile)

dictfile ='/home/s1520015/Experiment/ASMCFG/Nets/dict_instruction_name.txt'
admatrixJson = '/home/s1520015/Experiment/adjacency_matrix.txt'
cfgJson= '/home/s1520015/Experiment/ASMCFG/Nets/FLOW016_CFG_test.json'
dict = createWordDict(filename =dictfile)
# # print dict
CFG2AdMatrices(cfgJson = cfgJson ,tokdict= dict, admatrixJson= admatrixJson)