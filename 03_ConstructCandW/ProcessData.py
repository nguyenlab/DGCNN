# parse source code ---> AST
# split into: train, CV, test
import json
import os
import numpy as np
import gl
import pycparser
import common_params

def sourceCode2AST(text):
    parser = pycparser.c_parser.CParser()
    ast = parser.parse(text=text)  # Parse code to AST
    if common_params.reConstruct:  # reconstruct braches of For, While, DoWhile
        ast.reConstruct()
    return ast
def sourceFile2AST(filename):
    with open(filename,'r') as f:
        text = f.read()
        ast = sourceCode2AST(text)
    return ast
# for database: each problem is contained in a directory
def generateTrain_CV_Test_AST_Json(datadir='', ratio =[0.6,0.2,0.2]):
    data =[]
    for pi in xrange(1, gl.numOut + 1):
        subdir = str(pi) + '/'
        for onefile in os.listdir(datadir + subdir):
            onefile = datadir + subdir + onefile
            data.append((onefile, pi - 1))
    # split into train,CV, test
    np.random.seed(314159)
    np.random.shuffle(data)

    numInst = len(data)
    print 'Instances =', numInst

    numTrain = int(ratio[0] * numInst)
    numCV = int((ratio[0]+ ratio[1]) * numInst)
    print 'numTrain : ', numTrain
    print 'numCV : ', numCV - numTrain
    print 'numTest', numInst - numCV
    jsonObjs = []


    path = common_params.jsondir + common_params.dataname
    # write train AST in json format
    for idx in range(0, numTrain):
        onefile, label = data[idx]
        ast_str = []
        ast = sourceFile2AST(onefile)
        ast.toNewickFormat(value=ast_str)
        ast_str = ''.join(ast_str)

        jsonObjs.append({'label': label, 'ast': ast_str})

    with open(path+common_params.jsonfold['train'], 'w') as f:
        json.dump(jsonObjs,f)

    jsonObjs = []
    # write CV AST in json format
    for idx in range(numTrain, numCV):
        onefile, label = data[idx]
        ast_str = []
        ast = sourceFile2AST(onefile)
        ast.toNewickFormat(value=ast_str)
        ast_str = ''.join(ast_str)

        jsonObjs.append({'label': label, 'ast': ast_str})

    with open(path+common_params.jsonfold['CV'], 'w') as f:
        json.dump(jsonObjs, f)

    jsonObjs = []
    # write test AST in json format
    for idx in range(numCV, numInst):
        onefile, label = data[idx]
        ast_str = []
        ast = sourceFile2AST(onefile)
        ast.toNewickFormat(value=ast_str)
        ast_str = ''.join(ast_str)

        jsonObjs.append({'label': label, 'ast': ast_str})

    with open(path + common_params.jsonfold['test'], 'w') as f:
        json.dump(jsonObjs, f)


generateTrain_CV_Test_AST_Json(datadir=common_params.datadir);