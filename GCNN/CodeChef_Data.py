import sys
sys.path.append('../pycparser')
sys.path.append('../')
sys.path.append('../03_ConstructCandW')
from pycparser import c_parser

import json
import os
import numpy as np
import gl

import gcnn_params as params
import common_params
from TreeData_IO import tree2Graph
from main_statistic import AST_Statistics

problem ='SUMTRIAN' # FLOW016,  MNMX, SUBINC, SUMTRIAN
datapath = params.datapath # 'Z:/Experiment/CodeChef/'
astjson ='_AST.json'
graphjson ='_AstGraph.json'
randomseed = 314159

labels_dict ={}
labels_dict['tick-icon.gif'] = 0 # correct answer
labels_dict['cross-icon.gif'] = 1 # wrong answer
labels_dict['alert-icon.gif'] = 2 #
labels_dict['clock_error.png'] = 3 #
labels_dict['runtime-error.png'] = 4 #
parser = c_parser.CParser()
def getSourceCode(loc): # lines of code
    code =''
    m_comment=[]
    for line in loc:
        content = line.strip()
        idx = line.find('/*')
        if idx>=0:
            m_comment.append(1)
            cmd = line[:idx]
            line = line[idx+2:]
            code +='\n' + cmd
        idx = line.find('*/')
        if idx>=0:
            if len(m_comment)>0:
                m_comment.pop()
            cmd = line[idx+2:]
            code +='\n' + cmd
            continue
        if len(m_comment)>0:
            continue
        idx = line.find('//')
        if idx>=0:
            line = line[:idx]
        if content.startswith('#include') or content.startswith('using') \
                or content.startswith('#define'):
            continue
        code += '\n' + line
    return code
def loadjSonCode(datafile, outerror):

    with open(datafile, 'r') as infile:
        json_Objs = json.load(infile)

    data =[]
    for obj in json_Objs:
        lang, code, label = obj['lang'],obj['code'], obj['check_box']
        idx = label.rfind('/')
        label = label[idx+1:]
        text = getSourceCode(code)
        # parse source code
        try:
            ast = parser.parse(text=text)  # Parse code to AST
            # print text
            # print type(ast)
            if ast.NodeNum()<10:
                # print ast.NodeNum()
                raise NameError('No code')
            # common_params.ignoreDecl = True  # prune declaration branches
            # common_params.reConstruct = True  # rename While, DoWhile, For ==> Loop

            # ast = parser.parse(text=text)  # Parse code to AST
            # if common_params.reConstruct:  # reconstruct braches of For, While, DoWhile
            #     ast.reConstruct()

            if label not in labels_dict:
                print 'not found:', label
                labels_dict[label] = len(labels_dict)
            label = labels_dict[label]
            # ast.show()
            data.append((label, ast))

            # common_params.ignoreDecl = False  # prune declaration branches
            # common_params.reConstruct = False  # rename While, DoWhile, For ==> Loop
        except Exception as ex:
            # print ex
            if not lang.upper().startswith('C'):
                continue
            outerror.write('\n\nLabel: '+str(label)+'\n')
            text ='\n'.join(code)
            outerror.write(text.encode("UTF-8"))
    # common_params.ignoreDecl = True  # prune declaration branches
    # common_params.reConstruct = True  # rename While, DoWhile, For ==> Loop
    # data[0][1].show()
    return data
def ReadData():
    outerror = open(datapath+problem+'error.txt','w')
    ast_data =[]
    count =0
    for onefile in os.listdir(datapath + problem+'/'):
        codefile = datapath + problem+'/'+onefile
        onefile_data = loadjSonCode(codefile, outerror)
        # onefile_data[0][1].show()
        ast_data.extend(onefile_data)

        count +=1
        # if count>10:
        #     break
    outerror.close()
    # write ast to json
    with open(datapath+problem+ astjson, 'w') as outfile:
        jsonObjs =[]
        for label, ast in ast_data:
            ast_str = []
            ast.toNewickFormat(value=ast_str)
            ast_str = ''.join(ast_str)
            # print text
            jsonObjs.append({'label':label,'ast':ast_str})
        json.dump(jsonObjs, outfile)
    # write ast graph to json
    with open(datapath+problem+ graphjson, 'w') as outfile:
        jsonObjs =[]
        for label, ast in ast_data:
            g =  tree2Graph(ast)
            g.label = label
            jsonObjs.append(g.dump())
        json.dump(jsonObjs, outfile)
    # write statistics
    asts = [x[1] for x in ast_data]
    AST_Statistics(asts,datapath+problem+'_AST_statistic' )
    print '# Instances:', len(ast_data)

def generateTrain_CV_Test(datafile='', type ='AST'):
    # Read Json File, Split into Train, CV, Test
    # type = AST or Graph
    with open(datafile, 'r') as f:
        jsonObjs = json.load(f)

    name =''
    if type =='AST':
        name = astjson
    else:
        name = graphjson

    numInst = len(jsonObjs)
    # shuffle data

    np.random.seed(314159)
    np.random.shuffle(jsonObjs)

    print 'Instances:', len(jsonObjs)
    numTrain = int(.6 * numInst)
    numCV = int(.8* numInst)
    print 'numTrain : ',numTrain
    print 'numCV : ' ,numCV-numTrain
    print 'numTest', numInst -numCV
    # write train data
    f = open(datapath+problem+'_train'+ name,'w')
    pdata = jsonObjs[:numTrain]
    json.dump(pdata, f)
    f.close()
    # write CV data
    f = open(datapath+problem+'_CV'+ name,'w')
    pdata = jsonObjs[numTrain:numCV]
    json.dump(pdata, f)
    f.close()
    # write test data
    f = open(datapath+problem+'_test'+ name,'w')
    pdata = jsonObjs[numCV:]
    json.dump(pdata, f)
    f.close()

def showTree(node, offset=0):
    print ' ' * offset + str(node.name)
    for child in node.descendants:
        showTree(child, offset=offset + 2)
if __name__ =='__main__':
    # read data from crawler directory, parse to AST, Save in Json format
    ReadData()
    # split into Train -CV - Test
    generateTrain_CV_Test(datapath+problem+astjson, type='AST')
    generateTrain_CV_Test(datafile=datapath+problem+graphjson, type='Graph')

    # text ='''
    # int main()
    # {
    #     int x=3;
    #     if (x+2>3)
    #         print ("hello");
    # }
    # '''
    # text=[]
    # text.append('/*Name: Pham Ngoc Cuong')
    # text.append('ID:se04888')
    # text.append('workshop int y;*/')
    # text.append('#include <stdio.h>')
    # text.append('int main(){')
    # text.append(' long long t, n, a, i, sum, min; /* this command is ok */')
    # text.append(' scanf("%lld", &t);')
    # text.append(' while (t--){')
    # text.append('  scanf("%lld", &n);')
    # text.append('  scanf("%lld", &a);')
    # text.append('  min = a;')
    # text.append('  for (i = 1; i < n; i++){')
    # text.append('   scanf("%lld", &a);')
    # text.append('   if (a < min)')
    # text.append('    min = a; ')
    # text.append('  }')
    # text.append('  sum = min * (n-1);')
    # text.append('  printf("%lld", sum);')
    # text.append(' }')
    # text.append(' return 0;')
    # text.append(' }')
    # print getSourceCode(text)

    # parser = pycparser.c_parser.CParser()
    # ast = parser.parse(text=text)  # Parse code to AST
    # ast.show()
    # #
    # ast_str =[]
    # ast.toNewickFormat(value = ast_str)
    # text = ''.join(ast_str)
    # print text
    # print 'Load Tree NewIck'
    # text ='(((((IdentifierType)TypeDecl)FuncDecl)Decl,(((IdentifierType)TypeDecl)Decl,(ID,ID)BinaryOp,((ID)UnaryOp,(((IdentifierType)TypeDecl)Decl,(ID,ID)BinaryOp,(((IdentifierType)TypeDecl,ID)ArrayDecl)Decl,((((IdentifierType)TypeDecl,Constant)Decl)DeclList,(ID,ID)BinaryOp,(ID)UnaryOp,((ID,(ID,ID)ArrayRef)BinaryOp,(ID,(ID,(ID,ID)BinaryOp)ExprList)FuncCall)Compound)For,((ID,((ID,Constant)ArrayRef,(ID,Constant)BinaryOp)BinaryOp)BinaryOp,ID)BinaryOp)Compound)While,(Constant)Return)Compound)FuncDef)FileAST'
    # from newick import loads
    # root = loads(text)
    # showTree(root[0],0)
    print 'Done'


