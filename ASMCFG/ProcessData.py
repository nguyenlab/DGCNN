import sys
sys.path.append('../pycparser')
sys.path.append('../Data_IO')

import json
import os
import numpy as np
from ASM2CFG import CFGfromASM
from pycparser import c_parser
from ReadData import saveArray

parser = c_parser.CParser()
def getSourceCode(loc): # lines of code
    # preprovessing: remove comments before parsing into AST
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
def checkFileAndDelete(dir_ccode ='', dir_scode =''):
    # delete syntax error files
    # get successfully parsed files
    asmfiles={}
    for filename in os.listdir(dir_scode):
        asmfiles[filename] = filename

    for filename in os.listdir(dir_ccode):
        if filename+'.s' not in asmfiles:
            os.remove(dir_ccode+ filename)
def generateAllProbInOneTrain_CV_Test(data_dirs =[], dest_dir=''):

    ast_data =[]
    cfg_data =[]

    for d in data_dirs:
        c_dir = d+'/' # path to C code
        s_dir = d + '_S/' # path to asm code
        for c_file in os.listdir(c_dir):
            if c_file.endswith('.c'):
                id1 = c_file.find('_')
                id2 = c_file.find('.c')
                label = (int)(c_file[id1 + 1:id2])
                # read C file parse to ast file
                with open(c_dir+c_file,'r') as f:
                    text = getSourceCode(f.readlines())
                    # print text
                    ast = parser.parse(text=text)  # Parse code to AST
                    ast_str = []
                    ast.toNewickFormat(value=ast_str)
                    ast_str = ''.join(ast_str)
                    # print text
                    ast_data.append((label, ast_str))
                # read ASM file and construct CFG
                g = CFGfromASM(s_dir+c_file+'.s')
                g.label = label
                cfg_data.append(g)

    # shuffle data
    seeds =[314159,267192, 283098, 152385, 928127]
    for fold, s in enumerate(seeds):
        np.random.seed(s)
        np.random.shuffle(ast_data)

        np.random.seed(s)
        np.random.shuffle(cfg_data)

        numTrain = int(.6 * len(ast_data))
        numCV = int(.8 * len(ast_data))

        fold +=1
        print 'Fold', fold
        print 'numTrain : ', numTrain
        print 'numCV : ', numCV - numTrain
        print 'numTest', len(ast_data) - numCV

        # training data
        json_ast =[]
        json_cfg =[]
        for i in xrange(0, numTrain):
            (label, ast) = ast_data[i]
            json_ast.append({'label':label, 'ast': ast})

            g = cfg_data[i]
            json_cfg.append(g.dump())

        # write training data to file
        with open(dest_dir+'Fold{0}_ast_train.txt'.format(fold),'w') as fout:
            json.dump(json_ast, fout)
        with open(dest_dir+'Fold{0}_cfg_train.txt'.format(fold),'w') as fout:
            json.dump(json_cfg, fout)

        # CV data
        json_ast =[]
        json_cfg =[]
        for i in xrange(numTrain, numCV):
            (label, ast) = ast_data[i]
            json_ast.append({'label': label, 'ast': ast})

            g = cfg_data[i]
            json_cfg.append(g.dump())

        # write CV data to file
        with open(dest_dir + 'Fold{0}_ast_CV.txt'.format(fold), 'w') as fout:
            json.dump(json_ast, fout)
        with open(dest_dir + 'Fold{0}_cfg_CV.txt'.format(fold), 'w') as fout:
            json.dump(json_cfg, fout)

        # test data
        json_ast =[]
        json_cfg =[]
        for i in xrange(numCV, len(ast_data)):
            (label, ast) = ast_data[i]
            json_ast.append({'label': label, 'ast': ast})

            g = cfg_data[i]
            json_cfg.append(g.dump())

        # write test data to file
        with open(dest_dir + 'Fold{0}_ast_test.txt'.format(fold), 'w') as fout:
            json.dump(json_ast, fout)
        with open(dest_dir + 'Fold{0}_cfg_test.txt'.format(fold), 'w') as fout:
            json.dump(json_cfg, fout)

# def generateTrain_CV_Test(data_dir ='', problems =[], dest_dir=''):
#     seed = 314159
#     for prob in problems:
#         ast_data = []
#         cfg_data = []
#         source_files =[]
#
#         c_dir = data_dir + prob+'/' # path to C code
#         s_dir = data_dir + prob + '_S/' # path to asm code
#         for c_file in os.listdir(c_dir):
#             if c_file.endswith('.c'):
#                 source_files.append(c_dir + c_file)
#                 id1 = c_file.find('_')
#                 id2 = c_file.find('.c')
#                 label = (int)(c_file[id1 + 1:id2])
#                 # read C file parse to ast file
#                 with open(c_dir+c_file,'r') as f:
#                     text = getSourceCode(f.readlines())
#                     # print text
#                     ast = parser.parse(text=text)  # Parse code to AST
#                     ast_str = []
#                     ast.toNewickFormat(value=ast_str)
#                     ast_str = ''.join(ast_str)
#                     # print text
#                     ast_data.append((label, ast_str))
#                 # read ASM file and construct CFG
#                 g = CFGfromASM(s_dir+c_file+'.s')
#                 g.label = label
#                 cfg_data.append(g)
#
#         # shuffle data
#         np.random.seed(seed)
#         np.random.shuffle(ast_data)
#
#         np.random.seed(seed)
#         np.random.shuffle(cfg_data)
#
#         np.random.seed(seed)
#         np.random.shuffle(source_files)
#
#         numTrain = int(.6 * len(ast_data))
#         numCV = int(.8 * len(ast_data))
#
#         print 'Data', prob
#         print 'numTrain : ', numTrain
#         print 'numCV : ', numCV - numTrain
#         print 'numTest', len(ast_data) - numCV
#
#         # training data
#         json_ast =[]
#         json_cfg =[]
#         txt_source =[]
#         for i in xrange(0, numTrain):
#             (label, ast) = ast_data[i]
#             json_ast.append({'label':label, 'ast': ast})
#
#             g = cfg_data[i]
#             json_cfg.append(g.dump())
#
#             txt_source.append(source_files[i])
#         # write training data to file
#         with open(dest_dir+'{0}_AST_train.json'.format(prob),'w') as fout:
#             json.dump(json_ast, fout)
#         with open(dest_dir+'{0}_CFG_train.json'.format(prob),'w') as fout:
#             json.dump(json_cfg, fout)
#         saveArray(txt_source,dest_dir+'{0}_source_train.txt'.format(prob))
#
#         # CV data
#         json_ast =[]
#         json_cfg =[]
#         txt_source = []
#         for i in xrange(numTrain, numCV):
#             (label, ast) = ast_data[i]
#             json_ast.append({'label': label, 'ast': ast})
#
#             g = cfg_data[i]
#             json_cfg.append(g.dump())
#
#             txt_source.append(source_files[i])
#
#         # write CV data to file
#         with open(dest_dir + '{0}_AST_CV.json'.format(prob), 'w') as fout:
#             json.dump(json_ast, fout)
#         with open(dest_dir + '{0}_CFG_CV.json'.format(prob), 'w') as fout:
#             json.dump(json_cfg, fout)
#         saveArray(txt_source, dest_dir + '{0}_source_CV.txt'.format(prob))
#
#         # test data
#         json_ast =[]
#         json_cfg =[]
#         txt_source = []
#         for i in xrange(numCV, len(ast_data)):
#             (label, ast) = ast_data[i]
#             json_ast.append({'label': label, 'ast': ast})
#
#             g = cfg_data[i]
#             json_cfg.append(g.dump())
#
#             txt_source.append(source_files[i])
#
#         # write test data to file
#         with open(dest_dir + '{0}_AST_test.json'.format(prob), 'w') as fout:
#             json.dump(json_ast, fout)
#         with open(dest_dir + '{0}_CFG_test.json'.format(prob), 'w') as fout:
#             json.dump(json_cfg, fout)
#         saveArray(txt_source, dest_dir + '{0}_source_test.txt'.format(prob))

def generateTrain_CV_Test(data_dir ='', problems =[], dest_dir=''):
    seed = 314159
    for prob in problems:
        source_files =[]

        c_dir = data_dir + prob+'/' # path to C code
        for c_file in os.listdir(c_dir):
            if c_file.endswith('.c'):
                source_files.append(c_dir + c_file)
        # shuffle data

        np.random.seed(seed)
        np.random.shuffle(source_files)

        numTrain = int(.6 * len(source_files))
        numCV = int(.8 * len(source_files))

        print 'Data', prob
        print 'numTrain : ', numTrain
        print 'numCV : ', numCV - numTrain
        print 'numTest', len(source_files) - numCV

        # training data
        txt_source =[]
        for i in xrange(0, numTrain):
            txt_source.append(source_files[i])
        # write training data to file
        saveArray(np.array(txt_source),dest_dir+'{0}_source_train.txt'.format(prob))

        # CV data
        txt_source = []
        for i in xrange(numTrain, numCV):
            txt_source.append(source_files[i])
        # write CV data to file
        saveArray(np.array(txt_source), dest_dir + '{0}_source_CV.txt'.format(prob))

        # test data
        txt_source = []
        for i in xrange(numCV, len(source_files)):
            txt_source.append(source_files[i])

        # write test data to file
        saveArray(np.array(txt_source), dest_dir + '{0}_source_test.txt'.format(prob))

# path ='/home/s1520015/Experiment/ASMCFG/SourceCode/'
# dir_ccode =[path+'SUBINC', path+'FLOW016', path+'MNMX', path+'SUMTRIAN']
# generateAllProbInOneTrain_CV_Test(dir_ccode,'/home/s1520015/Experiment/ASMCFG/SourceCode/')

data_dir ='/home/s1520015/Experiment/ASMCFG/SourceCode/'
dest_dir = '/home/s1520015/Experiment/ASMCFG/Nets/'
problems =['SUBINC', 'FLOW016', 'MNMX', 'SUMTRIAN']
generateTrain_CV_Test(data_dir=data_dir, problems=problems, dest_dir=dest_dir)

# for cdir in dir_ccode:
#     checkFileAndDelete(cdir+'/',cdir+'_S/')
