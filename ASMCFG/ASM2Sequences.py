import os
import re

from enum import Enum
import sys
sys.path.append('../Data_IO')
sys.path.append('../GCNN')
import  ASM2CFG
from ASM2CFG import getParamType
import numpy as np
from ReadData import saveArray2Text

def DOCfromASM(asmfile ,ignoreDecl = True, useOp = False, sen_Sepa='<sssss>'):
    # useOp: instruction name and operands
    # sen_sepa: the symbol which separates sentences
    doc =[]
    blocks, blocks_dict = ASM2CFG.getBlocks(asmfile, ignoreDecl)
    for b in blocks:
        sen =[]
        for inst in b.instructions:
            if not useOp:
               sen.append(inst.name)
            else:
                content = inst.name
                for p in inst.params:
                    content += '_'+ getParamType(p)
                sen.append(content)
        if len(sen)>0:
            doc.append(' '.join(sen)) # create the sentence from the set of words
    doc_seq = sen_Sepa.join(doc)
    return doc_seq # create and return the document

def generateTrain_CV_Test(data_dir ='', problems =[], dest_dir='',useOp = False):
    seed = 314159
    for prob in problems:
        doc_data = []

        c_dir = data_dir + prob+'/' # path to C code
        s_dir = data_dir + prob + '_S/' # path to asm code
        for c_file in os.listdir(c_dir):
            if c_file.endswith('.c'):
                id1 = c_file.find('_')
                id2 = c_file.find('.c')
                label = (int)(c_file[id1 + 1:id2])
                # read ASM file
                doc = DOCfromASM(s_dir+c_file+'.s',ignoreDecl = True, useOp = useOp, sen_Sepa='<sssss>')
                doc_data.append([label, doc])

        # shuffle data
        np.random.seed(seed)
        np.random.shuffle(doc_data)


        numTrain = int(.6 * len(doc_data))
        numCV = int(.8 * len(doc_data))

        print 'Data', prob
        print 'numTrain : ', numTrain
        print 'numCV : ', numCV - numTrain
        print 'numTest', len(doc_data) - numCV

        # training data
        txt_docs = []
        for i in xrange(0, numTrain):
            (label, doc) = doc_data[i]
            txt_docs.append('\t\t'.join(['N','N',str(label),doc]))

        op_ext = ''
        if useOp:
            op_ext ='_Op'
        saveArray2Text(txt_docs,dest_dir+'{0}_Seq{1}_train.txt'.format(prob, op_ext))

        # CV data
        txt_docs = []
        for i in xrange(numTrain, numCV):
            (label, doc) = doc_data[i]
            txt_docs.append('\t\t'.join(['N', 'N', str(label), doc]))

        saveArray2Text(txt_docs, dest_dir + '{0}_Seq{1}_CV.txt'.format(prob, op_ext))

        # test data
        txt_docs =[]
        for i in xrange(numCV, len(doc_data)):
            (label, doc) = doc_data[i]
            txt_docs.append('\t\t'.join(['N', 'N', str(label), doc]))

        saveArray2Text(txt_docs, dest_dir + '{0}_Seq{1}_test.txt'.format(prob, op_ext))

data_dir='/home/s1520015/Experiment/ASMCFG/SourceCode/'
problems=['FLOW016','MNMX','SUBINC','SUMTRIAN']
dest_dir='/home/s1520015/Experiment/ASMCFG/Nets/'
generateTrain_CV_Test(data_dir=data_dir, problems=problems, dest_dir=dest_dir,useOp = True)
# asmfile='D:/C_SourceCode/Code/testfunc.s'
# asmfile = 'Z:/Experiment/ASMCFG/SourceCode/FLOW016_S/06444_3.c.s'
# d = DOCfromASM(asmfile,True,True)
# print d