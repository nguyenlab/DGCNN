import constructNetwork_TBCNN_Sib as TC
import cPickle as p
from nn import serialize

import nn
import pycparser
from nn import FFNN
import sys,os
sys.path.append('../nn')
from pycparser import c_parser, c_ast, parse_file

from InitParam import *

import gl
from nn import Token
import numpy as np
sys.setrecursionlimit(1000000)

datadir =  'D:/data/original_data/'

destfile = 'D:/data/prun_semantic/ast2branches.txt'
f = open(destfile,'w')

procount = 0;
parser = pycparser.c_parser.CParser()

for subdir in os.listdir(datadir):
    cat = subdir
    if not subdir.endswith('/'):
        subdir = subdir + '/'
    #print '111111111111111111111'
    '''
    if len(os.listdir(datadir + subdir))<550:
        continue
    '''
    count = 0
    procount+=1
    print '!!!!!!!!!!!!!!!!!!  procount = ',procount
    #if procount >3:
    #    break
    for onefile in os.listdir(datadir + subdir):

        #print 'oneoneoneoneone!!!!!!!!!! '
        filename = onefile
        onefile = datadir + subdir + onefile
        try:
            #print '-----------------', onefile, '-----------------'
            #ast = parse_file(onefile, use_cpp=True)
            if onefile.endswith(".txt"):
                instream = open(onefile, 'r')
                text = instream.read()
                instream.close()
                #print (text)
                ast = parser.parse(text=text)
                # get major function
                #ast = GetMajorFunction(ast)
                ast.reConstruct()
                #reconstruct data

                #ast.show()
                # print(ast)
                f.write(filename)
                f.write("\t\tAST\t\t")
                f.write(cat + '\t\t')
                ast.savepathsroot2leaf(rootpath='', f=f, branchsepa = '|')
                f.write('\n')

                # ast.show()
                # print(ast)
                # f.write(onefile)
                # f.write("\t\tAST\t\t")
                # f.write(cat + '\t\t')
                # ast.exporttofile(offset=0, attrnames=False, nodenames=False, _my_node_name='PVA', f=f)
                # (self, buf=sys.stdout, offset=0, attrnames=False, nodenames=False, showcoord=False, _my_node_name=None)
                # ast.traverse2paragraph(buf = f,offset=0, attrnames=False, nodenames=False, _my_node_name='PVA')
                # f.write('\n')
                #print 'ooooops, parsing error for', onefile
            #print ast
        except:
            print 'ooooops, parsing error for', onefile
            continue