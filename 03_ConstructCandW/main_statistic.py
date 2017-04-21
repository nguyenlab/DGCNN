import pycparser
import sys, os

sys.path.append('../nn')

from InitParam import *

import gl
from nn import Token
import numpy as np

sys.setrecursionlimit(1000000)

def computeLeafNum(root, nodes, depth=0):
    if len(root.children) == 0:
        root.leafNum = 1
        root.childrenNum = 1
        return 1, 1, depth  # leafNum, childrenNum
    root.allLeafNum = 0
    avgdepth = 0.0
    for child in root.children:
        leafNum, childrenNum, childAvgDepth = computeLeafNum(nodes[child], nodes, depth + 1)
        root.leafNum += leafNum
        root.childrenNum += childrenNum
        avgdepth += childAvgDepth * leafNum
    avgdepth /= root.leafNum
    root.childrenNum += 1
    return root.leafNum, root.childrenNum, avgdepth


def ConstructNodes(ast, name, parent, pos, nodes, leafs):
    if name == None:
        name = ast.__class__.__name__

    Node = Token.token(name , 0, parent, pos)
    if len(ast.children()) == 0:
        leafs.append(Node)
    else:
        nodes.append(Node)

    curid = len(nodes)
    for idx, (name, child) in enumerate(ast.children()):
        ConstructNodes(child, None, curid, idx, nodes, leafs)

def AdjustOrder(nodes):
    nodes.reverse()
    length = len(nodes)
    for node in nodes:
        if node.parent != None:
            node.parent = length - node.parent

import func_defs

def GetMajorFunction(root):
    v = func_defs.FuncDefVisitor()
    v.visit(root)
    nodes = v.nodes;

    major = nodes[0]
    for cnode in nodes:
        if (major.NodeNum() < cnode.NodeNum()):
            major = cnode
    return major;
def AST_Statistics(asts=[], outname=''):

        out = open(outname,'w')
        for ast in asts:
            nodes = []
            leafs = []
            # print 'constructing the network'
            ConstructNodes(ast, 'root', None, None, nodes, leafs)
            nodes.extend(leafs)
            AdjustOrder(nodes)

            for nidx in xrange(len(nodes)):
                if nodes[nidx].parent != None:
                    nodes[nodes[nidx].parent].children.append(nidx)

            leafNum, childrenNum, childAvgDepth = computeLeafNum(nodes[-1], nodes)
            out.write(str(leafNum)+',')
            out.write(str(childrenNum)+',')
            out.write(str(childAvgDepth)+'\n')

        out.close()
def GenerateAST_Data(filename = '', out = None, justMajorFunc = False, reConstruct = False, ignoreDecl = False):

    parser = pycparser.c_parser.CParser()
    # print '-----------------', onefile, '-----------------'
    # ast = parse_file(onefile, use_cpp=True)
    if filename.endswith(".txt"):
        instream = open(filename, 'r')
        text = instream.read()
        instream.close()

        from  pycparser import c_ast
        c_ast.ignoreDecl= ignoreDecl

        ast = parser.parse(text=text)
        # get major function
        if justMajorFunc== True:
            ast = GetMajorFunction(ast)
        if reConstruct == True:
            ast.reConstruct()

        nodes = []
        leafs = []
        # print 'constructing the network'
        ConstructNodes(ast, 'root', None, None, nodes, leafs)
        nodes.extend(leafs)
        AdjustOrder(nodes)

        for nidx in xrange(len(nodes)):
            if nodes[nidx].parent != None:
                nodes[nodes[nidx].parent].children.append(nidx)

        leafNum, childrenNum, childAvgDepth = computeLeafNum(nodes[-1], nodes)
        out.write(str(leafNum)+',')
        out.write(str(childrenNum)+',')
        out.write(str(childAvgDepth)+'\n')


# datadir = 'D:/data/original_data/'
# procount = 0;
# parser = pycparser.c_parser.CParser()
#
# filename = 'D:/statistic_mf.txt'
# out = open(filename, 'w')
#
# for subdir in os.listdir(datadir):
#     if not subdir.endswith('/'):
#         subdir = subdir + '/'
#
#     count = 0
#     procount += 1
#     print '!!!!!!!!!!!!!!!!!!  procount = ', procount
#     # if procount >3:
#     #    break
#     for onefile in os.listdir(datadir + subdir):
#
#         # print 'oneoneoneoneone!!!!!!!!!! '
#         filename = onefile
#         onefile = datadir + subdir + onefile
#
#         GenerateAST_Data(onefile, out=out, justMajorFunc=True, reConstruct=False, ignoreDecl=False)
#
# out.close()
# print 'Done!!!!!!'
# print 'procount = ', procount

