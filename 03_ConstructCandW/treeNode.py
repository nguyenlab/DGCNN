import sys
from newick import loads

class Node:
    content = None
    parent = None
    children = None
    pos = 0
    def __init__(self, content, parent, children):
        self.content = content
        self.parent = parent
        if children is not None:
            self.children = children
        else:
            self.children = []
    def show(self, buf = sys.stdout,offset=0):
        lead = ' ' * offset
        buf.write(lead + self.content+ '\n')

        for child in self.children:
            child.show(
                buf,
                offset=offset + 2)

    def toString(self,strValue=None, delimitor='('):
        if delimitor == '(':
            d1 = '('
            d2 = ')'
        else:
            d1 = '{'
            d2 = '}'
        strValue.append(self.content)

        for child in self.children:
            strValue.append(d1)
            child.toString(strValue = strValue, delimitor=delimitor)
            strValue.append(d2)
def LoadTree(filename=None): # stanford parse tree format
    #filename = 'D:/graphs.txt'
    reader = open(filename, 'r')

    nodes = {}

    for line in reader:
        line = line.replace('\n', '')
        words = line.split('-')
        if words[0] not in nodes.keys():
            nodes[words[0]] = Node(words[0],None, None)
        if words[1] not in nodes.keys():
            nodes[words[1]] = Node(words[1],None, None)
        parent = nodes[words[0]]
        child = nodes[words[1]]
        parent.children.append(child)

    return nodes['root']

def loadNewickTree(text):
    newick_root = loads(text)
    root = Node('', None, None)
    convertFromNewIck(root, newick_root[0])
    return root
def convertFromNewIck(parent,newick_node):
    if parent.content=='':
        parent.content = str(newick_node.name)
        c_node = parent
    else:
        c_node = Node(str(newick_node.name), parent=parent, children=None)
        parent.children.append(c_node)
    for child in newick_node.descendants:
        convertFromNewIck(c_node, child)


def LoadTokenMap(tokfile=None):
    reader = open(tokfile, 'r')
    tokenMap ={}

    for line in reader:
        line = line.replace('\n', '')
        words = line.split('-')
        for word in words:
            if word not in tokenMap.keys():
                tokenMap[word] = len(tokenMap)

    return tokenMap

# text ='(((((IdentifierType)TypeDecl)FuncDecl)Decl,(((IdentifierType)TypeDecl)Decl,(ID,ID)BinaryOp,((ID)UnaryOp,(((IdentifierType)TypeDecl)Decl,(ID,ID)BinaryOp,(((IdentifierType)TypeDecl,ID)ArrayDecl)Decl,((((IdentifierType)TypeDecl,Constant)Decl)DeclList,(ID,ID)BinaryOp,(ID)UnaryOp,((ID,(ID,ID)ArrayRef)BinaryOp,(ID,(ID,(ID,ID)BinaryOp)ExprList)FuncCall)Compound)For,((ID,((ID,Constant)ArrayRef,(ID,Constant)BinaryOp)BinaryOp)BinaryOp,ID)BinaryOp)Compound)While,(Constant)Return)Compound)FuncDef)FileAST'
# root = loadNewickTree(text)
# root.show()