import json
import sys
sys.path.append('../')
sys.path.append('../03_ConstructCandW')

import gl

import common_params
from treeNode import loadNewickTree
# config net structures
def getLabel(label):
    if label >0 and gl.numOut==2:
            return 1
    return label
def writeXY(jsonAST='', outfile =''):
    count =0
    with open(jsonAST,'r') as f:
        jsonObjs = json.load(f)
    f_out = open(outfile,'wb')
    for  obj in jsonObjs:
        label = getLabel(obj["label"])
        f_out.write(str(label)+' {')
        ast_newick = obj["ast"]
        ast = loadNewickTree(text=ast_newick)
        ast_seq =[]
        ast.toString(strValue=ast_seq,delimitor='{')
        f_out.write(''.join(ast_seq))
        f_out.write('}\n')
        count+=1
    f_out.close()
    return count

if __name__ == "__main__":
    for dataname in ['SUMTRIAN', 'FLOW016', 'MNMX', 'SUBINC']:
        # dataname = common_params.dataname
        jsondir =  common_params.jsondir
        numInst ={}
        for fold in ['train','CV','test']:
            jsonAST = jsondir+ dataname + common_params.jsonfold[fold]
            outfile = dataname+ '_XY_'+ fold

            numInst[fold] = writeXY(jsonAST, common_params.xypath + outfile)

    print 'Done!!'
