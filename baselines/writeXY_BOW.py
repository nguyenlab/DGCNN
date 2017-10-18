import json
import sys
sys.path.append('../')
sys.path.append('../03_ConstructCandW')

import gl

import common_params
from treeNode import loadNewickTree
# config net structures
def getLabel(label):
    if gl.numOut==2:
        if label >0:
            return 1
        else:
            return -1
    return label
def writeXY(jsonfiles=[], outfiles =[], outworddict='', ext ='arff'):
    tokDict={}
    counts =[0]* len(jsonfiles)
    for idx, jsonAST in enumerate(jsonfiles):
        with open(jsonAST,'r') as fin:
            jsonObjs = json.load(fin)
        f_out = open(outfiles[idx],'w')
        for  obj in jsonObjs:
            label = getLabel(obj["label"])
            f_out.write(str(label))
            ast_newick = obj["ast"]
            ast = loadNewickTree(text=ast_newick)
            ast_seq =[]
            delimitors = ['{','}']
            ast.toString(strValue=ast_seq, delimitor= delimitors[0])

            tokNum ={}
            for tok in ast_seq:
                if tok == delimitors[0] or tok ==delimitors[1]:
                    continue
                if tok not in tokDict: # add token to dictionary if it does not exist
                    lenDict = len(tokDict)
                    tokDict[tok] = lenDict +1
                id = tokDict[tok]
                if id not in tokNum:
                    tokNum[id] = 1
                else:
                    tokNum[id] += 1
            vector =''
            for tokid, tokcount in tokNum.items():
                vector += ' {0}:{1}'.format(tokid, tokcount)
            f_out.write(vector +'\n')

            counts[idx] +=1
        f_out.close()
    # write word dict
    with open(outworddict,'w') as f_out:
        for tok, tokid in tokDict.items():
            f_out.write('{0} {1}\n'.format(tok, tokid))
    return counts
if __name__ == "__main__":
    for dataname in ['SUMTRIAN', 'FLOW016', 'MNMX', 'SUBINC']:
    # dataname = common_params.dataname
        jsondir =  common_params.jsondir
        jsonfiles =[]
        outfiles =[]
        for fold in ['train','CV','test']:
            jsonfiles.append(jsondir+ dataname + common_params.jsonfold[fold])
            outfiles.append(common_params.xypath + dataname + '_XY_BoW_'+ fold)

        outworddict = common_params.xypath+dataname+'_worddict'
        numInst = writeXY(jsonfiles=jsonfiles, outfiles=outfiles, outworddict= outworddict)

    print 'Done!!'
