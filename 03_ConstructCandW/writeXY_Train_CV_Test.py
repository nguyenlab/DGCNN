import sys
sys.path.append('../')
import json
import os

import commonFunctions
import common_params
import gl
from treeNode import loadNewickTree
# config net structures

# import main_RNN as NetStruct
# import main_TBCNN as NetStruct
import main_TBCNN_Sib as NetStruct

def InitNetbyText(text=''): # in newick format
    ast = loadNewickTree(text=text)

    nodes = []
    leafs = []
    # print 'constructing the network'
    commonFunctions.ConstructNodes(ast, 'root', None, None, nodes, leafs, common_params.tokenMap)
    # print len(nodes)
    # print nodes[0].word
    nodes.extend(leafs)
    # print len(nodes)
    commonFunctions.AdjustOrder(nodes)
    # print '---------------------------------------'
    for ii in xrange(len(nodes)):
        # print ii
        inode = nodes[ii]
        # print ii,'  ',inode.word,'  ',inode.parent,'  ',nodes[inode.parent].word,'  ',inode.pos
    layers = NetStruct.InitByNodes(nodes)

    return layers

def getLabel(label):
    if label >0 and gl.numOut==2:
            return 1
    return label
def writeXY(jsonAST='', xfile ='', yfile=''):
    count =0
    with open(jsonAST,'r') as f:
        jsonObjs = json.load(f)
    f_x = open(xfile,'wb')
    f_y = open(yfile,'w')
    for  obj in jsonObjs:
        label = getLabel(obj["label"])
        f_y.write(str(label) + '\n')
        ast_str = obj["ast"]
        layers = InitNetbyText(ast_str)
        commonFunctions.WriteNet(f_x,layers)

        count+=1
    f_x.close()
    f_y.close()
    return count

def config_Paths(dataname=''):
    if not os.path.exists(common_params.xypath):
        os.makedirs(common_params.xypath)
    # if dataname=='OJ': # OJ database
    #     # OJ database
    #     # common_params.jsondir = 'C:/Users/anhpv/Desktop/data/'
    #     # common_params.xypath = common_params.jsondir + 'xy/'
    #     if not os.path.exists(common_params.xypath):
    #         os.makedirs(common_params.xypath)
    # if dataname =='MNMX': # MNMX database
    #     # common_params.jsondir = 'D:/CodeChef/clone/'
    #     # common_params.xypath = common_params.jsondir + 'xy/'
    #     if not os.path.exists(common_params.xypath):
    #         os.makedirs(common_params.xypath)

if __name__ == "__main__":
    dataname = common_params.dataname
    config_Paths(dataname)

    jsondir =  common_params.jsondir
    netstructure=NetStruct.netstructure #'tbcnn'
    numInst ={}
    xname ={}
    yname ={}
    for fold in ['train','CV','test']:
        jsonAST = jsondir+ dataname + common_params.jsonfold[fold]
        xfile = dataname+ '_'+ netstructure+ '_X'+ fold
        xname[fold] = xfile

        yfile = dataname +'_'+ netstructure+ '_Y'+ fold+'.txt'
        yname[fold] = yfile

        numInst[fold] = writeXY(jsonAST, common_params.xypath + xfile, common_params.xypath + yfile)

    commonFunctions.generateSettingContent(common_params.xypath+'../settings_'+dataname+'.txt',
        {'numtrain': numInst['train'], 'numcv': numInst['CV'], 'numtest': numInst['test'], 'output': gl.numOut,
         'paramFile': NetStruct.paramsFile, 'xtrain': xname['train'], 'xcv': xname['CV'],
         'xtest': xname['test'],
         'ytrain': yname['train'], 'ycv': yname['CV'], 'ytest': yname['test'],'database': dataname})

    # config_Paths()
    #
    # jsondir =  common_params.jsondir
    # netstructure=NetStruct.netstructure #'tbcnn'
    # numInst ={}
    # xname ={}
    # yname ={}
    # for fold in range(1,6):
    #     dataname = 'Fold'+str(fold)
    #     for dts in ['train','CV','test']:
    #         jsonAST = '{0}Fold{1}_ast_{2}.txt'.format(jsondir,fold, dts)
    #         xfile = dataname+ '_'+ netstructure+ '_X'+ dts
    #         xname[dts] = xfile
    #
    #         yfile = dataname +'_'+ netstructure+ '_Y'+ dts+'.txt'
    #         yname[dts] = yfile
    #
    #         numInst[dts] = writeXY(jsonAST, common_params.xypath + xfile, common_params.xypath + yfile)
    #
    #     commonFunctions.generateSettingContent(common_params.xypath+'../settings_'+dataname+'.txt',
    #         {'numtrain': numInst['train'], 'numcv': numInst['CV'], 'numtest': numInst['test'], 'output': gl.numOut,
    #          'paramFile': NetStruct.paramsFile, 'xtrain': xname['train'], 'xcv': xname['CV'],
    #          'xtest': xname['test'],
    #          'ytrain': yname['train'], 'ycv': yname['CV'], 'ytest': yname['test'],'database': dataname})

    print 'Done!!'