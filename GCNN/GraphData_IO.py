import json
import random
import struct
from operator import itemgetter

import numpy as np
import os

import re

from Graph import Graph, GVertex

def LoadTokenTypeDict(filename =''):
    file = open(filename, "r")
    idx =0
    dict={}
    file.readline() # ignore header
    for line in file:
        items = line.rstrip().split()
        if len(items)>=2:
            dict[items[0]]= items[1]
    return dict

def LoadVocab(vocabfile =''):
    print 'Load vocab from:', vocabfile
    file = open(vocabfile, "r")
    idx =0
    vectors=[]
    dict={}
    vecsize =0
    for line in file:
        items = line.rstrip().split()
        if len(items)<=2:
            vecsize = int(items[1])
            continue
        if len(items[1:]) != vecsize:
            word=''
            vectors.append(items)
        else:
            word= items[0]
            vectors.append(items[1:])
        dict[word] = idx
        idx = idx +1

    vectors = np.reshape(vectors,-1)
    # convert to float
    # print vectors[0]
    vectors = [float(i) for i in vectors]
    return dict, vectors, vecsize
def getGraphFromTextFile(filename =''):
    reader = open(filename,'r')
    # print filename
    #ignore 3 first rows
    reader.readline()
    reader.readline()
    reader.readline()
    #
    nodename_dict={}
    nodeid =0
    g = Graph()
    for line in reader:
        line = line.strip()
        if line =='}':
            break
        if '[' not in line:
            break
        idx = line.index('[')

        edgeinfor = line[:idx]
        edge = edgeinfor.split(' -> ')

        if len(edge) ==1: # vertex
            vertex = GVertex.getVertexFromVertexInfor(line)
            vertex.id = nodeid
            # add to dictionary
            nodename_dict[vertex.name] = vertex
            #add to graph
            g.addVetex(vertex)

            nodeid +=1
        else:
            name1 = edge[0].strip()
            name2 = edge[1].strip()
            # add to graph
            if name1 not in nodename_dict.keys():
                vertex = GVertex.getVertexFromEdge(name1)
                vertex.id = nodeid
                # add to dictionary
                nodename_dict[vertex.name] = vertex
                # add to graph
                g.addVetex(vertex)
                nodeid+=1
                # print 'Not found vertex: ', name1
            if name2 not in nodename_dict.keys():
                vertex = GVertex.getVertexFromEdge(name2)
                vertex.id = nodeid
                # add to dictionary
                nodename_dict[vertex.name] = vertex
                # add to graph
                g.addVetex(vertex)
                nodeid += 1
                # print 'Not found vertex: ', name2
            g.addEdge(nodename_dict[name1], nodename_dict[name2])

    return g
def dataStatistic(datadir=''):

    vertex_cout =[]
    edge_count =[]
    tokens ={}
    for subdir in os.listdir(datadir):
        if not subdir.endswith('/'):
            subdir = subdir + '/'

        if os.path.isfile(datadir + subdir[:len(subdir) - 1]):
            continue


        for onefile in os.listdir(datadir + subdir):
            filename = onefile
            onefile = datadir + subdir + onefile

            if filename.endswith(".dot"):
                g = getGraphFromTextFile(onefile)
                vertexes = g.getVertexes().values()
                for v in vertexes:
                    tokens[v.token] = v.token
                vertex_cout.append(len(g.getVertexes()))
                edge_count.append(len(g.getEdges()))
    # write statistics
    out = open(datadir+'/statistic.csv','w')
    out.write('vertex, edge\n');
    for idx in xrange(len(vertex_cout)):
        out.write(str(vertex_cout[idx])+','+str(edge_count[idx])+'\n')
    out.close()
    #write tokens
    out = open(datadir + '/token', 'w')
    for tok in tokens:
        out.write(tok+'\n');
    out.close()
def writeGraph2Json(datadir, out):
    jsonObjs =[]
    for onefile in os.listdir(datadir):
        if onefile.endswith(".dot"):
            g = getGraphFromTextFile(datadir + onefile)
            jsonObjs.append(g.dump())

     #write to file
    with open(out, 'w') as outfile:
        json.dump(jsonObjs, outfile)
def readGraphFromJson(jsonFile=''): # 1: non-virus, 2: virus
    graphs=[]
    with open(jsonFile, 'r') as f:
        jsonObjs = json.load(f)
        for obj in jsonObjs:
            g = Graph.load(obj)
            graphs.append(g)
    return graphs

# def createTokenVecs(datafiles='', out='', vecsize=30, statistic_out =''):
#     tokdict ={}
#     opdict ={}
#     g_infor =[]
#     for file in datafiles:
#         with open(file, 'r') as f:
#             jsonObjs = json.load(f)
#             for obj in jsonObjs:
#                 g = Graph.load(obj)
#                 g_infor.append([len(g.Vs), len(g.Es)])
#                 for v in g.Vs.values():
#                     tokdict[v.token] = v.toktype
#                     for idx in range(1, len(v.content)):
#                         opdict[v.content[idx]] =v.content[idx]
#     # write token to file
#     tokens =[]
#     for tok in tokdict:
#         tokens.append((tok, tokdict[tok]))
#     # sort by type: ASM or API
#     tokens=sorted(tokens, key=itemgetter(1))
#
#     f = open(out, 'w')
#     f.write(str(len(tokens))+' '+ str(vecsize)+'\n')
#     for tok, toktype in tokens:
#         f.write(tok + ' ')
#         vec = [random.uniform(-1, 1) for v in xrange(vecsize)]
#         vec = [str(i) for i in vec]
#         f.write(' '.join(vec))
#         f.write('\n')
#     f.close()
#
#     if statistic_out =='':
#         return
#     f = open(statistic_out,'w')
#     f.write('Vertex, Edges\n')
#     for nv, ne in g_infor:
#         f.write(str(nv)+','+str(ne)+'\n')
#     f.close()
def searchContentInFile(datadir='', searchValue=''):
    for onefile in os.listdir(datadir):
        if onefile.endswith('.dot'):
            with open(datadir+onefile,'r') as f:
                content = f.read()
                if content.find(searchValue) != -1:
                    print datadir+onefile
                    #print content
                    #break
def getGraphsFromDataDir(datadir, classlabel =0):
    # read graph from .dot files and return graphs according Json format
    jsonObjs =[]
    count = 1
    for onefile in os.listdir(datadir):
        if onefile.endswith(".dot"):
            g = getGraphFromTextFile(datadir + onefile)
            if classlabel>=0:
                g.label = classlabel
            jsonObjs.append(g.dump())

            count +=1
    return jsonObjs
def splitJson(jsonFile='', fold = 2):
    with open(jsonFile, 'r') as f:
        jsonObjs = json.load(f)

    numObj = len(jsonObjs)
    print 'Total =', numObj
    N = numObj/fold
    for idx in range(1, fold+1):
        begin =(idx-1)* N
        end = idx *N
        if idx == fold:
            end = numObj
        foldObjs = jsonObjs[begin: end]
        print 'Fold ', idx,' = ', len(foldObjs)

        outfile = jsonFile
        outfile = outfile.replace('.json','_'+str(idx)+'.json')
        with open(outfile,'w') as f:
            json.dump(foldObjs, f)
def jsonGraphStatistics(jsonFiles, out):
    f = open(out,'w')
    f.write('Vertex, edges, max degree(In+Out)\n')
    instanceCount ={}
    for onefile in jsonFiles:
        graphs = readGraphFromJson(jsonFile=onefile)
        for g in graphs:
            f.write(str(len(g.Vs))+','+ str(len(g.Es))+','+ str(g.maxDegree())+'\n')
            if g.label not in instanceCount:
                instanceCount[g.label] =0
            instanceCount[g.label] += 1
    for l in instanceCount:
        f.write('#inst of label '+ str(l)+': '+ str(instanceCount[l])+'\n')
    f.close()
def getListofTokens(jsonFiles=[], out='',withOps=True):
    tokDict = {}
    for onefile in jsonFiles:
        graphs = readGraphFromJson(jsonFile= onefile)
        for g in graphs:
            Vs= g.getVertexes()
            for (vid, v) in Vs.items():
                if withOps:
                    tok = ','.join(v.content)
                else:
                    tok = v.token
                tokDict[tok] = v.toktype
    # sort by type
    if not withOps:
        tokDict['val'] = 'CFG'
        tokDict['reg'] = 'CFG'
        tokDict['api'] ='CFG'
        tokDict['asm'] ='CFG'
    tokDict = [(tok, toktype) for tok, toktype in tokDict.items()]
    tokDict.sort(key=lambda x: x[1])
    f = open(out, 'w')
    for tok, toktype in tokDict:
        f.write(tok+'\n')
    f.close()
def modifyGraphVertices(jsonFiles, out):
    # remove .dl, .dll
    # remove _ at begining
    pattern = '(^_*)|(.|_)(dll|dl)$'

    vtok ={}
    contentTok ={}
    for onefile in jsonFiles:
        jsonObjs =[]
        graphs = readGraphFromJson(jsonFile=onefile)
        for g in graphs:
            for (vidx, v) in g.Vs.items():
                v.token = re.sub(pattern, '', v.token)
                idx = v.token.find('_')
                if idx>0:
                   v.token = v.token[:idx]

                vtok[v.token] = v.token
                # v.token = v.token.strip('_')
                for cid in range(0, len(v.content)):
                    token = v.content[cid]
                    token = re.sub(pattern, '', token)
                    idx = token.find('_')
                    if idx>0:
                        v.content[cid] = token[:idx]

                    contentTok[v.content[cid]] = v.content[cid]
                    # v.content[cid] = v.content[cid].strip('_')
            jsonObjs.append(g.dump())
        # write graphs to file
        f = open(onefile,'w')
        json.dump(jsonObjs, f)
        f.close()
    f = open(out, 'w')
    f.write('Vertex tokens:\n')
    for tok in vtok:
        f.write(tok + '\n')
    f.write('Content tokens:\n')
    for tok in contentTok:
        f.write(tok+'\n')
    f.close()

if __name__ == "__main__":
    #statistic CodeChef data
    datapath = '/home/s1520015/Experiment/CodeChef/OriginalTrees/'
    problem ='SUMTRIAN'
    jsonGraphStatistics([datapath+ problem+ '_AstGraph.json'], datapath+'../'+ problem+'graph_statistic' )
    problem ='MNMX'
    jsonGraphStatistics([datapath+ problem+ '_AstGraph.json'], datapath+'../'+ problem+'graph_statistic' )
    problem ='FLOW016'
    jsonGraphStatistics([datapath+ problem+ '_AstGraph.json'], datapath+'../'+ problem+'graph_statistic' )
    problem ='SUBINC'
    jsonGraphStatistics([datapath+ problem+ '_AstGraph.json'], datapath+'../'+ problem+'graph_statistic' )

    # splitJson('Z:/Experiment/CodeChef/OriginalTrees/SUMTRIAN_train_AstGraph.json',2)
    # path ='C:/Users/anhpv/Desktop/CFG/Experiment/5Folds/Fold1/Virus/'
    # fold ='Training/'
    # name= 'GCNN'
    # datapath = path+fold

    # datapath='C:/Users/anhpv/Desktop/CFG/DataForCheckImp/'
    # writeGraph2Json(datapath+name+'/', datapath+name+'.json')

    # write token vectors
    # # print 'Create Token Vecs'
    datapath = '/home/s1520015/Experiment/5Folds/'
    datafiles=[]
    for idx in range(1,6):
        datafiles.append(datapath+'/Fold'+str(idx)+'/dataFold'+str(idx))

    jsonGraphStatistics(datafiles, datapath + '/graph_statistic')
    # # createTokenVecs(datafiles= datafiles,out=datapath+'instruction_vec.txt', vecsize=30, statistic_out=datapath+'statistics.txt')
    # # modifyGraphVertices(datafiles, datapath+'listoftokens.txt')
    # getListofTokens(datafiles, datapath+'dict_instruction_no_ops.txt',withOps= False)
    # getListofTokens(datafiles, datapath+'dict_instruction_ops.txt',withOps= True)

    # list all tokens
    # datafiles=[]
    # for idx in range(1,6):
    #     datafiles.append('/Fold'+str(idx)+'/dataFold'+str(idx))
    # getListofTokens(jsonFiles=datafiles,out='instruction_ops.txt', ignoreOps=False)

    # check content
    # searchContentInFile(datapath+'Fold2/NonVirus/','KeInitializeSpinLoc') #KeInitializeSpinLoc, KeNumberProcessors


    # path ='/home/s1520015/Experiment/5Folds/Fold1/Virus/'
    #dataStatistic(path)
    #print 'Done'
    # g = getGraphFromTextFile(path+'5a0efeb3fad0c865c649820ec3ff27c8254a1aab8be226915f87e539f3475f67_test_model.dot')
    # jsonObj = g.dump()
    # with open(path+'../test.json','w') as f:
    #     f.write('token content\n')
    #     for (id,v) in g.Vs.items():
    #         f.write(v.token+' '+','.join(v.content))
    #         f.write('\n')
    #
    # print 'Graph 2'
    # g = getGraph(path+'test.dot')
    # g.show()
    # json_object = g.dump()
    # with open(path+'test.json', 'w') as outfile:
        #json.dumps([o.dump() for o in my_list_of_ipport])
        # json.dump([json_object], outfile)
    # print json_object
    # with open(path + 'test.json', 'r') as outfile:
    #     json_object = json.load(outfile)
    #     g= Graph.load(json_object[0])
    #     g.show()
    # path = 'C:/Users/anhpv/Desktop/CFG/'
    # g = getGraph(path + 'check1.dot')

    # print 'Create Tokeen Vecs'
    # datapath = '/home/s1520015/Experiment/ASMCFG/SourceCode/5Folds/CFG/'
    # datafiles = []
    # datafiles.extend([datapath+'Fold1_cfg_train.txt',datapath+'Fold1_cfg_CV.txt',datapath+'Fold1_cfg_test.txt'])
    # # createTokenVecs(datafiles= datafiles,out=datapath+'instruction_vec.txt', vecsize=30, statistic_out=datapath+'statistics.txt')
    # # modifyGraphVertices(datafiles, datapath+'listoftokens.txt')
    # getListofTokens(datafiles, datapath + 'dict_instruction_no_ops.txt', withOps=False)
    # getListofTokens(datafiles, datapath + 'dict_instruction_ops.txt', withOps=True)