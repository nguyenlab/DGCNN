import sys
import numpy as np
from Graph import Graph, GVertex

sys.path.append('../nn')
sys.path.append('../')
import Layers as Lay, Connections as Con, Activation

import gcnn_params
##############################
# hyperparam
toktypeDict = gcnn_params.toktypeDict
class Vinfo:
    def __init__(self, id, data = None,income=None, outgo =None):
        if income is None:
            self.income=[]
        else:
            self.income = income
        if outgo is None:
            self.outgo = []
        else:
            self.outgo = outgo
        if data is None:
            self.data=[]
        else:
            self.data =data
        self.id = id
    def inDegree(self):
        return len(self.income)
    def outDegree(self):
        return len(self.outgo)
    def show(self, buf = sys.stdout):
        print 'id = ',str(self.id)
        print  'data=',self.data
        print  'incoming', self.income
        print  str(self.numIn())
        print 'outgoing', self.outgo
        print str(self.numOut()),'\n\n'

def MergeNodes(graph):
    # sort vertexes by incoming degree
    vertexes = graph.getVertexes() # dict of nodes: id - node
    numVertexes = len(vertexes)

    edges = graph.getEdges() # list of edges[(id1--->id2)]
    numEdges = len(edges)

    vertexes_info ={}
    # get information of vertexes
    for idx in range(numVertexes):
        v= vertexes[idx]
        vinfor =Vinfo(id=v.id)
        for v1, v2 in edges: # outgoing edge
            if idx== v1:
                vinfor.outgo.append(v2)
            if idx == v2: # incoming edge
                vinfor.income.append(v1)
        vertexes_info[idx] = vinfor
    # sum degree of incoming nodes and outgoing nodes
    minid =0
    minDegree =0
    for idx in vertexes_info:
        vinfor = vertexes_info[idx]
        if vinfor.inDegree()<minid:
            minDegree = vinfor.inDegree()
            minid = idx

    # start from node minid
    newgraph = Graph()
    stack = []
    stack.append(minid)
    ver_list =[i for i in range(0, numVertexes)]
    newid =0
    vid_newV ={}

    while len(ver_list)>0 and len(stack)>0:
        curID = stack.pop()
        if curID not in ver_list:
            continue
        ver_list.remove(curID)

        # Create new node gather all neighbors of the current node
        new_V = GVertex(id=newid, name='', token='', toktype='', content='')
        vid_newV[curID] = new_V
        newid += 1

        name =str(curID)+'_'
        neighbors =[]
        neighbors.extend(vertexes_info[curID].income)
        neighbors.extend(vertexes_info[curID].outgo)
        for v in neighbors:
            if v not in ver_list:
                continue
            name +=str(v)+'_'
            ver_list.remove(v)
            # push neighbor nodes to consider in next step
            stack.extend(vertexes_info[v].income)
            stack.extend(vertexes_info[v].outgo)
            stack = [i for i in np.unique(stack)]
            #print stack
            vid_newV[v] = new_V

        new_V.name = name
        new_V.token= name
        newgraph.addVetex(new_V)
    # add edges for new graph
    edgelist={}
    for v1, v2 in edges:
        newv1 = vid_newV[v1]
        newv2 =  vid_newV[v2]
        edgestr = str(newv1.id)+'_'+ str(newv2.id)
        if edgestr in edgelist or newv1==newv2:
            continue
        edgelist[edgestr] =''
        newgraph.addEdge(newv1, newv2)

    return newgraph, vid_newV
def ContructConv_Pooling(graph,word_dict , inputLayers, layers,
                         numInput, numCon, poolAll,
                         Wconv_root, Wconv_in, Wconv_out, Bconv
                         ):
    vertexes = graph.getVertexes()  # dict of nodes: id - node
    numVertexes = len(vertexes)

    edges = graph.getEdges()  # list of edges[(id1--->id2)]

    vertexes_info = {}
    # get information of vertexes
    for idx in range(numVertexes):
        v = vertexes[idx]
        vinfor = Vinfo(id=v.id, data=v.getData(toktypeDict))
        for v1, v2 in edges:  # outgoing edge
            if idx == v1:
                vinfor.outgo.append(v2)
            if idx == v2:  # incoming edge
                vinfor.income.append(v1)
        vertexes_info[idx] = vinfor
    # sum degree of incoming nodes and outgoing nodes
    n_degrees = {}
    for idx in vertexes_info:
        d_innode = 0
        d_outnode = 0
        vinfor = vertexes_info[idx]
        for nid in vinfor.outgo:
            neighbor = vertexes_info[nid]
            d_outnode += neighbor.inDegree() + neighbor.outDegree()

        for nid in vinfor.income:
            neighbor = vertexes_info[nid]
            d_innode += neighbor.inDegree() + neighbor.outDegree()

        n_degrees[vinfor.id] = (d_innode, d_outnode)


    if inputLayers is None: # first layers of embedding

        inputLayers= [None] * numVertexes

        # construct the embedding layer for each vertex
        for idx in xrange(numVertexes):
            # get vertex
            vinfor = vertexes_info[idx]
            token = vinfor.data[0]  # get the token of the current view

            if token in word_dict:
                bidx = word_dict[token] * numInput
            else:
                bidx = 0

            inputLayers[idx] = Lay.layer('vec_' + str(idx) + '_' + token, \
                                          range(bidx, bidx + numInput), \
                                         numInput
                                          )
            inputLayers[idx].act = 'embedding'

        # add embedding layers to Layers
        for idx in range(0, numVertexes):
            layers.append(inputLayers[idx])
    convLayers = [None]*numVertexes
    for idx in xrange(numVertexes):
        # current vertex V[idx]
        vinfor = vertexes_info[idx]
        token = vertexes[idx].token  # get the token

        conLayer = Lay.layer('Convolve_' + token, Bconv, numCon)
        conLayer.act = 'convolution'
        convLayers[idx] = conLayer
        layers.append(conLayer)

        rootCon = Con.connection(inputLayers[idx], conLayer, numInput, numCon, Wconv_root)
        # add connections
        dsum_innodes, dsum_outnodes = n_degrees[idx]
        for n in vinfor.outgo:
            # print token, vertexes_info[n].inDegree(), ',', vertexes_info[n].outDegree(), ',', n_sumDegree
            Wcoef = 1.0  # * (vertexes_info[n].inDegree() + vertexes_info[n].outDegree()) / dsum_outnodes
            if Wcoef != 0:
                neighborCon = Con.connection(inputLayers[n], conLayer, numInput, numCon, Wconv_out,
                                             Wcoef=Wcoef)

        for n in vinfor.income:
            # print token, vertexes_info[n].inDegree(), ',', vertexes_info[n].outDegree(), ',', n_sumDegree
            Wcoef = 1.0  # * (vertexes_info[n].inDegree() + vertexes_info[n].outDegree()) / dsum_innodes
            if Wcoef != 0:
                neighborCon = Con.connection(inputLayers[n], conLayer, numInput, numCon, Wconv_in,
                                                 Wcoef=Wcoef)
    # Pooling layers
    newgraph, oldId_newV = MergeNodes(graph)
    if poolAll == False:
        numPool = len(newgraph.getVertexes())
        pool = [None] * numPool
        for idx in range(0, numPool):
            pool[idx] =  Lay.PoolLayer('pooling', numCon)
        for idx in range(0, numVertexes):
            # create pooling connection
            poolCon = Con.PoolConnection(convLayers[idx], pool[oldId_newV[idx].id])
    else:
        pool = [None]
        pool[0] =  Lay.PoolLayer('pooling', numCon)
        for idx in range(0, numVertexes):
            # create pooling connection
            poolCon = Con.PoolConnection(convLayers[idx], pool[0])
    layers.extend(pool)
    return newgraph, pool

def ConstructGraphConvolution(graph,word_dict, numFea, numCon, numDis, numOut, \
                             Wconv_root, Wconv_in, Wconv_out, Bconv, \
                             Wdis, Woutput, Bdis, Boutput
                             ):
    # def ContructConv_Pooling(graph, word_dict, inputLayers, layers,
    #                          numInput, numCon,
    #                          Wconv_root, Wconv_in, Wconv_out, Bconv
    #                          ):
    inputLayers = None
    layers =[]
    numInput = numFea
    pool = None
    poolAll = False
    for c in range(len(numCon)):
        if c ==len(numCon)-1:
            poolAll = True
        graph, pool = ContructConv_Pooling(graph, word_dict, inputLayers, layers,
                             numInput, numCon[c], poolAll,
                             Wconv_root[c], Wconv_in[c], Wconv_out[c], Bconv[c]
                             )
        numInput = numCon[c]
        inputLayers = pool

    # discriminative layer
    discriminative = Lay.layer('discriminative', Bdis, numDis)
    discriminative.act = 'hidden'
    # pool ---> discriminative
    con = Con.connection(pool[0], discriminative, numCon[-1], numDis, Wdis)

    output = Lay.layer('outputlayer', Boutput, numOut)
    output.act = 'softmax'
    # discriminative ---> output
    outcon = Con.connection(discriminative, output, numDis, numOut, Woutput)

    if numOut > 1:
        output._activate = Activation.softmax
        output._activatePrime = None
    layers.append(discriminative)
    layers.append(output)
    # add successive connections
    numlayers = len(layers)
    for idx in xrange(numlayers):
        if idx > 0:
            layers[idx].successiveLower = layers[idx - 1]
        if idx < numlayers - 1:
            layers[idx].successiveUpper = layers[idx + 1]
    return layers
