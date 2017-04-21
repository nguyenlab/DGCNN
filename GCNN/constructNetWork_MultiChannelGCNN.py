import sys

sys.path.append('../nn')
sys.path.append('../')
import Layers as Lay, Connections as Con, Activation

import gcnn_params
##############################
# hyperparam
# toktypeDict = gcnn_params.toktypeDict
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


def ConstructGraphConvolution(graph,word_dict,toktypeDict,numView, numFea, numCon, numDis, numOut, \
                             Wconv_root, Wconv_in, Wconv_out, Bconv, \
                             Wdis, Woutput, Bdis, Boutput
                             ):
    #word_dict:
    #numView, numFea, numCon, numDis, numOut, \
    #Wconv_root    [[conv1_view1, conv1_view2, ...], conv2, ...]]
    #Wconv_neighbor  [[conv1_view1, conv1_view2, ...], conv2, ...]]
    #Bconv        [conv1, conv2, ...]
    #Wdis[pool_view1, pool_view2, ...---> Dis]
    vertexes = graph.getVertexes() # dict of nodes: id - node
    numVertexes = len(vertexes)

    edges = graph.getEdges() # list of edges[(id1--->id2)]
    numEdges = len(edges)

    vertexes_info ={}
    # get information of vertexes
    for idx in range(numVertexes):
        v= vertexes[idx]
        vinfor =Vinfo(id=v.id, data=v.getData(toktypeDict, withOps = gcnn_params.withOps))
        for v1, v2 in edges: # outgoing edge
            if idx== v1:
                vinfor.outgo.append(v2)
            if idx == v2: # incoming edge
                vinfor.income.append(v1)
        vertexes_info[idx] = vinfor
    # sum degree of incoming nodes and outgoing nodes
    n_degrees ={}
    for idx in vertexes_info:
        d_innode =0
        d_outnode =0
        vinfor = vertexes_info[idx]
        for nid in vinfor.outgo:
            neighbor = vertexes_info[nid]
            d_outnode += neighbor.inDegree() + neighbor.outDegree()

        for nid in vinfor.income:
            neighbor = vertexes_info[nid]
            d_innode += neighbor.inDegree() + neighbor.outDegree()

        n_degrees[vinfor.id] =(d_innode, d_outnode)

    layers=[]

    emb_layers = numView *[None] # embeddinglayers
    for v in range(0, numView):
            emb_layers[v] = [None]* numVertexes

    for view in range(0, numView):
        # emb_layers[view] = [None] * numVertexes
        # construct the embedding layer for each vertex
        for idx in xrange(numVertexes):
            # get vertex
            vinfor = vertexes_info[idx]
            token = vinfor.data[view] # get the token of the current view

            if token in word_dict:
                bidx = word_dict[token] * numFea
            else:
                # print token
                bidx =0
            emb_layers[view][idx] = Lay.layer('vec_' + str(idx) + '_' + token, \
                                    range(bidx, bidx + numFea), \
                                    numFea
                                    )
            emb_layers[view][idx].act = 'embedding'
                # connect from previous ---> current
    # add embedding and mapping layers to Layers
    for v in range(0, numView):
        for idx in range(0, numVertexes):
            layers.append(emb_layers[v][idx])

    num_Pre = numFea # the size of previous layers

    pre_layer = emb_layers

    for c in xrange(len(numCon)): # convolutional layers
        current_layer ={} # current layer
        for idx in xrange (numVertexes):
            # current vertex V[idx]
            vinfor = vertexes_info[idx]
            token = vinfor.data[0]  # get the token

            conLayer = Lay.layer('Convolve'+str(c)+'_' + token+'_V'+str(view+1), Bconv[c], numCon[c])
            conLayer.act = 'convolution'
            layers.append(conLayer)
            current_layer[idx] = conLayer

            # add connections
            if c==0: # all views of embedding ---> convolution 1
                for v in range(0, numView):
                    rootCon = Con.connection(pre_layer[v][idx], conLayer, num_Pre, numCon[c], Wconv_root[c][v])
                    # add connections
                    dsum_innodes, dsum_outnodes = n_degrees[idx]

                    for n in vinfor.outgo: # outgoing nodes
                        # print token, vertexes_info[n].inDegree(), ',', vertexes_info[n].outDegree(), ',', n_sumDegree
                        Wcoef = 1.0 #* (vertexes_info[n].inDegree()+vertexes_info[n].outDegree())/ dsum_outnodes
                        if Wcoef !=0:
                            neighborCon = Con.connection(pre_layer[v][n], conLayer, num_Pre, numCon[c], Wconv_out[c][v],Wcoef = Wcoef)
                    for n in vinfor.income: # outgoing nodes
                        # print token, vertexes_info[n].inDegree(), ',', vertexes_info[n].outDegree(), ',', n_sumDegree
                        Wcoef = 1.0 #* (vertexes_info[n].inDegree()+vertexes_info[n].outDegree())/ dsum_innodes
                        if Wcoef !=0:
                            neighborCon = Con.connection(pre_layer[v][n], conLayer, num_Pre, numCon[c], Wconv_in[c][v],Wcoef = Wcoef)
            else:
                rootCon = Con.connection(pre_layer[idx], conLayer, num_Pre, numCon[c], Wconv_root[c])
                # add connections
                dsum_innodes, dsum_outnodes = n_degrees[idx]
                for n in vinfor.outgo:
                    # print token, vertexes_info[n].inDegree(), ',', vertexes_info[n].outDegree(), ',', n_sumDegree
                    Wcoef = 1.0 #* (vertexes_info[n].inDegree() + vertexes_info[n].outDegree()) / dsum_outnodes
                    if Wcoef != 0:
                        neighborCon = Con.connection(pre_layer[n], conLayer, num_Pre, numCon[c], Wconv_out[c], Wcoef=Wcoef)

                for n in vinfor.income:
                    # print token, vertexes_info[n].inDegree(), ',', vertexes_info[n].outDegree(), ',', n_sumDegree
                    Wcoef = 1.0 #* (vertexes_info[n].inDegree() + vertexes_info[n].outDegree()) / dsum_innodes
                    if Wcoef != 0:
                        neighborCon = Con.connection(pre_layer[n], conLayer, num_Pre, numCon[c], Wconv_in[c],
                                                     Wcoef=Wcoef)

        num_Pre = numCon[c]
        pre_layer = current_layer

    pool = Lay.PoolLayer('pooling', numCon[-1])
    # connect from convolution ---> pooling
    for key in pre_layer:
        poolCon = Con.PoolConnection(pre_layer[key], pool)
    # add view layers to layers
    layers.append(pool)

    # discriminative layer
    discriminative = Lay.layer('discriminative', Bdis, numDis)
    discriminative.act = 'hidden'
    # pool ---> discriminative
    con = Con.connection(pool, discriminative, numCon[-1], numDis, Wdis)

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
