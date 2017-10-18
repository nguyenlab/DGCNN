import sys
sys.path.append('../03_ConstructCandW')
import GraphData_IO
# import main_GCNN_Pooling as GCNN
import main_MultiChannelGCNN as GCNN
import gcnn_params as params
datapath = params.datapath
word_dict, vectors, numFea = GraphData_IO.LoadVocab(vocabfile=datapath + params.tokvecFile)
toktypeDict = GraphData_IO.LoadTokenTypeDict(filename=datapath + params.toktype)
def testNet(filename):
    graph = GraphData_IO.getGraphFromTextFile(filename=filename)
    graph.show()
    print 'num of vertexes ', len(graph.getVertexes())
    print 'num of edges', len(graph.getEdges())
    layers = GCNN.InitByNodes(graph = graph , word_dict = word_dict, toktypeDict=toktypeDict)

    print 'Totally:', len(layers), 'layer(s)'
    numcon =0
    for l in layers:
        numcon += len(l.connectDown)
        if hasattr(l, 'bidx') and l.bidx is not None:
            print l.name, '\tBidx', l.bidx[0], '\tlen biases=', len(l.bidx)
        else:
            print l.name, 'numUnit', l.numUnit


        print "    Down:"
        for c in l.connectDown:
            if hasattr(c, 'Widx'):
                print "        ", c.xlayer.name, " -> ", '|', \
                    '(xnum= ', c.xnum, ', ynum= ', c.ynum, '), Wid = ', c.Widx[0], '(Woef=', c.Wcoef,')'
            else:
                print "        ", c.xlayer.name, " -> ", '|', \
                    '(xnum= ', c.xnum, ', ynum= ', c.ynum, ')'
    print 'Connections:', numcon
if __name__ == "__main__":
    # print GCNN.Wconv_root[0][1]
    # print GCNN.Wconv_neighbor
    # print GCNN.Bconv
    filename =datapath + 'test.dot'
    testNet(filename=filename)
    # arr =[[[1,2,3,4],[2,3,2]],[2,3],[5,6]]
    # print arr[0][0]
    # graph = GraphData_IO.getGraphFromTextFile(filename=filename)
    # graph.show()