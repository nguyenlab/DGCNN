import numpy as np
import GraphData_IO
from Graph import Graph, GVertex
from constructNetwork_GCNN_Pooling import Vinfo

filename = 'C:/Users/anhpv/Desktop/CFG/test.dot'
graph = GraphData_IO.getGraphFromTextFile(filename=filename)
graph.show()
def MergeNode(graph):
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

    return newgraph
if __name__ =='__main__':
    newgraph = MergeNode(graph)
    newgraph.show()