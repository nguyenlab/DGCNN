import sys
sys.path.append('../nn')
sys.path.append('../')
import Layers as Lay, Connections as Con, Activation

##############################
# hyperparam


        
def ConstructTreeConvolution(nodes, numFea, numOut,\
                        Wleft, Wright, Bconstruct,\
                        Woutput, Boutput,\
        ):
    
    
    numNodes = len(nodes)
    
    layers = []

    numLeaf = 0
    for idx in xrange(numNodes):
        node = nodes[idx]
        if len(node.children) == 0:
            numLeaf += 1
            tmplayer = Lay.layer('vec_'+str(idx)+'_' + node.word,\
                          node.bidx,\
                          numFea
                      )
            tmplayer.act = 'embedding'
            layers.append(tmplayer)      
    
    


      
            
            
# auto encoding
    # layers = |---leaf---|---non_leaf(autoencoded)---| (numNodes)
    numNonLeaf = numNodes - numLeaf

    layers.extend([None] * ( numNonLeaf))

    for idx in xrange(numLeaf, numNodes): 
        node = nodes[idx]
        #layers[ idx + numNonLeaf ] = layers [idx]
      
        tmplayer = Lay.layer('ae_'+str(idx)+'_'+node.word,\
                    Bconstruct[0], numFea)
        tmplayer.act = 'autoencoding'
        layers[idx] = tmplayer


    # add reconstruction connections
    for idx in xrange(0, numNodes):
        node = nodes[idx]
        if node.parent == None:
            continue
        tmplayer = layers[idx]
        parent = layers[ node.parent ]
        if node.leftRate != 0:
            Con.connection(tmplayer, parent,\
                             numFea, numFea, Wleft[0], Wcoef = node.leftRate * node.leafNum/nodes[node.parent].leafNum)
	          
        if node.rightRate != 0:
            Con.connection(tmplayer, parent,\
                             numFea, numFea, Wright[0], Wcoef = node.rightRate * node.leafNum/nodes[node.parent].leafNum)

    
    output = Lay.layer('outputlayer', Boutput[0], numOut)
    Con.connection(layers[-1], output, numFea, numOut, Woutput[0])
    if numOut > 1:
        output._activate = Activation.softmax
        output._activatePrime = None
        output.act = 'softmax'
    else:
        output._activate = Activation.dummySigmoid
        output._activatePrime = Activation.dummySigmoidPrime
    #layers.append(discriminative)
    layers.append(output)
    
# add successive connections
    numlayers = len(layers)
    for idx in xrange( numlayers ):
        if idx > 0:
            layers[idx].successiveLower = layers[idx-1]
        if idx < numlayers - 1:
            layers[idx].successiveUpper = layers[idx+1]
    return layers
