import sys

sys.path.append('../nn')
sys.path.append('../')
import Layers as Lay, Connections as Con, Activation


##############################
# hyperparam
class info:
    parent = None
    childrenList = None

    def __init__(self, parent=None):
        self.parent = parent
        self.childrenList = []


def ConstructTreeConvolution(nodes, numFea, numCon,numRecur, numDis, numOut, \
                             Wleft, Wright, Bconstruct, \
                             Wcomb_ae, Wcomb_orig, \
                             Wconv_root, Wconv_left, Wconv_right, Bconv, \
                             Wrecur_root, Wrecur_left, Wrecur_right, Brecur, \
                             Wconv_dis,Wrecur_dis , Woutput, Bdis, Boutput, \
                             poolCutoff
                             ):
    # nodes
    # numFea: # of the word/symbol feature size
    # numCon: # of the convolution size
    # Wleft:  left  weights of continous binary tree autoencoder
    # Wright: right weights of continous binary tree autoencoder
    # Bconstruct: the biase for the autoencoder
    # Wcomb_ae, Wcomb_orig: the weights for the combination of
    #                       autoencoder and the original vector
    #                       (no biase for this sate)
    # Wconv_root, Wconv_left, Wconv_right, Bconv: the weights for covolution
    # Bconv: Biases for covolution

    numNodes = len(nodes)

    layers = [None] * numNodes

    # construct layers for each node
    # layers = |---leaf---|---non_leaf---|
    numLeaf = 0
    for idx in xrange(numNodes):
        node = nodes[idx]
        if len(node.children) == 0:
            numLeaf += 1
        layers[idx] = Lay.layer('vec_' + str(idx) + '_' + node.word, \
                                range(node.bidx, node.bidx + numFea), \
                                numFea
                                )
        layers[idx].act = 'embedding'
    # auto encoding
    # layers = |---leaf---|---non_leaf(autoencoded)---| (numNodes)
    #          |---non_leaf(original)---|               ( numNonLeaf)
    numNonLeaf = numNodes - numLeaf

    layers.extend([None] * (2 * numNonLeaf))

    for idx in xrange(numLeaf, numNodes):
        node = nodes[idx]
        layers[idx + numNonLeaf] = layers[idx]
        tmplayer = Lay.layer('ae_' + str(idx) + '_' + node.word, \
                             Bconstruct, numFea)
        tmplayer.act = 'autoencoding'
        layers[idx] = tmplayer

    # add reconstruction connections
    for idx in xrange(0, numNodes):
        node = nodes[idx]
        if node.parent == None:
            continue
        tmplayer = layers[idx]
        parent = layers[node.parent]
        if node.leftRate != 0:
            leftcon = Con.connection(tmplayer, parent, \
                                     numFea, numFea, Wleft,
                                     Wcoef=node.leftRate * node.leafNum / nodes[node.parent].leafNum)
        if node.rightRate != 0:
            rightcon = Con.connection(tmplayer, parent, \
                                      numFea, numFea, Wright,
                                      Wcoef=node.rightRate * node.leafNum / nodes[node.parent].leafNum)

            # combinition of the constructed and original value
    # layers = |---leaf---|---non_leaf(combinition)---|                (numNodes)
    #          |---non_leaf(original)---|---non_leaf(ae)---|         (2 * numNonLeaf)
    for idx in xrange(numLeaf, numNodes):
        aelayer = layers[idx]
        origlayer = layers[idx + numNonLeaf]
        layers[idx + numNonLeaf * 2] = aelayer

        comlayer = Lay.layer('comb_' + str(idx) + '_' + nodes[idx].word, None, numFea)
        comlayer.act = 'combination'
        layers[idx] = comlayer
        # connecton auto encoded vector and original vector
        con_ae = Con.connection(aelayer, comlayer, numFea, numFea, Wcomb_ae)
        con_orig = Con.connection(origlayer, comlayer, numFea, numFea, Wcomb_orig)



    # CONSTRUCT RECURSIVE LAYER

    queue = [(numNodes - 1, None)]
    rootChildrenNum = len(nodes[-1].children) - 1
    recurLayers = {}  # the map of recursive layer
    # copy leaf
    for idx in xrange(0, numLeaf):  # leaf ---> recursive leaf: in numFea, out numRecur
        recurLayers[idx] = Lay.layer('Recur_' + str(idx) + '_' + nodes[idx].word, \
                                     Brecur, numRecur)
        Con.connection(layers[idx], recurLayers[idx], numFea, numRecur, Wrecur_root)

    while True:
        curLen = len(queue)
        # layerCnt.append( curLen )

        if curLen == 0:
            break
        nextQueue = []

        for (nodeidx, info) in queue:
            curLayer = layers[nodeidx]
            curNode = nodes[nodeidx]

            childNum = len(curNode.children)
            if childNum == 0:  # leaf node
                queue = nextQueue
                continue
            # create recursive node
            if nodeidx not in recurLayers.keys():
                recurLayer = Lay.layer('Recur_' + str(nodeidx) + '_' + curNode.word, \
                                       Brecur, numRecur)
                recurLayer.act = 'recursive'
                # layers.append(recurLayer)
                recurLayers[nodeidx] = recurLayer
            recurLayer = recurLayers[nodeidx]
            # add root connection from Combination layer
            rootCon = Con.connection(curLayer, recurLayer, numFea, numRecur, Wrecur_root)

            # for each child of the current node

            for child in curNode.children:
                childNode = nodes[child]
                if child not in recurLayers.keys():
                    childLayer = Lay.layer('Recur_' + str(child) + '_' + childNode.word, \
                                           Brecur, numRecur)
                    # layers.append(childLayer)
                    recurLayers[child] = childLayer
                childLayer = recurLayers[child]

                nextQueue.append((child, ''))  # add to child
                if childNum == 1:
                    leftWeight = .5
                    rightWeight = .5
                else:
                    rightWeight = childNode.pos / (childNum - 1.0)
                    leftWeight = 1 - rightWeight
                if leftWeight != 0:
                    leftCon = Con.connection(childLayer, recurLayer, \
                                             numRecur, numRecur, Wrecur_left, leftWeight)
                if rightWeight != 0:
                    rightCon = Con.connection(childLayer, recurLayer, \
                                              numRecur, numRecur, Wrecur_right, rightWeight)
            # end of each child of the current node
            queue = nextQueue
            # end of current layer




    # CONVOLVE!!! and POOOOL!!!
    # layers = |---leaf---|---non_leaf(combition)---| =>               (numNodes)
    #          |---non_leaf(original)---|---non_leaf(ae)---| =>        (2 * numNonLeaf)
    #          |------------convolution----------|
    queue = [(numNodes - 1, None)]

    poolLayer = Lay.PoolLayer('pooling', numCon)

    layerCnt = 0
    rootChildrenNum = len(nodes[-1].children) - 1

    while True:
        curLen = len(queue)
        # layerCnt.append( curLen )

        if curLen == 0:
            break
        nextQueue = []

        for (nodeidx, info) in queue:
            curLayer = layers[nodeidx]
            curNode = nodes[nodeidx]

            conLayer = Lay.layer('Convolve_' + curLayer.name, \
                                 Bconv, numCon)
            conLayer.act = 'convolution'
            layers.append(conLayer)
            # add root connection
            rootCon = Con.connection(curLayer, conLayer, numFea, numCon, Wconv_root)

            childNum = len(curNode.children)
            # print curLayer.name, info
            # pooling
            if layerCnt < poolCutoff:
                poolCon = Con.PoolConnection(conLayer, poolLayer)
            else:  # TODO if layerCnt >= poolCutoff
                if info == 'l' or info == 'lr':
                    poolCon = Con.PoolConnection(conLayer, poolLayer)
                if info == 'r' or info == 'lr':
                    poolCon = Con.PoolConnection(conLayer, poolLayer)

            # for each child of the current node

            for child in curNode.children:
                childNode = nodes[child]
                childLayer = layers[child]

                if layerCnt != 0 and info != 'u':
                    childinfo = info
                else:
                    rootChildrenNum = len(curNode.children) - 1
                    if rootChildrenNum == 0:
                        childinfo = 'u'
                    elif childNode.pos <= rootChildrenNum / 2.0:
                        childinfo = 'l'
                    else:  # childNode.pos > rootChildrenNum/2.0:
                        childinfo = 'r'
                        # else:
                        #    childinfo = 'lr'
                nextQueue.append((child, childinfo))  # add to child
                if childNum == 1:
                    leftWeight = .5
                    rightWeight = .5
                else:
                    rightWeight = childNode.pos / (childNum - 1.0)
                    leftWeight = 1 - rightWeight
                if leftWeight != 0:
                    leftCon = Con.connection(childLayer, conLayer, \
                                             numFea, numCon, Wconv_left, leftWeight)
                if rightWeight != 0:
                    rightCon = Con.connection(childLayer, conLayer, \
                                              numFea, numCon, Wconv_right, rightWeight)
            # end of each child of the current node
            queue = nextQueue

        layerCnt += 1
        # end of current layer

    layers.append(poolLayer)

    # reorder
    # layers = |---leaf---|---non_leaf(ae)---| =>               (numNodes)
    #          |---non_leaf(original)---|---non_leaf(comb)---| =>        (2 * numNonLeaf)
    #          |------------convolution----------|
    for idx in xrange(numLeaf, numLeaf + numNonLeaf):
        tmp = layers[idx]
        layers[idx] = layers[idx + 2 * numNonLeaf]
        layers[idx + 2 * numNonLeaf] = tmp
    # discrimitive layer
    # layers = |---leaf---|---non_leaf(ae)---| =>               (numNodes)
    #          |---non_leaf(original)---|---non_leaf(comb)---| =>        (2 * numNonLeaf)
    #          |------------convolution----------|
    #          |---3 pools---|
    ## POOOOOOOOOOOL
    #    # pool all
    #    poolLayer = Lay.PoolLayer('pool', numCon)
    #    for idx in xrange( numNodes + 2* numNonLeaf, len(layers) ):
    #        poolCon = Con.PoolConnection(layers[idx], poolLayer)
    #        poolLayer.connectDown.append(poolCon)
    #        layers[idx].connectUp.append(poolCon)
    #    layers.append(poolLayer)

    # discriminative layer
    # layers = |---leaf---|---non_leaf(ae)---| =>               (numNodes)
    #          |---non_leaf(original)---|---non_leaf(comb)---| =>        (2 * numNonLeaf)
    #          |------------convolution----------|
    #          |---3 pools---|
    #          |---discriminative layer-----|
    #          |--output--|

    lenlayer = len(layers)
    conbegin = lenlayer - 1

    discriminative = Lay.layer('discriminative', Bdis, numDis)
    discriminative.act = 'hidden'
    output = Lay.layer('outputlayer', Boutput, numOut)
    output.act = 'softmax'
    # Pooling ---> fully connected
    con = Con.connection(poolLayer, discriminative, numCon, numDis, Wconv_dis)

    # add recursive layer
    for idx in xrange(0, numNodes):
        layers.append(recurLayers[idx])
    # Recursive ---> fully connected
    rootRecur = recurLayers[numNodes - 1]
    con = Con.connection(rootRecur, discriminative, numRecur, numDis, Wrecur_dis)

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


# jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj
'''

        tmplayer = Lay.layer('ae_'+str(idx)+'_'+node.word,\
                    Bconstruct[0], numFea)

        layers[idx] = tmplayer



            conLayer = Lay.layer('Convolve_' + curLayer.name, \
                             Bconv[0], numCon)
            conLayer.act = 'convolution'


    discriminative = Lay.layer( 'discriminative', Bdis[0], numDis)
    discriminative.act = 'hidden'
    output = Lay.layer('outputlayer', Boutput[0], numOut)

        output.act = 'softmax'

'''