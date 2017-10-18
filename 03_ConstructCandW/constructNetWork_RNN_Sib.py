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


def ConstructTreeConvolution(nodes, numFea, numRecur, numDis, numOut, \
                             Wleft, Wright, Bconstruct, \
                             Wcomb_ae, Wcomb_orig, \
                             Wrecur_root, Wrecur_left, Wrecur_right, Wrecur_sib, Brecur, \
                             Wdis, Woutput, Bdis, Boutput, \
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

    # CONVOLVE!!! and POOOOL!!!
    # layers = |---leaf---|---non_leaf(combition)---| =>               (numNodes)
    #          |---non_leaf(original)---|---non_leaf(ae)---| =>        (2 * numNonLeaf)
    #          |------------convolution----------|
    queue = [(numNodes - 1, None)]

    rootChildrenNum = len(nodes[-1].children) - 1
    recurLayers ={} # the map of recursive layer
    #copy leaf
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
                recurLayer = Lay.layer('Recur_' + str(nodeidx) + '_'+ curNode.word, \
                                     Brecur, numRecur)
                recurLayer.act = 'recursive'
                #layers.append(recurLayer)
                recurLayers[nodeidx] = recurLayer
            recurLayer = recurLayers[nodeidx]
            # add root connection from Combination layer
            rootCon = Con.connection(curLayer, recurLayer, numFea, numRecur, Wrecur_root)
            # add connection from one previous sibling
            sibs_idx = curNode.siblings
            sibs_idx = [i for i in sibs_idx if i < nodeidx]
            if len(sibs_idx)>0:
                sibs_idx.sort(reverse=True)
                sib_idx = sibs_idx[0]
                sibNode = nodes[sib_idx]
                if sib_idx not in recurLayers.keys():
                    sibLayer = Lay.layer('Recur_' + str(sib_idx) + '_' + sibNode.word, \
                                           Brecur, numRecur)

                    recurLayers[sib_idx] = sibLayer
                sibLayer = recurLayers[sib_idx]

                sib_childrenNum = len(sibNode.children)
                if sib_childrenNum == 0:
                    sib_childrenNum = 1
                sib_Weight = 1.0 * sib_childrenNum / len(curNode.siblings)
                sibCon = Con.connection(sibLayer, recurLayer, \
                                        numRecur, numRecur, Wrecur_sib, sib_Weight)

            # for each child of the current node
            for child in curNode.children:
                childNode = nodes[child]
                if child not in recurLayers.keys():
                    childLayer = Lay.layer('Recur_' + str(child) + '_' + childNode.word, \
                                           Brecur, numFea)
                    #layers.append(childLayer)
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
    # add recursive layer

    for idx in xrange(0, numNodes):
        layers.append(recurLayers[idx])
    # reorder
    # layers = |---leaf---|---non_leaf(ae)---| =>               (numNodes)
    #          |---non_leaf(original)---|---non_leaf(comb)---| =>        (2 * numNonLeaf)
    #          |------------convolution----------|
    for idx in xrange(numLeaf, numLeaf + numNonLeaf):
        tmp = layers[idx]
        layers[idx] = layers[idx + 2 * numNonLeaf]
        layers[idx + 2 * numNonLeaf] = tmp

    # discriminative layer
    # layers = |---leaf---|---non_leaf(ae)---| =>               (numNodes)
    #          |---non_leaf(original)---|---non_leaf(comb)---| =>        (2 * numNonLeaf)
    #          |------------recursive----------|
    #          |---discriminative layer-----|
    #          |--output--|
    lenlayer = len(layers)

    rootRecur = recurLayers[numNodes-1]
    discriminative = Lay.layer('discriminative', Bdis, numDis)
    discriminative.act = 'hidden'
    output = Lay.layer('outputlayer', Boutput, numOut)
    output.act = 'softmax'
    # One Weight Size

    con = Con.connection(rootRecur, discriminative, numFea, numDis, Wdis)

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