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


def ConstructTreeConvolution(phrase_1, phrase_2, word_dict, numFea,numLeft, numRight, numJoint, numDis, numOut, \
                             Wleft, Wright, Bleft, Bright,
                             Wjoint_left, Wjoint_right, Bjoint,
                             Wdis, Wout, Bdis, Bout
                             ):
    # nodes
    # numFea: # of the word/symbol feature size
    phrase_1_len = len(phrase_1)
    phrase_2_len = len(phrase_2)


    layer_left = Lay.layer('left', Bleft, numLeft)
    layer_right = Lay.layer('right', Bright, numRight)
    #embedding layer 1
    emb_layer1 =[None] * phrase_1_len
    layers =[]

    for idx in xrange(phrase_1_len):
        word = phrase_1[idx]

        if word in word_dict.keys():
            # get index of the word in dictionary
            bidx = word_dict[word] * numFea

            emb_layer1[idx] = Lay.layer('vec_' + word + '_', \
                                    range(bidx, bidx + numFea), \
                                    numFea
                                    )
            emb_layer1[idx].act = 'embedding'
            con_left = Con.connection(emb_layer1[idx], layer_left, numFea, numLeft, Wleft)
            layers.append(emb_layer1[idx])


    #embedding layer 2
    emb_layer2 = [None] * phrase_2_len

    for idx in xrange(phrase_2_len):
        word = phrase_2[idx]

        if word in word_dict.keys():
            # get index of the word in dictionary
            if word in word_dict:
                bidx = word_dict[word] * numFea
            else:
                bidx =0
            emb_layer2[idx] = Lay.layer('vec_' + word + '_', \
                                        range(bidx, bidx + numFea), \
                                        numFea
                                        )
            emb_layer2[idx].act = 'embedding'
            con_right = Con.connection(emb_layer2[idx], layer_right, numFea, numRight, Wright)
            layers.append(emb_layer2[idx])

    layers.append(layer_left)
    layers.append(layer_right)

    joint = Lay.layer('joint', Bjoint, numJoint)
    con_left =  Con.connection(layer_left, joint, numLeft, numJoint, Wjoint_left)
    con_right = Con.connection(layer_right, joint, numRight, numJoint, Wjoint_right)
    layers.append(joint)

    discriminative = Lay.layer('discriminative', Bdis, numDis)
    discriminative.act = 'hidden'
    discon = Con.connection(joint, discriminative, numJoint, numDis, Wdis)

    output = Lay.layer('outputlayer', Bout, numOut)
    output.act = 'softmax'
    outcon = Con.connection(discriminative, output, numDis, numOut, Wout)

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