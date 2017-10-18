
import os, sys, numpy as np, copy
sys.path.append('../')

# from InitTreeConvolution import *
import gl
import cPickle as p
import test as test
#import FFNN

from InitParam import *
import write_param
sys.setrecursionlimit(100000000)

#tmptokenMap = dict()
#tmpCnt = 0

'''
The folder with suffix '1' contains parameters for experiment B, in which 
learning rate is 0.0025 initially, while the ohter folder contains parameters 
for experiment A, in which learning rate is 0.005 initially, and now is set to 
be 0.001

'''
tobegin  = 6

Config = 'restart' 
# Config = 'load'


if Config != 'continue':
    ######################
    # Configuration
    
    numCon = gl.numCon
    numWords = gl.numWords
    numDis = gl.numDis
    numOut = gl.numOut
    numPool = gl.numPool
    numFea = gl.numFea
    
    # numSen = 10
 
   
    Weights = np.array([])
    Biases = np.array([])
    
    
    f = file('../03_ConstructCandW/word_vector_cache.pkl', 'rb')
    wordsFea = p.load(f)
    Biases, Bword = InitParam(Biases, newWeights = wordsFea)
        
    Bword = np.array(Bword)
    f.close()
    wordsFea = np.array(wordsFea)
    if Config == 'restart':

        
        Weights, Wleft  = InitParam(Weights, num = numFea*numFea)
        Weights, Wright = InitParam(Weights, num = numFea*numFea)
        Biases,  Bconstruct=InitParam(Biases,  num = numFea)
        
        
        # output layer
        Weights, Wout = InitParam(Weights, num = numDis*numOut, upper = .0002, lower = -.0002)
        Biases,  Bout = InitParam(Biases,  newWeights = np.zeros( (numOut,1) ) )
        
        
        Weights = Weights.reshape((-1,1))
        Biases = Biases.reshape((-1,1))
        
        print len(Weights)
        print len(Biases)
        print Biases[0, 0], '     ', Biases[1, 0]
        
        
        write_param.write_binary('../param_pretrain', Weights, Biases)
        dsadssd
       
        tobegin = 0
        
    elif Config == 'load':

        #Weights = CP.load(file('param/param_30_Weights'))
        #Biases  = CP.load(file('param/param_30_Biases' ))
        Weights = p.load(file('param_rollback1/param_'+str(tobegin)+'_Weights'))
        Biases = p.load(file('param_rollback1/param_'+str(tobegin)+'_Biases'))
        
    ##################################
    ##################################
    # load pretraining weights
    gradWeights = np.zeros_like(Weights)
    gradBiases = np.zeros_like(Biases)
    
    ########################## #################
    ###########################################
    # Load data
    
    
    X_train = []
    X_CV    = []
    X_test  = []
    
    y_train = []
    y_CV    = []
    y_test  = []
    
    ####################################
    # ytrain    
    tmpy = []
    
    fy = open('TreeStructure/y_train.txt')
    while True:
        line = fy.readline()
        if not line:
            break
        line = line.strip()
        tmpy.append( float(line) )
    
    
    tmpy = np.array(tmpy)
    tmpy[tmpy<=.2] = 0
    tmpy[(tmpy > .2) & (tmpy <= .4)] = 1
    tmpy[(tmpy > .4) & (tmpy <= .6)] = 2
    tmpy[(tmpy > .6) & (tmpy <= .8)] = 3
    tmpy[(tmpy > .8) & (tmpy < 1)] = 4
    y_train_orig = np.asarray(tmpy, dtype=np.int)
    
    
    
    ####################################
    # y CV
    tmpy = []
    
    fy = open('TreeStructure/y_CV.txt')
    while True:
        line = fy.readline()
        if not line:
            break
        line = line.strip()
        tmpy.append( float(line) )
    
    
    tmpy = np.array(tmpy)
    tmpy[tmpy<=.2] = 0
    tmpy[(tmpy > .2) & (tmpy <= .4)] = 1
    tmpy[(tmpy > .4) & (tmpy <= .6)] = 2
    tmpy[(tmpy > .6) & (tmpy <= .8)] = 3
    tmpy[(tmpy > .8) & (tmpy < 1)] = 4
    y_CV_orig = np.asarray(tmpy, dtype=np.int)
    
    ####################################
    # y test
    tmpy = []
    
    fy = open('TreeStructure/y_test.txt')
    while True:
        line = fy.readline()
        if not line:
            break
        line = line.strip()
        tmpy.append( float(line) )
    
    
    tmpy = np.array(tmpy)
    tmpy[tmpy<=.2] = 0
    tmpy[(tmpy > .2) & (tmpy <= .4)] = 1
    tmpy[(tmpy > .4) & (tmpy <= .6)] = 2
    tmpy[(tmpy > .6) & (tmpy <= .8)] = 3
    tmpy[(tmpy > .8) & (tmpy < 1)] = 4
    y_test_orig = np.asarray(tmpy, dtype=np.int)
    
    
    #####################################
    # X_train
    
    filedir = 'network_train/'
    length = len( y_train_orig )
    for i in xrange(length):
        onefile = filedir + str(i)
        if os.path.exists( onefile ):
            X_train.append( onefile )
            y_train.append( y_train_orig[i] )
                        
    ####################################
    filedir = 'network_CV/'
    length = len( y_CV_orig )
    for i in xrange(length):
        onefile = filedir + str(i)
        if os.path.exists( onefile ):
            X_CV.append( onefile )
            y_CV.append( y_CV_orig[i] )

    ###################################
    filedir = 'network_test/'
    length = len( y_test_orig )
    for i in xrange(length):
        onefile = filedir + str(i)
        if os.path.exists( onefile ):
            X_test.append( onefile )
            y_test.append( y_test_orig[i] )
    
wordsFea = wordsFea.reshape((-1,1))

indicator = np.array([1.0]* len(wordsFea) + [0]*(len(Biases) -  len(wordsFea) ) ).reshape((-1,1))

indicator *= gl.C_embed

wordsFea = np.r_[wordsFea, np.array([0]*(len(Biases) -  len(wordsFea) )).reshape((-1,1))   ]

numToCV = len(y_CV)
numToTrain = len(y_train)
numToTest = len(y_test)


J = 0
print numToCV
print numToTrain

def saveParam(Weights, Biases, ite):
    fout = open('param_rollback1/param_' + str(ite) + '_Weights', 'wb')
    p.dump(Weights,fout)
    fout.close()
    fout = open('param_rollback1/param_' + str(ite) + '_Biases', 'wb')
    p.dump(Biases, fout)
    fout.close()


np.random.seed(16497)
np.random.shuffle(X_train)
np.random.seed(16497)
np.random.shuffle(y_train)
np.random.seed(16497)
np.random.shuffle(X_CV)
np.random.seed(16497)
np.random.shuffle(y_CV)
np.random.seed(16497)
np.random.shuffle(X_test)
np.random.seed(16497)
np.random.shuffle(y_test)

best_accuracy_CV = 0.43
best_accuracy_test = 0.43
t = 0
for ite in xrange(tobegin, 2000):
    for nInst in xrange(t, numToTrain):
        if ite > 0 and nInst % 30000 == 0:
            testJ, accuracy = test.test(X_CV, y_CV, Weights, Biases)
            if accuracy > best_accuracy_CV:
                fout = open('param_rollback1/param_' + str(accuracy) + '_Weights_CV_best', 'wb')
                p.dump(Weights,fout)
                fout.close()
                fout = open('param_rollback1/param_' + str(accuracy) + '_Biases_CV_best', 'wb')
                p.dump(Biases, fout)
                fout.close()
                best_accuracy_CV = accuracy
            elif nInst == 0:
                saveParam(Weights, Biases, ite)
            f = open('History log/log_rollback1.txt', 'a+')
            f.write('ite: ' + str(ite) + ' CV accuracy = ' + str(accuracy) + ' cost = ' + str(testJ/numToCV) + ' cases = '
                + str(numToCV) + '\n')
            f.close()
            f = open('History log/log_rollbacktest1.txt', 'a+')
            testJ, accuracy = test.test(X_test, y_test, Weights, Biases)
            if accuracy > best_accuracy_test:
                fout = open('param_rollback1/param_' + str(accuracy) + '_Weights_test_best', 'wb')
                p.dump(Weights,fout)
                fout.close()
                fout = open('param_rollback1/param_' + str(accuracy) + '_Biases_test_best', 'wb')
                p.dump(Biases, fout)
                fout.close()
                best_accuracy_test = accuracy
            elif nInst == 0:
                saveParam(Weights, Biases, ite)
            f.write('ite: ' + str(ite) + ' test accuracy = ' + str(accuracy) + ' cost = ' + str(testJ/numToTest) + ' cases = '
                + str(numToTest) + '\n')
            f.close()
        fin = open(X_train[nInst], 'rb')
        Xnet = p.load(fin)

        fin.close()
        h = FFNN.forwardpropagation(Xnet[0], None, Weights, Biases)
        t = y_train[nInst]
        
        J -= np.log( h[t] )
        Xnet[-1].dE_by_dy = copy.copy(h)
        Xnet[-1].dE_by_dy[t] -= 1
        Xnet[-1].dE_by_dz = copy.copy(Xnet[-1].dE_by_dy)
        
        FFNN.backpropagation(Xnet[-1], Weights, Biases, gradWeights, gradBiases)
        
        Weights -= gl.alpha * gradWeights 
        Biases  -= gl.alpha * gradBiases
        
        # regularization on embeddings
        #Biases -= (gl.alpha * indicator) * (Biases - wordsFea)
        
       #  Biases[Bword] += gl.C_embed * wordsFea[Bword].reshape((-1,1)) 
        
        gradWeights *= gl.decay
        gradBiases  *= gl.decay
        if nInst % 1000 == 0:
            print 'ite', ite, 'nInst', nInst,  J / 1000
            J = 0
    t = 0
# test.test(X_train[:numToTrain], y_train[:numToTrain], Weights, Biases)

# print 'testing', test.test(X_test, y_test, Weights, Biases)

# ite 14 nInst 1000 J [ 1.39523724]
# ite 14 nInst 2000 J [ 1.30152513]