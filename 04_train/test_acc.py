import sys, cPickle as p, numpy as np
sys.path.append('../nn/')
import FFNN
import copy

def test(Xtest, ytest, Weights, Biases):
    testNum = len(ytest)
    correct = 0
    J = 0
    avg = 0
    for i in xrange(testNum):
        if type(Xtest[i]) == type('a'):
            fin = open(Xtest[i])
            X = p.load(fin)
            fin.close()
        else:
            X = Xtest[i]
        #print len(X)
        FFNN.cleanActivation(X[0])
        h = FFNN.forwardpropagation(X[0], None, Weights, Biases)
        #print h.T
        t = ytest[i]
        
        J +=  - np.log( h[t] )
        avg += h[t]
        predict = h.argmax()
        
        #print 'test case (', i, '), h = ', predict, '(predicted ', h[predict], ')',  t, '(actural',\
        #        h[t], ')'
        if predict == t:
            correct += 1
    print '       average target output', (avg + .0)/testNum
#    if bestAccuracy < .2:
#        print 'oops, too bad'
    print J, (correct + .0) /testNum
    return J / testNum, (correct + .0)/ testNum
