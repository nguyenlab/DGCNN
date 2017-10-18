import os
import numpy as np

def readAdditional():
    X_train = []
    y_train = []
    
    ####################################
    # ytrain    
    tmpy = []
    
    fy = open('../y/y_train_small.txt')
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

    
    filedir = '../nets/net_train_small/'
    length = len( y_train_orig )
    for i in xrange(length):
        onefile = filedir + str(i) + '.txt'
        if os.path.exists( onefile ):
            X_train.append( onefile )
            y_train.append( y_train_orig[i] )
                        
    return X_train, y_train
    re
    
def readXy():
    X_train = []
    X_CV    = []
    X_test  = []
    
    y_train = []
    y_CV    = []
    y_test  = []
    
    ####################################
    # ytrain    
    tmpy = []
    
    fy = open('../Xy/y_train')
    while True:
        line = fy.readline()
        if not line:
            break
        line = line.strip()
        tmpy.append( float(line) )
    
    
    tmpy = np.array(tmpy)
    #tmpy[tmpy <= .5] = 0
    #tmpy[tmpy > .5] = 1
    tmpy[tmpy<=.2] = 0
    tmpy[(tmpy > .2) & (tmpy <= .4)] = 1
    tmpy[(tmpy > .4) & (tmpy <= .6)] = 2
    tmpy[(tmpy > .6) & (tmpy <= .8)] = 3
    tmpy[(tmpy > .8) & (tmpy < 1)] = 4
    y_train_orig = np.asarray(tmpy, dtype=np.int)
    
    
    
    ####################################
    # y CV
    tmpy = []
    
    fy = open('../Xy/y_CV')
    while True:
        line = fy.readline()
        if not line:
            break
        line = line.strip()
        tmpy.append( float(line) )
    
    
    tmpy = np.array(tmpy)
    
    #tmpy[tmpy <= .5] = 0
    #tmpy[tmpy > .5] = 1
    tmpy[tmpy<=.2] = 0
    tmpy[(tmpy > .2) & (tmpy <= .4)] = 1
    tmpy[(tmpy > .4) & (tmpy <= .6)] = 2
    tmpy[(tmpy > .6) & (tmpy <= .8)] = 3
    tmpy[(tmpy > .8) & (tmpy < 1)] = 4
    y_CV_orig = np.asarray(tmpy, dtype=np.int)
    
    ####################################
    # y test
    tmpy = []
    
    fy = open('../Xy/y_test')
    while True:
        line = fy.readline()
        if not line:
            break
        line = line.strip()
        tmpy.append( float(line) )
    
    
    tmpy = np.array(tmpy)
    
    #tmpy[tmpy <= .5] = 0
    #tmpy[tmpy > .5] = 1
    # five classes
    tmpy[tmpy<=.2] = 0
    tmpy[(tmpy > .2) & (tmpy <= .4)] = 1
    tmpy[(tmpy > .4) & (tmpy <= .6)] = 2
    tmpy[(tmpy > .6) & (tmpy <= .8)] = 3
    tmpy[(tmpy > .8) & (tmpy < 1)] = 4
    y_test_orig = np.asarray(tmpy, dtype=np.int)
    
    
    #####################################
    # X_train
    
    filedir = '../nets/net_train/'
    length = len( y_train_orig )
    for i in xrange(length):
        onefile = filedir + str(i) + '.txt'
        if os.path.exists( onefile ):
            X_train.append( onefile )
            y_train.append( y_train_orig[i] )
    print 'strange:', len(X_train)
                 
    ####################################
    filedir = '../nets/net_CV/'
    length = len( y_CV_orig )
    for i in xrange(length):
        onefile = filedir + str(i) + '.txt'
        if os.path.exists( onefile ):
            X_CV.append( onefile )
            y_CV.append( y_CV_orig[i] )

    ###################################
    filedir = '../nets/net_test/'
    length = len( y_test_orig )
    for i in xrange(length):
        onefile = filedir + str(i) + '.txt'
        if os.path.exists( onefile ):
            X_test.append( onefile )
            y_test.append( y_test_orig[i] )
            
    return X_train, X_CV, X_test, y_train, y_CV, y_test
    

if __name__ == '__main__':
    X_train, X_CV, X_test, y_train, y_CV, y_test = readXy()
