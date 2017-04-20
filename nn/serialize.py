import numpy as np
import sys
import cPickle as p
sys.path.append('../')
import gl, FFNN
import copy, Activation as act
import struct

def write_data(fname, X_train, X_CV, X_test, y_train, y_CV, y_test):
    f = file(fname, 'w')
    
    f.write( str(len(X_train)) + '\n')
    f.write( str(len(X_CV))    + '\n')
    f.write( str(len(X_test))  + '\n')
    
    for i in X_train:
        f.write(str(i) + '\n')
    for i in X_CV:
        f.write(str(i) + '\n')
    for i in X_test:
        f.write(str(i) + '\n')
    
    for i in y_train:
        f.write(str(i) + '\n')
    for i in y_CV:
        f.write(str(i) + '\n')
    for i in y_test:
        f.write(str(i) + '\n')

    f.close()


def serialize(layers, fname):
    
    f = file( fname, 'wb' )
    num_lay = struct.pack('i', len(layers) )
    if num_lay <=2 :
        print error
    f.write( num_lay)
    
    num_con = 0
    
    #################################
    # preprocessing, compute some indexes
    for i, layer in enumerate(layers):
        layer.idx = i
        num_con += len(layer.connectUp )
        for (icon, con) in enumerate(layer.connectDown):
            con.ydownid = icon
            
    num_con = struct.pack('i', num_con )
    f.write( num_con )
    
    #################################
    # layers

    for layer in layers:
        # name
        # = struct.pack('s', layer.name )
        # numUnit        
        tmp = struct.pack('i', layer.numUnit )
        f.write( tmp )
        # numUp
        tmp = struct.pack('i', len(layer.connectUp))
        f.write( tmp )
        # numDown
        tmp = struct.pack('i', len(layer.connectDown))
        f.write( tmp )
        
        
        if  layer.layer_type == 'p': # pooling
            if layer.poolType == 'max':
                tlayer = 'x'
            elif layer.poolType == 'sum':
                tlayer = 'u'
            tmp = struct.pack('c', tlayer)
            f.write( tmp )
            
        elif layer.layer_type == 'o': # ordinary nodes

            if layer.act == 'embedding':
                tlayer = 'e'
            elif layer.act == 'autoencoding':
                tlayer = 'a'
            elif layer.act == 'convolution':
                tlayer = 'c'
            elif layer.act == 'combination':
                tlayer = 'b'
            elif layer.act == "ReLU":
                tlayer = 'r'
            elif layer.act == 'softmax':
                tlayer = 's'
            elif layer.act == 'hidden':
                tlayer = 'h'
            elif layer.act == 'recursive':
                tlayer = 'v'
            else:
                print "error"
                return layer
            tmp = struct.pack('c', tlayer)


            f.write( tmp )
            bidx = -1
            if layer.bidx != None:
                bidx = layer.bidx
                bidx = bidx[0]
            
            tmp = struct.pack('i', bidx)
            f.write(tmp)
    
    #########################
    # connections
    for layer in layers:
        for xupid, con in enumerate(layer.connectUp):
            # xlayer idx   
            tmp = struct.pack('i', layer.idx)
            
            f.write( tmp )
            # ylayer idx
            tmp = struct.pack('i', con.ylayer.idx )
            f.write( tmp)
            # idx in x's connectUp 
            tmp = struct.pack('i', xupid)
            f.write( tmp )
            # idx in y's connectDown
            tmp = struct.pack('i', con.ydownid)
            f.write( tmp )
            if con.ylayer.layer_type == 'p':
                Widx = -1
            else:
                Widx = con.Widx
                Widx = Widx[0]
          
            tmp = struct.pack('i', Widx)
            f.write(tmp)
            if Widx >= 0:
                tmp = struct.pack('f', con.Wcoef)
                f.write( tmp )
    f.close()
