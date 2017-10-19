# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:47:56 2014

@author: mou
"""

import numpy as np
from helper import *
    

def cleanDerivatives(layer):
    curLayer = layer
    while curLayer != None:
        curLayer.dE_by_dz = None
        curLayer.dE_by_dy = None
        curLayer = curLayer.successiveLower
        
def cleanActivation(layer0):
    curLayer = layer0
    while True:
        curLayer.z = None
        curLayer.y  = None
        if curLayer.successiveUpper == None:
            break
        else:
            curLayer = curLayer.successiveUpper 
    
def fpWrapper(x, theta):
    layer0, X, numW, target = x
    W = theta[0:numW]
    b = theta[numW:]
    cleanActivation(layer0)
    y = forwardpropagation(layer0, X, W, b)
    return computeMSE(target, y)
    
def forwardpropagation(layer0, X, W, b): # TODO remove Xnet
    if X != None: 
        numData = X.shape[1]
        layer0.y = X
        layer0.z = X
    else:
        numData = 1
    # with size numUnitCur x numData
    curLayer = layer0
    while True:
        # apply the activation function
        curLayer.computeY(b, numData)
        
        if curLayer.successiveUpper == None:
            return curLayer.y
        # feed forward
        for con in curLayer.connectUp:
            con.forwardprop(W, numData)
        curLayer = curLayer.successiveUpper        
        
    ### end of while loop over all layers
### end of forward propagation

def backpropagation(outlayer, Weights, Biases, gradWeights, gradBiases):
    
    numData = outlayer.y.shape[1]
    curLayer = outlayer
    while curLayer != None: # for each layer
        # dE/dy has size <numOutput> by <numData>
        #if curLayer.dE_by_dz == None and curLayer.z != None:
        if curLayer.dE_by_dy == None:
            curLayer = curLayer.successiveLower
            continue
        curLayer.updateB(gradBiases)
        
        # back propogation        
        for con in curLayer.connectDown:
            con.backprop(Weights, gradWeights, numData)
        # end of each upward connection
        curLayer = curLayer.successiveLower
    # end of all layers
        
    pass

def displayNetwork(layers, Weights, Biases):
    for idx, curlayer in enumerate(layers):
        
        print 'layer', idx, '================================'

        #curlayer.display(Biases, ['connections', 'activation', 'derivatives'])
        curlayer.display(Biases)
        if curlayer.connectUp == None:
            print '    parents: None'
        else:
            for con in curlayer.connectUp:
                con.display(Weights, 'parents')

        if curlayer.connectDown == None:
            print '    children: None'
        else:
            for con in curlayer.connectDown:
                con.display(Weights, 'children')
                      
        if curlayer.successiveUpper == None:
            print '    successive upper layer: None'
        else:
            print '    successive upper layer:', curlayer.successiveUpper.name
        if curlayer.successiveLower == None:
            print '    successive lower layer: None'
        else:
            print '    successive lower layer:', curlayer.successiveLower.name
        print ''

