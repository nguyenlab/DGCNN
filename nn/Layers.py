# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:41:15 2014

@author: mou
"""
import numpy as np
import Activation as activate
from helper import *

class layer:
    layer_type = "o" # ordinary
    act = "ReLU"
    name = None
    attr = None
    idx = 0
    _activate = activate.dummyReLU
    _activatePrime = activate.dummyReLUPrime
    #_activate = dummyIdentity
    #_activatePrime = dummyEye    
    numUnit = 0
    z = None # with size numUnit by numData
    y = None # with size numUnit by numData
    bidx = None # the biases lower the level
    # with size numUnit
    dE_by_dz = None
    dE_by_dy = None
    connectUp = None
    # (layer, lbegin, lnum, ubegin, unum)
    connectDown = None
    successiveUpper = None
    successiveLower = None
    # get b
    
    def __init__(self, name, Bidx, numunit):
        ''' initialized the weight indices
            Bidx is the biase indices
        '''
        self.name = name
        self.numUnit = numunit
        self.bidx = Bidx
        self.connectUp = []
        self.connectDown = []
        pass
    
    def f(self, x):
        return self._activate(x)
    def fprime(self):
        return self._activatePrime(self.y)
    def computeY(self, b, numData):
      try:
        if self.z == None and self.y == None:
            self.y = b[self.bidx].reshape(-1,1)
        if self.y == None:
            if self.bidx != None:
                bias = b[self.bidx] 
                bias = bias.reshape(-1,1) # convert into column 
                self.z += np.tile(bias, (1, numData))
            self.y = self.f( self.z )
      except:
        print self.name, self.bidx[:5]
        asdfs
    def updateB(self, gradBiases):
        if self.dE_by_dy == None:
            return
        if self.bidx == None:
            self.dE_by_dz = self.dE_by_dy * self.fprime()
            return
        if self.z != None:
            if self.dE_by_dz == None:
                self.dE_by_dz = self.dE_by_dy * self.fprime()
            dEdb = np.sum(self.dE_by_dz, 1).reshape((-1,1)) 
        else: # y = b
            dEdb = self.dE_by_dy
        #gradient of curlayer.b
        self.bidx = np.array(self.bidx)
        gradBiases[self.bidx] += dEdb

    def display(self, Biases, para = ['connections']):
        if self.bidx == None:
            print 'name = ', self.name, \
                  'without biases'
        else:
            print 'name = ', self.name, \
              'with biases = ', self.bidx[0:5], '=', Biases[self.bidx]
        if 'activation' in para:
            print '  y:', self.y
            print '  z:', self.z
        if 'derivatives' in para:
            print '  dE_by_dy', self.dE_by_dy
            print '  dE_by_dz', self.dE_by_dz


class PoolLayer:
    idx = 0
    layer_type = 'p' # pooling
    name = None
    attr = None
    poolType = None
    numUnit = 0
    z = None # with size numUnit by numData
    y = None # with size numUnit by numData
    # with size numUnit
    dE_by_dz = None
    dE_by_dy = None
    connectUp = None
    connectDown = None
    successiveUpper = None
    successiveLower = None
    # get b
    
    def __init__(self, name, numunit, poolType = 'max'):
        ''' initialized the weight indices
            Bidx is the biase indices
        '''
        self.name = name
        self.numUnit = numunit
        self.connectUp = []
        self.connectDown = []
        self.poolType = poolType
        pass
    def computeY(self, b, numData):
        if self.y != None:
            return
        if self.poolType == 'max':
            if self.z == None:
                self.y = np.zeros((self.numUnit,numData)) #+ .5
                self.z = self.y
            else:
                self.y = np.max(self.z, axis = 0)
        elif self.poolType == 'sum':
            if self.z == None:
                self.z = np.zeros((self.numUnit,numData))
                self.y = self.z
            else:
                self.y = np.sum(self.z, axis = 0)
    def updateB(self, gradBiases = None):
        pass
    def display(self, Biases, para = ['connections']):
        print 'name = ', self.name, 'type = ', self.poolType, 'pooling, numUnit =', self.numUnit
        
#import FFNN
#pool = PoolLayer('pool', 3)
#FFNN.cleanActivation(pool)
#FFNN.forwardpropagation(pool, None, None, None)
