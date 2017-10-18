# -*- coding: utf-8 -*-
"""
Created on Sat May 31 15:07:05 2014

@author: mou
"""
import FFNN



inlayer = FFNN.layer('input', range(0,2), 2)
hidlayer = FFNN.layer('hid',  range(2,4), 2)
outlayer = FFNN.layer('out',  range(4,6), 2)

con1 = FFNN.connection(inlayer, hidlayer, 0, 2, 0, 2, range(0,4), Wcoef = .3)
con2 = FFNN.connection(hidlayer, outlayer, 0, 2, 0, 2, range(4,8), Wcoef = 1)

inlayer.connectUp.append(con1)
hidlayer.connectDown.append(con1)

hidlayer.connectUp.append(con2)
outlayer.connectDown.append(con2)

inlayer.successiveUpper = hidlayer
hidlayer.successiveUpper = outlayer
outlayer.successiveLower = hidlayer
hidlayer.successiveLower = inlayer

import numpy as np
Weights = np.random.uniform(-0.02, .02, 8).reshape(-1)
Biases  = np.random.uniform(-0.02, .02, 6).reshape(-1)

gradWeights = np.zeros_like(Weights)
gradBiases  = np.zeros_like(Biases)

FFNN.forwardpropagation(inlayer, None, Weights, Biases)

outlayer.dE_by_dy = np.ones_like( outlayer.y )

FFNN.backpropagation(outlayer, Weights, Biases, gradWeights, gradBiases)
#    def layer(self, name, Bidx, numunit):
#    def connection(self, xlayer, ylayer, xbegin, xnum, ybegin, ynum, Widx, Wcoef=1):
# Weights, Biases, gradWeights, gradBiases