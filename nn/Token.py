# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:34:37 2014

@author: mou
"""

class token:
    word = None
    bidx = None
    parent = None
    children = None
    pos = 0
    leftRate = 0
    rightRate = 0
    sibRate =0
    leafNum = 0
    childrenNum = 0

    def __init__(self, word, bidx, parent, pos = 0):
        self.word = word
        self.bidx = bidx
        self.parent = parent
        self.children = []
        self.siblings =[]
        self.pos = pos
