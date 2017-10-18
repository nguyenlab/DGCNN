# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 22:00:14 2014

@author: seke
"""
import os
import numpy as np
ulpast=""#我的文件在aaa文件夹下

txtread=file("StatisticsLines104","r")
txtread2 = file("Statistics104","r")
txtread3 = file("StatisticsMaxDepth104","r")
#txtreadNoPre=file("NoPreCon600alpha0d3NoChange.txt","r")

#txtwrite=open("PreDataRecord.txt","w")
lines = []
numNodes = []
numLayers = []
layerAvgDepth = []
maxDepth = []
txtread.readline()


def stdDeviation(a):
    l=len(a)
    m=sum(a)/l
    d=0.0
    for i in a: 
        d+=(i-m)**2
        
    #print 'd = ',d
    #d1 = d/l
    #print 'd1 = ',d1
    #print 'std = ',d1**0.5
    return (d/l)**0.5

ctmp = 0

a = [2,2,2,2,2,4,4,4,4]
print stdDeviation(a)
while True:
    s=txtread.readline()
    
    if not s:
        break
    lines.append(float(s))
    ctmp += 1
    
    '''

    info = s.split()
        
    train_acc.append(float(info[16]))
    cv_acc.append( float(info[7]))
    test_acc.append( float(info[9]))
    ''' 
#print ctmp        
#y = std(lines,offset = 1) 
print 'lines : ' 
mean = sum(lines)/len(lines)
print mean
std = stdDeviation(lines)
print std
print np.std(lines)  
       
ctmp = 0
txtread2.readline()
while True:
    s=txtread2.readline()
    
    if not s:
        break

    info = s.split()
    numNodes.append(float(info[0]))
    numLayers.append(float(info[3]))
    layerAvgDepth.append(float(info[2]))
    ctmp += 1
    
#print ctmp
    
    
ctmp = 0
txtread3.readline()
while True:
    s=txtread3.readline()

    if not s:
        break
    maxDepth.append(float(s))
    ctmp += 1
    
print 'maxDepth : ' 
mean = sum(maxDepth)/len(maxDepth)
print mean
print np.std(maxDepth,ddof = 1)        
    
print 'numNodes : ' 
mean = sum(numNodes)/len(numNodes)
print mean
std = stdDeviation(numNodes)
print std
print np.std(numNodes)     
        
print 'numLayers : ' 
mean = sum(numLayers)/len(numLayers)
print mean
std = stdDeviation(numLayers)
print std
print np.std(numLayers)         
       
print 'layerAvgDepth : ' 
mean = sum(layerAvgDepth)/len(layerAvgDepth)
print mean
std = stdDeviation(layerAvgDepth)
print std
print np.std(layerAvgDepth)        
        
        
import pylab as plt

#plt.plot(train_acc, 'b')

#if s[0]!='#' and (s[0]!='u' or s[1]!='s') :
    #   txtwrite.write(s)
    
txtread.close()


