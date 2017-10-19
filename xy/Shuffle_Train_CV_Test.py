
import cPickle as p
import numpy as np
import sys,os


#config = 'test'

#numCV = 1101
#numTrain = 161391#159274#176362
#numTrain = 1
#numTest = 2210
#d = '../../network_' + config


network = []
y = []
files = []
#num = numTest

#datadir =  '../../network/'
datadir = 'D:/data/prun_semantic_minor/network/'
pronum = 104
config = 'data'
print 'begin!'
procount = 0
#read training data

for pi in xrange(1,pronum+1):
    subdir = str(pi) + '/'
    #print '111111111111111111111'
    #print 'pi = ',str(pi)
    '''
    if len(os.listdir(datadir + subdir))<500:
        continue
    '''
    procount += 1
    #print 'procount = ',procount
    firstFile = True
    for onefile in os.listdir(datadir + subdir):
        if firstFile:
            print subdir,'/',onefile,'\n'
            firstFile=False
        filename = onefile
        onefile = datadir + subdir + onefile
        f = file(onefile, 'rb')
        network.append((f.read(),procount-1))
        files.append(subdir + filename)
        f.close()


np.random.seed(314159)
np.random.shuffle(network)
np.random.seed(314159)
np.random.shuffle(files)

print len(network)
numTrain = int(.6*len(network))
numCV = int(.8*len(network))
print 'numTrain : ',numTrain
print 'numCV : ' ,numCV-numTrain
print 'numTest',len(network)-numCV
print 'final procount = ',procount
'''
58000
numTrain :  34800
numCV :  11600
numTest 11600
final procount =  116
'''
f_filefold = file('filefold_shuffle', 'w')
f = file(config+'_train', 'wb')
f_y = file(config+'_ytrain.txt', 'w')
for i in xrange(0,numTrain):
    (tf,ti) = network[i]
    f.write(tf)
    #print ti
    f_y.write(str(ti) + '\n')
    f_filefold.write(files[i]+'\n')
f.close()
f_y.close()
#f_filefold.close()

# CV data
#f_filefold = file('filefold_CV', 'w')
f = file(config+'_CV', 'wb')
f_y = file(config+'_yCV.txt', 'w')
for i in xrange(numTrain,numCV):
    (tf,ti) = network[i]
    f.write(tf)
    #print ti
    f_y.write(str(ti) + '\n')
    f_filefold.write(files[i]+'\n')
f.close()
f_y.close()
#f_filefold.close()

#test data
#f_filefold = file('filefold_test', 'w')
f = file(config+'_test', 'wb')
f_y = file(config+'_ytest.txt', 'w')
for i in xrange(numCV,len(network)):
    (tf,ti) = network[i]
    f.write(tf)
    #print ti
    f_y.write(str(ti) + '\n')
    f_filefold.write(files[i]+'\n')
f.close()
f_y.close()
f_filefold.close()
print 'Done!!'