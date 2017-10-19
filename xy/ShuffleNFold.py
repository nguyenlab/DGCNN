
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
g =['2','3']
group = g[0]+'_' + g[1]
datadir =  'D:/data/Fold_Data/'+ group+'/Network/'
pronum = 2
config = group + '_'
print 'begin!'
procount = 0
for pi in xrange(1,pronum+1):
    subdir = str(pi) + '/'
    #print '111111111111111111111'
    print 'pi = ',str(pi)
    '''
    if len(os.listdir(datadir + subdir))<500:
        continue
    '''
    procount += 1
    print 'procount = ',procount
    for onefile in os.listdir(datadir + subdir):
        #print 'oneoneoneoneone!!!!!!!!!! '
        filename = onefile
        onefile = datadir + subdir + onefile
        f = file(onefile, 'rb')
        network.append((f.read(),procount-1))
        files.append(g[procount-1]+'/' + filename)
        f.close()


np.random.seed(314159)
np.random.shuffle(network)
np.random.seed(314159)
np.random.shuffle(files)
# write files fold
f = file(config+'files_fold', 'w')
for i in xrange(0,len(files)):
    f.write(files[i]+'\n')

f.close()

#
# N = len(network)
# print 'num of instances', N
#
# '''
# 58000
# numTrain :  34800
# numCV :  11600
# numTest 11600
# final procount =  116
# '''
# NFold = 10
# for fold in xrange(1, NFold+1):
#     begin = (fold-1) * N / NFold;
#     end = fold * N / NFold;
#     #print 'Fold', fold, 'from ', begin, 'to ', end
#     #write training data
#     f = file(config+'fold'+ str(fold) +'_train', 'wb')
#     f_y = file(config+'fold'+ str(fold) +'_ytrain.txt', 'w')
#
#     for i in xrange(0,begin):
#         (tf,ti) = network[i]
#         f.write(tf)
#         #print ti
#         f_y.write(str(ti) + '\n')
#     for i in xrange(end,N):
#         (tf,ti) = network[i]
#         f.write(tf)
#         #print ti
#         f_y.write(str(ti) + '\n')
#     f.close()
#     f_y.close()
#     # write testing data
#     f = file(config+'fold'+ str(fold) + '_test', 'wb')
#     f_y = file(config+'fold'+ str(fold) +'_ytest.txt', 'w')
#     for i in xrange(begin,end):
#         (tf,ti) = network[i]
#         f.write(tf)
#         #print ti
#         f_y.write(str(ti) + '\n')
#     f.close()
#     f_y.close()

print 'Done!!'