
import cPickle as p
import numpy as np



config = 'test'

numCV = 1101
numTrain = 161391#159274#176362
#numTrain = 1
numTest = 2210
d = '../../network_pretrain_' + config

network = []
y = []
num = numTest
for i in xrange(num):
    f = file(d + '/' + str(i), 'rb')
   
    network.append(f.read())
    f.close()

f_y = open('y_' + config + '.txt')
line = f_y.readline()
n = 0
while line != None:
    line = line.strip()
    if line == '':
        break
    y.append(line)
    line = f_y.readline()
f_y.close()


   
print len(network)

f = file('network_pretrain_' + config + '_large', 'wb')
f_y = file('y_' + config + '_shuffled.txt', 'w')
for i in xrange(num):
    f.write(network[i])
    f_y.write(y[i] + '\n')
f.close() 
f_y.close()



config = 'CV'


d = '../../network_pretrain_' + config

network = []
y = []
num = numCV
for i in xrange(num):
    f = file(d + '/' + str(i), 'rb')
   
    network.append(f.read())
    f.close()

f_y = open('y_' + config + '.txt')
line = f_y.readline()
n = 0
while line != None:
    line = line.strip()
    if line == '':
        break
    y.append(line)
    line = f_y.readline()
f_y.close()

   
print len(network)

f = file('network_pretrain_' + config + '_large', 'wb')
f_y = file('y_' + config + '_shuffled.txt', 'w')
for i in xrange(num):
    f.write(network[i])
    f_y.write(y[i] + '\n')
f.close() 
f_y.close()



config = 'train'
d = '../../network_pretrain_' + config


fout_y = file('y_' + config + '_shuffled.txt', 'w')
fout_x = file( 'network_pretrain_' + config + 
        '_large', 'wb' )

network = []
y = []
num = numTrain
for i in xrange(num):
    f = file(d + '/' + str(i), 'rb')
   
    network.append(f.read())
    f.close()

f_y = open('y_' + config + '.txt')
line = f_y.readline()
n = 0
while line != None:
    line = line.strip()
    if line == '':
        break
    y.append(line)
    line = f_y.readline()
f_y.close()

np.random.seed(164937)
np.random.shuffle(network)
np.random.seed(164937)
np.random.shuffle(y)


for i in xrange(num):
    n = n + 1
    fout_x.write(network[i])
    fout_y.write(y[i] + '\n')
  
 

print len(y)   
    
print n
fout_x.close()
fout_y.close()
