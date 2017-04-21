import random

import numpy as np
# intarr= [0,1,2,3,4,5]
# a= np.reshape(intarr, (2, 3)) # C-like index ordering
# print a
# a = np.reshape(intarr, (2, 3), order='A') # equivalent to C ravel then C reshape
# print a
# a = np.reshape(intarr, (2, 3), order='F') # Fortran-like index ordering
# print a
# a = np.reshape(intarr, (-1,2)) # Fortran-like index ordering
# print a
# print intarr
# import parser
# text =""
# ast = parser.parse(text=text)
# numFea = 4
# w1 = (np.eye(numFea) / 2)
# print w1
# w1 = w1.reshape(-1)
# print w1
# numVertexes = 3
# numView = 4
# count =1
# emb_layers = numView * [None]
# for v in range(0, numView):
#     emb_layers[v] = numVertexes*[None]
#     for vertex in range(0, numVertexes):
#         emb_layers[v][vertex] = count
#         count +=1
#
# print emb_layers
# inputLayers= [None] * numVertexes
# for idx in range(0, numVertexes):
#     inputLayers[idx] = idx
# print  inputLayers
# tokens =['val','reg']
# vecsize = 30
# for tok in tokens:
#     vec = [random.uniform(-1, 1) for v in xrange(vecsize)]
#     vec = [str(i) for i in vec]
#     print tok + ' ' +' '.join(vec)
# data = ['ABC']
# data.append('g_unknown')
# print data
# s ='_strlwr_msvcrt.dll'
# idx = s.find('_')
# s= s.strip('_')
# print idx, s

# import re
# s = "__set_app_type_msvcrt.dl"
# replaced = re.sub('(^_*)|(.|_)(dll|dl)$', '', s)
# print replaced
path ='Z:/Experiment/Yen_Results/CodeChef/'
file ='out-decision-ast_out_sumtrian_42.txt'
f= open(path+file,'r')
lines = f.readlines()
f.close()

lines[0] = lines[0].strip()
dec_vals = lines[0].split(' ')
for v in dec_vals:
    print v
print 'End value\n'
lines[1] = lines[1].strip()
truelabel = lines[1].split(' ')
for l in truelabel:
    print l
