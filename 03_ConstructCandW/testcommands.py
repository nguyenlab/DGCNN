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
# path ='Z:/Experiment/Yen_Results/CodeChef/'
# file ='out-decision-ast_out_sumtrian_42.txt'
# f= open(path+file,'r')
# lines = f.readlines()
# f.close()
#
# lines[0] = lines[0].strip()
# dec_vals = lines[0].split(' ')
# for v in dec_vals:
#     print v
# print 'End value\n'
# lines[1] = lines[1].strip()
# truelabel = lines[1].split(' ')
# for l in truelabel:
#     print l
# def toGraphViz(Vs, Es):
#     gviz = 'digraph finite_state_machine { node [shape = rectangle];\n'
#
#     for id in Vs:
#         gviz +='v{0}_{1}[label="{0}\\n{1}"];\n'.format(id, Vs[id])
#     for (v1, v2) in Es:
#         gviz += 'v{0}_{1} -> v{2}_{3};\n'.format(v1, Vs[v1],v2, Vs[v2])
#
#     gviz+='}'
#     return gviz
# def CheckConnect(Vs, Es):
#     labels ={}
#     for idx, vid in enumerate(Vs.keys()):
#         labels[vid] = idx
#     for vid1, vid2 in Es:
#         oldlabel = labels[vid1]
#         for vid, l in labels.items():
#             if l == oldlabel:
#                 labels[vid] = labels[vid2]
#             labels[vid1] = labels[vid2]
#         print labels
#     mainlabel = labels[1]
#     connectedmain =[]
#     for vid, l in labels.items():
#         if l == mainlabel:
#             connectedmain.append(vid)
#     print connectedmain
#
# Vs ={0:'v0',1:'v1',2:'v2',3:'v3',4:'v4'}
# Es =[(1,3),(0,2),(2,3),(1,4)]
# g = toGraphViz(Vs, Es)
# print g
# CheckConnect(Vs, Es)
# import re
#
# val = 'val'
# reg = 'reg'
# name = 'name'
# def getParamType(ptype):
#     if ptype.startswith('%') or ptype.__contains__('(%'):
#         return reg
#     if re.match('[-\$]*\d+$', ptype, flags=0) or ptype.startswith('"'):
#         return val
#     return name
#
# print getParamType('"%lld%lld"')

s = '00000_0.c'
id1 = s.find('_')
id2 = s.find('.c')
print s[id1+1:id2]