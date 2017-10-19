import random

import numpy as np

def LoadVocab(vocabfile =''):
    file = open(vocabfile, "r")
    idx =0
    vectors=[]
    dict={}
    vecsize =0
    for line in file:
        items = line.rstrip().split()
        if len(items)<=2:
            vecsize = int(items[1])
            continue
        word= items[0]
        dict[word] = idx
        idx = idx +1

        vectors.append(items[1:])
    vectors = np.reshape(vectors,-1)
    # convert to float
    vectors = [float(i) for i in vectors]
    return dict, vectors, vecsize

# def LoadVocabFromTrain_TestFiles(vocabfiles):
#
#     dict={}
#     dict[''] =0
#     idx =1
#     # get word from vocab files
#     for datafile in vocabfiles:
#         file = open(datafile, "r")
#         for line in file:
#             phrases = line.split('|||')
#             phrase_1 = phrases[0].lstrip().rstrip().split(' ')
#             phrase_2 = phrases[1].lstrip().rstrip().split(' ')
#
#             phrase_1.extend(phrase_2)
#
#             for word in phrase_1:
#                 if word in dict.keys():
#                     continue
#
#                 dict[word] = idx
#                 idx = idx +1
#         file.close()
#
#
#
#     vecsize = 50
#
#     vocabsize = len(dict)
#
#     # convert to float
#     vectors = [random.uniform(-1, 1) for v in xrange(vocabsize)]
#     return dict, vectors, vecsize