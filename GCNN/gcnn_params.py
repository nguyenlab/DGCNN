import os
import random

import GraphData_IO

datapath ='/home/s1520015/Experiment/CodeChef/'
xypath = '/home/s1520015/Experiment/CodeChef/MNMX/xy/'
# datapath ='/home/s1520015/Experiment/5Folds/'
# xypath = '/home/s1520015/Experiment/GCNN_NoOps/2V/xy/'
experiment = 'Virus' # Virus or CodeChef
withOps = False # contain operands in assembly instructions
tokvecFile ='tokvec.txt'    # vector representation
toktype ='dict_tokType.txt' # mapping token ---> type of token

numDis = 600
numOut = 2

numView =1
numCon =[100,600]

# Create vector for tokens
if experiment =='Virus':
    # input: vec_instruction.txt,
    #         vec_tokType.txt
    #output: tokvec
    reCreate = False
    if withOps:
        tokvecFile='vec_embedding_ops.txt'
        print 'mode = Not ignore ops'
        if not os.path.exists(datapath + tokvecFile) or reCreate: # create vector representation
            tokDict, tok_vectors, numFea = GraphData_IO.LoadVocab(vocabfile=datapath + 'vec_instruction.txt')
            toktypeDict, toktype_vectors, numFea = GraphData_IO.LoadVocab(vocabfile=datapath + 'vec_tokType.txt')

            # load list of instructions:ops
            f = open(datapath+'dict_instruction_ops.txt','r')
            inst_ops =[]
            for line in f.readlines():
                inst_ops.append(line.strip())
            f.close()
            # write instruction vec
            f = open(datapath+tokvecFile,'w')
            f.write(str(len(inst_ops)+ len(toktypeDict))+' '+str(numFea)+'\n')
            for inst in inst_ops:
                items = inst.split(',')
                # print items
                if items[0] not in tokDict:
                    idx =0
                else:
                    idx = tokDict[items[0]]
                vec = tok_vectors[idx*numFea: (idx+1)*numFea]
                numitem = 1
                for idx in range(1, len(items)):
                    if items[idx] not in tokDict:
                        print 'not found:',items[idx]
                        continue
                    numitem +=1
                    idx = tokDict[items[idx]]
                    vec =[sum(x) for x in zip(vec,tok_vectors[idx*numFea: (idx+1)*numFea])]
                vec = [i/numitem for i in vec]

                vec =[str(i) for i in vec]
                f.write(inst+' '+ ' '.join(vec)+'\n')
            # write token type vector
            for type, idx in toktypeDict.items():
                vec =toktype_vectors[idx*numFea: (idx+1)*numFea]
                vec = [str(i) for i in vec]
                f.write(type+' '+' '.join(vec)+'\n')
            f.close()
    else:
        tokvecFile ='vec_embedding_no_ops.txt'
        print 'mode = ignore ops'
        if not os.path.exists(datapath + tokvecFile) or reCreate:  # create vector representation
            tokDict, tok_vectors, numFea = GraphData_IO.LoadVocab(vocabfile=datapath + 'vec_instruction.txt')
            toktypeDict, toktype_vectors, numFea = GraphData_IO.LoadVocab(vocabfile=datapath + 'vec_tokType.txt')
            f = open(datapath+tokvecFile,'w')
            f.write(str(len(tokDict)+ len(toktypeDict))+' ' + str(numFea) + '\n')
            for tok, idx in tokDict.items():
                vec = tok_vectors[idx * numFea: (idx + 1) * numFea]
                vec = [str(i) for i in vec]
                f.write(tok + ' ' + ' '.join(vec) + '\n')
            # write token type vector
            for type, idx in toktypeDict.items():
                vec = toktype_vectors[idx*numFea: (idx+1)*numFea]
                vec = [str(i) for i in vec]
                f.write(type + ' ' + ' '.join(vec) + '\n')
            f.close()
def generateTokVec(tokfile, out, vecsize, lower= -1, upper=1):
    # randomly generate vector representation for tokens
    toks =[]
    f = open(tokfile,'r')
    for line in f.readlines():
        toks.append(line.strip())
    f.close()
    # randomly generate
    f = open(out,'w')
    f.write(str(len(toks))+' '+str(vecsize)+'\n')
    for t in toks:
        vec = [random.uniform(lower, upper) for v in xrange(vecsize)]
        vec = [str(i) for i in vec]
        f.write(t +' '+' '.join(vec)+'\n')
    f.close()
def generateTokTypeVec(toktypefile, out, vecsize,lower= -1, upper=1):
    # randomly generate vector representation for tokens
    toks ={}
    f = open(toktypefile,'r')
    f.readline() # ignore header row
    for line in f.readlines():
        items = line.strip().split(' ')
        toks[items[1]] = items[1]
    f.close()
    # randomly generate
    f = open(out,'w')
    f.write(str(len(toks))+' '+str(vecsize)+'\n')
    for t in toks:
        vec = [random.uniform(lower, upper) for v in xrange(vecsize)]
        vec = [str(i) for i in vec]
        f.write(t +' '+' '.join(vec)+'\n')
    f.close()

# random.seed(314159)
# generateTokVec(datapath+'dict_instruction_no_ops.txt', datapath+'vec_instruction.txt',vecsize=30)
# generateTokTypeVec(datapath+'dict_tokType.txt', datapath+'vec_tokType.txt', vecsize=30)

# load token types dictionary
# toktypeDict = None
# if os.path.exists(datapath + 'toktypeDict.txt'):
#     toktypeDict = GraphData_IO.LoadTokenTypeDict(filename=datapath + 'toktypeDict.txt')
#     print 'Load token types from: ', datapath+ 'toktypeDict.txt'
# else:
#     print 'Not found:', datapath+'toktypeDict.txt'

# datafiles ={}
# datafiles['train_virus'] = datapath+'Training/GCNN.json'
# datafiles['train_nonvirus'] = datapath+'Training/NonVirus.json'
# datafiles['test_virus'] = datapath+'Testing/GCNN.json'
# datafiles['test_nonvirus'] = datapath+'Testing/NonVirus.json'

# groups =['g_unknown','g_control','g_arithmetic','g_call','g_move','g_return','g_cond_jump','g_jump']
#
# for g in groups:
#     vec = [random.uniform(-1, 1) for v in xrange(30)]
#     vec = [str(i) for i in vec]
#     print  g +' '+' '.join(vec)
# numTok = 1424
# print '@relation dataset\n'
# for i in range(numTok):
#     print '@attribute F'+str(i+1)+' numeric'
# print '@attribute F'+str(numTok+1)+'{True, False}'
# print '\n@data\n'