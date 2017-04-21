import GraphData_IO
import gcnn_params as params

datapath = params.datapath

word_vec, vectors, numFea = GraphData_IO.LoadVocab(vocabfile=datapath + 'tokvec.txt')
#get list of tokens
word_dict ={}
idx =0
for w in word_vec:
    if w.startswith('g_'):
        continue
    word_dict[w] = idx
    idx +=1

def getBagofWordVectors(dtfile ='', outfile='', classlabel =1):

    label ='True' # virus
    if classlabel ==1: # non-virus
        label = 'False'

    f = open(outfile,'w')
    graphs = GraphData_IO.readGraphFromJson(jsonFile=dtfile)
    count =0
    for g in graphs:
        vec =[0] * len(word_dict)
        vertexes = g.getVertexes()
        for v in vertexes:
            token = vertexes[v].token
            if token not in word_dict:
                idx=0
            else:
                idx = word_dict[token]
            vec[word_dict[token]] +=1
        # write vector to file
        vec = [str(i) for i in vec]
        f.write(' '.join(vec))

        f.write(' '+label+'\n')
        count +=1
        if count>=10:
            break
    f.close()
if __name__=='__main__':
    datafile = params.datafiles
    getBagofWordVectors(datafile['train_nonvirus'], datapath + 'vec_train_nonvirus', classlabel=1)
    getBagofWordVectors(datafile['train_virus'], datapath+'vec_train_virus', classlabel=2)
    getBagofWordVectors(datafile['test_nonvirus'], datapath + 'vec_test_nonvirus', classlabel=1)
    getBagofWordVectors(datafile['test_virus'], datapath + 'vec_test_virus', classlabel=2)

