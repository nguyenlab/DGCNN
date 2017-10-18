import GraphData_IO

def getBagofWordVectors(jsonFiles =''):

    word_dict={}
    for file in jsonFiles:
        fout = open(file+'.bow','w')
        graphs = GraphData_IO.readGraphFromJson(jsonFile=file)
        for g in graphs:
            vec ={}
            for vid, v in g.Vs.items():
                tok = v.token
                if tok not in word_dict:
                    word_dict[tok] = len(word_dict)+ 1
                tok_id = word_dict[tok]
                if tok_id not in vec:
                    vec[tok_id] = 1
                else:
                    vec[tok_id] +=1
            # write vector
            fout.write('label_'+str(g.label)+' ')
            feature_vec =[]

            for tok_id in sorted(vec):
                feature_vec.append(str(tok_id)+':'+str(vec[tok_id]))
            fout.write(' '.join(feature_vec))
            fout.write('\n')
        # close file
        fout.close()

if __name__=='__main__':
    datapath = '/home/s1520015/Experiment/5Folds/'
    datafiles = []
    for idx in range(1, 6):
        datafiles.append(datapath + '/Fold' + str(idx) + '/dataFold' + str(idx))

    getBagofWordVectors(datafiles)

    datapath = '/home/s1520015/Experiment/CodeChef/OriginalTrees/'
    problem = 'SUMTRIAN'
    getBagofWordVectors([datapath + problem + '_train_AstGraph.json', datapath + problem + '_CV_AstGraph.json', datapath + problem + '_test_AstGraph.json'])
    problem = 'MNMX'
    getBagofWordVectors([datapath + problem + '_train_AstGraph.json', datapath + problem + '_CV_AstGraph.json', datapath + problem + '_test_AstGraph.json'])
    problem = 'FLOW016'
    getBagofWordVectors([datapath + problem + '_train_AstGraph.json', datapath + problem + '_CV_AstGraph.json', datapath + problem + '_test_AstGraph.json'])
    problem = 'SUBINC'
    getBagofWordVectors([datapath + problem + '_train_AstGraph.json', datapath + problem + '_CV_AstGraph.json', datapath + problem + '_test_AstGraph.json'])
