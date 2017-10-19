import numpy as np

def readTextFile(filename=''):
    # read a file and an arrays of texts of lines
    file = open(filename, "r")
    texts =[]
    for line in file:
        line = line.rstrip()
        if line != '':
            texts.append(line)
    return texts
def createWordDict(filename=''):
    # read all words and create a dict of word-id
    words = readTextFile(filename=filename)
    id =0
    dict ={}
    for w in words:
        if w in dict:
            print w
            continue
        dict[w] = id
        id +=1
    return dict
def saveArray2Text(array, filename):
    with open(filename, 'w') as f:
        for item in array:
            f.write(str(item)+'\n')

def saveArray(array, filename,separator=', '):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(filename, 'w') as f:
        f.write(np.array2string(array, separator=separator))
def loadArray(filename, type=''):
    array = np.loadtxt(filename, delimiter=', ')
    if type =='float':
        array = np.apply_along_axis(lambda y: [float(i) for i in y], 0, array)
    if type =='int':
        array = np.apply_along_axis(lambda y: [int(i) for i in y], 0, array)
    return array