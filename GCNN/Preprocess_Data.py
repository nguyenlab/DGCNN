import os
from shutil import copyfile
import numpy as np

def insertText(filename='', libs=''):

    file = open(filename, 'r+b')
    #return top of file
    content = file.read()
    file.seek(0)
    file.truncate()
    file.write(libs)
    file.write(content)
    file.close()

def replaceText(filename,oldString, newString):
    file = open(filename, 'r+b')
    # return top of file
    content = file.read()
    content=content.replace(oldString, newString)
    file.seek(0)
    file.truncate()

    file.write(content)
    file.close()

def addLibs():
    datadir = 'D:/data/data_Ctype/'
    # datadir ='D:/C_SourceCode/'
    #
    #     libs ="""#include<stdio.h>
    # #include<iostream>
    # #include <iomanip>
    # #include <math.h>
    # # include<stdlib.h>
    # # include<string.h>
    # using namespace std;
    # """

    count = 0
    for subdir in os.listdir(datadir):
        if not subdir.endswith('/'):
            subdir = subdir + '/'

        if os.path.isfile(datadir + subdir[:len(subdir) - 1]):
            continue

        print '!!!!!!!!!!!!!!!!!!  procount = ', subdir
        for onefile in os.listdir(datadir + subdir):
            filename = onefile
            onefile = datadir + subdir + onefile

            if filename.endswith(".c"):
                if not os.path.isfile(onefile + '.exe'):  # if not be compiled
                    count += 1
                    # addLib2FileC(filename=onefile,libs = libs)
                    # replaceText(filename= onefile,oldString='void main',newString= 'int main')
                    print filename
    print 'No. of files:', count
def copyExe(sourcedir, desdir):
    for subdir in os.listdir(sourcedir):
        if not subdir.endswith('/'):
            subdir = subdir + '/'
        subdirName = subdir.replace('/','')

        if os.path.isfile(sourcedir + subdir[:len(subdir) - 1]):
            continue

        print '!!!!!!!!!!!!!!!!!!  procount = ', subdir
        for onefile in os.listdir(sourcedir + subdir):
            filename = onefile
            onefile = sourcedir + subdir + onefile

            if filename.endswith(".exe"):
                 copyfile(onefile, desdir + subdirName+'_' + filename)
                 #print subdirName+'_' + filename
def checkFileAndDelete(source1 ='', source2 ='',destdir =''  ):
    files1={}
    files2 ={}

    var1 ='test_model.dot'
    var2 ='model.dot'
    #delete overlap files
    sources =[source1, source2]

    for s in sources:
        for filename in os.listdir(s):
            if filename.endswith(var1): #_test_model.dot
                othername = filename.replace(var1,var2)
                if os.path.exists(s + othername ):
                    os.remove(s + filename)
                    print s + filename

def SplitnFolds(nonvirusPath, virusPath, destPath,nFold=5):
    nonvirus_files =[]
    for filename in os.listdir(nonvirusPath):
        if filename.endswith('.dot'):
            nonvirus_files.append(filename)

    virusfiles =[]
    for filename in os.listdir(virusPath):
        if filename.endswith('.dot'):
            virusfiles.append(filename)
    # shuffle
    np.random.seed(314159)
    np.random.shuffle(nonvirus_files)
    np.random.shuffle(virusfiles)
    # split into n Fold
    step = 1.0/nFold
    bvidx =0
    bnonvidx = 0
    for fold in range(0,nFold):
        foldPath = destPath+'Fold'+str(fold+1)+'/'
        if not os.path.exists(foldPath):
            os.makedirs(foldPath)

        foldvirus = foldPath+'GCNN/'
        if not os.path.exists(foldvirus):
            os.makedirs(foldvirus)
        # copy virus to fold
        endidx = (int)((fold+1) * step * len(virusfiles))

        for idx in range(bvidx, endidx):
            copyfile(virusPath + virusfiles[idx], foldvirus+ virusfiles[idx] )
        bvidx = endidx

        foldnonvirus = foldPath + 'NonVirus/'
        if not os.path.exists(foldnonvirus):
            os.makedirs(foldnonvirus)

        endidx =(int) ((fold+1) * step * len(nonvirus_files))
        for idx in range(bnonvidx, endidx):
            copyfile(nonvirusPath + nonvirus_files[idx], foldnonvirus + nonvirus_files[idx])
        bnonvidx = endidx

    # for filename in os.listdir(source1):
    #     if filename.endswith('.dot'):
    #         idx = filename.find('.')
    #         name = filename[:idx]
    #         files1[name] = source1+ filename
    # # for name in files1:
    # #     print name,': ' ,files1[name]
    #
    #
    # for filename in os.listdir(source2):
    #     if filename.endswith('.dot'):
    #         idx = filename.find('.')
    #         name = filename[:idx]
    #         if name in files1:
    #             size1 = os.path.getsize(files1[name])
    #             size2 = os.path.getsize(source2+filename)
    #             if size1>size2:
    #                 copyfile(files1[name], destdir+ name+'.dot')
    #             else:
    #                 copyfile(source2+filename, destdir + name+'.dot')
    #         else:
    #             copyfile(source2 + filename, destdir + name+'.dot')
def splitTrainTest(sourcePath='',trainPath='', testPath ='' ):
    files=[]
    for subdir in os.listdir(sourcePath):
        if not subdir.endswith('/'):
            subdir = subdir + '/'

        for onefile in os.listdir(sourcePath + subdir):
            files.append((subdir, onefile,sourcePath+subdir+onefile))

    np.random.seed(314159)
    np.random.shuffle(files)

    numTrain = (int)(len(files)*0.8)
    # copy training
    for idx in range(0, numTrain):
        (subdir, name, fullname) = files[idx]
        copyfile(fullname, trainPath + subdir + name)

    # copy to test
    for idx in range(numTrain, len(files)):
        (subdir, name, fullname) = files[idx]
        copyfile(fullname, testPath + subdir + name)
def copyData(sourceDir, desDir, filetype='.c'):
    for subdir in os.listdir(sourceDir):
        subdir_fullpath = sourceDir+subdir
        if os.path.isfile(subdir_fullpath): # file
            continue
        subdir_fullpath +='/'

        subdir_name = subdir.replace('/','')
        for onefile in os.listdir(subdir_fullpath):
            if not onefile.endswith(filetype):
                continue
            copyfile(subdir_fullpath+onefile, desDir+ subdir_name+'_' + onefile )
def deleteEmptyFiles():
    path ='C:/Users/anhpv/Desktop/CFG/Experiment/'
    emptyInfo = path +'EmptyGraphs.txt'
    f = open(emptyInfo,'r')
    for line in f.readlines():
        line = line.strip()
        if line =='':
            continue
        onefile = path+line
        if os.path.isfile(onefile):
            os.remove(onefile)
    f.closed
if __name__ == "__main__":
    # sourceDir='D:/data/data_Ctype/'
    # desDir ='D:/data/data_exe/'
    # #
    # source2 = 'C:/Users/anhpv/Desktop/CFG/Experiment/OriginalData/GCNN/'
    # source1 ='C:/Users/anhpv/Desktop/CFG/Experiment/OriginalData/NonVirus/'
    # destdir ='C:/Users/anhpv/Desktop/CFG/Experiment/NonVirus_Merged/'
    # checkFileAndDelete(source1=source1, source2= source2, destdir= destdir)
    #
    # # copyExe(sourcedir= sourceDir, desdir= desDir)
    # sourcePath = 'C:/Users/anhpv/Desktop/CFG/Experiment/OriginalData/'
    # trainPath = 'C:/Users/anhpv/Desktop/CFG/Experiment/Training/'
    # testPath = 'C:/Users/anhpv/Desktop/CFG/Experiment/Testing/'
    # splitTrainTest(sourcePath= sourcePath, trainPath=trainPath, testPath=testPath)

    # # program processing
    # sourceDir ='D:/data/data_Ctype/'
    # desDir ='D:/data_Ctype/'
    # copyData(sourceDir= sourceDir, desDir= desDir)

    #split data

    # source1 ='C:/Users/anhpv/Desktop/CFG/Experiment/OriginalData/NonVirus/'
    # source2 = 'C:/Users/anhpv/Desktop/CFG/Experiment/OriginalData/GCNN/'
    # destPath ='C:/Users/anhpv/Desktop/CFG/Experiment/5Folds/'
    # SplitnFolds(nonvirusPath=source1, virusPath=source2,destPath=destPath)

    deleteEmptyFiles()