1. We extend the source code for constructing and running networks of Mou et al. (https://sites.google.com/site/treebasedcnn/)
To run TBCNN or SibStCNN	
	
	parameters in gl.py
	0. install CBLAS and BLAS
	1. run TBCNN/03-ConstructCandW/main_TBCNN.py(or main_TBCNN_Sib.py) to construct networks
	2. run TBCNN/xy/Shuffle.py to combine and shuffle
	3. run TBCNN/04-train/train_TBCNN.py(or train_TBCNN_Sib.py) to create paramTest
	4. make sure *fp in TBCNN/src/main.cpp is right
	5. make sure f_train, f_CV, f_test, … six pathes in TBCNN/src/read_data.cpp are right.
	6. run sh TBCNN.sh
	7. make sure the parameters in setting.txt file (num_train,	num_CV, num_test, output) are right 
	8. run TBCNNtest
 
2. kNN_TED
	+ use pycparser to parse program into ASTs and save by the format as follows: http://evolution.genetics.washington.edu/phylip/newicktree.html
	+ The implementation of kNN with TED. To run this program, copy the training and test data into directory "data"
	+ Input parameters in file "args.txt" including k, the filenames of training data,test data, and the output data
3. WekaExtension 
	- The extension of some algorithms in Weka
	- To run this program:
		 + Copy the training and test data into directory "data"
		 + set parameters including: training file, test file, output data file.
		 + The classifier parameters according the format of Weka
		 + The name of the classifier
4. TBCNN_kNN-TED
	+ Run TBCNN and set mode in seting.txt file is 0 to obtain values of the output layers
	+ run kNN_TED (set the number of nearest neighbors)
	+ get output of TBCNN and kNN_TED to turn the combination parameter
5. TBCNN + SVM
	+ Run TBCNN and set mode in seting.txt file is 0 to obtain vector representation
	+ Convert data into Weka format
	+ put the datasets in WekaExtension/data
	+ set parameters in WekaExtension/data/args
	+ run WekaExtension/ml_algorithm.jar
6. AST datasets and the output data
	AST datasets and the output data are share at https://drive.google.com/folderview?id=0B0rkFuvCgmwgdlhuYm5Hc2UySmM&usp=sharing
	 The data for runing models and output data corresponding to four scenarios (original ASTs, ASTs pruned by minor procedures, ASTs pruned by semantic methods, and ASTs pruned by two methods) are stored in folders AST_OR, AST_MF, AST_SE, AST_SE_MF, respectively.
