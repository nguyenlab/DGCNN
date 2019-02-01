0. The Project includes two parts:
	- GCNN/main_MultiChannelGCNN.py		generate the computation for CFG of each program
	- CNN written in C for training the neural network. Compiler the CNN:
		+ install CBLAS and BLAS
		+ run sh CNN.sh
1. Prepare data
	- Generate CFG data for training, validation and testing.
		+ run ASMCFG/ProcessData.py
		+ make sure some parameters: "data_dir" - the directory containing assembly files, "dest_dir" - the directory for storing the CFG data, 'problems'- names of the datasets
2. Generate computations for CFGs  
	- the common parameters for networks are in "gcnn_params.py"
	- run GCNN/main_MultiChannelGCNN.py
3. Training the network
	8. run CNN setting.txt
	
Please cite our paper entiles "DGCNN: A convolutional neural network over large-scale labeled graphs"  published in Neural Network 2018, if you used in your research.


