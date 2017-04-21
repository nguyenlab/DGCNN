1. Download datasets by the following link: http://www.mediafire.com/file/c0a2n0kayxpa6kd/CodeChef.rar
2. Compile the code for running networks
	0. install CBLAS and BLAS
	1. run sh CNN.sh
	2. Copy CNNtest to the running directory
3. Construct the networks
	1. Create the a directory for running experiments (for e.g. GCNN_1V)
	2. Create a subfolder namely "xy" inside the running directory (GCNN_1V/xy)
	3. Check common parameters in GCNN/gcnn_params:
        + datapath: the path to the data directory (downloaded from above link)
	    + xypath : the path to "xy" directory
		+ experiment = 'CodeChef'
		+ numView =1 # the number of views
	4. Choose the dataset in GCNN/main_MultiChannelGCNN.py (assign problem variable to MNMX, FLOW16, SUBINC or SUMTRIAN 	
	5. Run GCNN/main_MultiChannelGCNN.py. The script constructs networks and parameters for running the experiment on this data.
	6. Run CNNtest with one parameter: settings_<datase tname>.txt. For e.g.: ./CNNtest settings_MNMX.txt

