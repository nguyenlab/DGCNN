0.Introduction
  The project include two parts:
	construct_nn		preprocess the data set 
	FFNN			train the Tree-based convolution neural network

1.Preprocess
  Generate networks for each sentence pair.
	python construct_nn/Paraphrase/process.py
  Mix these networks into one file.
	python construct_nn/Paraphrase/mix.py
  Note that if you want to change the model, please change the line 3 in construct_nn/Paraphrase/process.py, which is:
	import constructTBCNN_pair_difference_product as Construct

  	we provide four models including:
		constructTBCNN_pair_difference_product
		constructTBCNN_pair_difference
		constructTBCNN_pair_product
		constructTBCNN_pair

2.Training
  Start training the model using FFNN/FFNN. First please compile the whole project.
	sh FFNN/FFNN.sh
  Then you can run the FFNN.
	./FFNN/FFNN
  Note that if you want to train different network, please change the input file name, including:
	line 64,66 in  FFNN/src/main.cpp
	line 168,169,170 in  FFNN/src/read_data.cpp
