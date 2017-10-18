//#include "global.h"
//#include <string>
//#include "FFNN.h"
//#include <cmath>
//#include"read_data.h"
//#include"predict.h"
//#include <stdlib.h>
//#include<math.h>
//#include <stdio.h>
//#include"activation.h"
//#include <malloc.h>
//#include <ctime>
//#include "file_io.cpp"
//#include "DESolver.cpp"
//#include "DE_CNN.cpp"
//using namespace std;
//
//// net parameters
//int tobegin = 1;
//int nEpoch;
//float learn_rate;
//int n_miniGD = 0;
//bool n_miniGDchange = false;
//float beta = 0.0;
//
//Layer ** global;
//
//void InitNetParams();
//void SortPairs(Pair p[], int N);//number of instances
//void Run_DE_CNN(int N_DIM, float *objs, float * pop);
//void TrainNet(int tobegin, int nEpoch);
//int main(int argc, char* argv[])
//{
//	// set POPSIZE and Generations in DE_CNN file
//	InitNetParams();
//	
//	float *objs, *pop;
//	int N_DIM = num_weights + num_biases - BConvWid - BConvBias;
//	printf("\nDIM = %d (weights:%d,  biases:%d) - from conv layer\n",N_DIM, num_weights-BConvWid,num_biases-BConvBias);
//	
//	objs = (float*)malloc(N_POP*sizeof(float));
//	pop = (float*)malloc(N_POP*N_DIM*sizeof(float));
//	Run_DE_CNN(N_DIM, objs, pop);
//	
//	printf("\n\nBest best ind:\n");
//	// get k-best
//	int K = 5;
//	
//	Pair energyId_Val[N_POP];
//	int i;
//	for(i=0; i<N_POP; i++)
//	{
//		energyId_Val[i] = Pair(i, objs[i]);
//		printf("\nobj: %lf", objs[i]);
//	}
//	SortPairs(energyId_Val,N_POP); //std::sort(energyId_Val, energyId_Val+N_POP);
//	
//	float *ind;
//	for(i=0; i<K; i++) // get k
//	{	
//		int id = energyId_Val[i].Id;
//		printf("\n\nGet %d - %d:\n",i, id);
//		
//		printf("\n%d %lf",id, energyId_Val[i].Value);
//		
//		ind = pop + id * N_DIM; // poit to individual id 
//		// copy values of ind id ---> weights & bias
//		memcpy(weights+BConvWid,ind, (num_weights-BConvWid)*sizeof(float));
//		memcpy(biases+BConvBias, ind+(num_weights-BConvWid), (num_biases-BConvBias)*sizeof(float));
//	
//		// continue train net
//		printf("\nTrain net, weights:%lf %lf\n", weights[0],weights[1]);
//		TrainNet(tobegin, nEpoch);
//	}
//
//    //float *pop = solver.Pop_Contents();	
//    //for (i=0;i<N_DIM && i<10;i++)
//	//	printf("[%d]: %lf\n",i,pop[maxid*N_DIM +i]);
//	
//	
//	
//}
//void SortPairs(Pair p[], int N)
//{
//	int i, j;
//  for ( i = 0 ; i < ( N - 1 ) ; i++ )
//   {
//      int position = i;
// 
//      for ( j = i + 1 ; j < N ; j++ )
//      {
//         if ( p[position].Value > p[j].Value )
//            position = j;
//      }
//      if ( position != i )
//      {
//         Pair swap = p[i];
//         p[i] = p[position];
//         p[position] = swap;
//      }
//   }
//}
//void Run_DE_CNN(int N_DIM, float *objs, float * pop)
//{
//	//int N_DIM = num_weights + num_biases;
//	//printf("\nDIM = %d (weights:%d,  biases:%d\n)",N_DIM, num_weights,num_biases);
//	float min[N_DIM];
//	float max[N_DIM];
//	int i;
//
//	DE_CNNSolver solver(N_DIM,N_POP);
//
//	for (i=0;i<N_DIM;i++)
//	{
//		max[i] =  100.0;
//		min[i] = -100.0;
//	}
//	//stBest1Exp			0
//	//stRand1Exp			1
//	//stRandToBest1Exp	2
//	//stBest2Exp			3
//	//stRand2Exp			4
//	//stBest1Bin			5
//	//stRand1Bin			6
//	//stRandToBest1Bin	7
//	//stBest2Bin			8
//	//stRand2Bin			9
//	solver.Setup(min,max,stBest1Exp,0.9,1.0);
//	
//	printf("Calculating...\n\n");
//	solver.Solve(MAX_GENERATIONS);
//		
//	float *solution = solver.Solution();
//
//	// list of fitness values
//	float *p;
//	
//	p = solver.PopEnergy();
//	memcpy(objs,p,N_POP*sizeof(float));
//	// population (nPop * nDim)
//	p = solver.Pop_Contents();
//	memcpy(pop, p, N_POP*N_DIM*sizeof(float));	
//}
//
//void InitNetParams()
//{
//
//	learn_rate;
//	float K;
//	//eta = 1.0;
//	batch_size = 100;
//	n_miniGD = 0;
//	n_miniGDchange = false;
//	int tmp1 = 0;
//
//	beta = 0.0;
//	
//	cout << "WARNING: no parameters" << endl;
//	batch_size = 100;
//	C_weights = 0.000001;
//	
//	alpha = 0.1;
//	beta = 0.7;
//	//beta = 0;
//	p_dropout1 = 0.5;
//	p_dropout2 = 0;
//
//	learn_rate = alpha / (1 + beta * n_miniGD);
//	K = 0.0001;
//
//	beta =  (1.0 / beta - 1.0) / 3000.0;
//	eta = learn_rate * sqrt(K);
//
//	C_weights /= learn_rate;
//
//	K *= (batch_size * batch_size );
//
//
//
//	cout << "batch = " << batch_size << "; C_weights = " <<
//			C_weights << "; learn_rate = " << learn_rate << endl;
//	//"; P(c/h dropout) = " << p_dropout1 <<
//		//	"; P(a/ae dropout) = " << p_dropout2<<endl;
//
//	cout << flush;
//
//	int n_sample = 0;
//	float train_correct = 0;
//
//	num_train = 31200;
//	num_CV = 10400;
//	num_test = 10400;
//	int n_train = 1;
//	//cout<<"good!!"<<endl;
//	const char * fp = "paramTest";
//	ReadParam(fp);
//
//
//	//ReadIndex("samplepath");
//
//	for(int i = 0; i< num_weights; i++){
//		Wcoef_AdaGrad[i] = K;
//	}
//	for(int i = 0; i < num_biases; i++){
//		Bcoef_AdaGrad[i] = K;
//	}
//	cout<<"Begin Read"<<endl;
//
//	ReadAllData();
//	cout << "INFO: Data loaded" << endl;
//
//
//
//	isTraining = true;
//	srand(314159);
//	//ReadTrainNetwork(6);
//	//ReadTrainNetwork(7);
//	cout<<num_weights<<endl;
//	
//	if(tobegin != 1){
//	    readTBCNNParam2(tobegin,alpha,n_miniGDchange);
//	    n_miniGD = tobegin*num_train/batch_size;
//	}
//	
//	char* params[10] ;
//	getparameters(params);
//	
//	nEpoch = atoi(params[1]);
//	printf("\nBegin = %d, Epoch =%d\n",tobegin, nEpoch);
//	/*int pmark = atoi(params[3]);
//	int mode = atoi(params[5]);
//	printf("\nEpoch =%d\n",nEpoch);
//	printf("\nmark pos = %d\n",  pmark);
//	if (mode == 0)
//		printf("\nmode = Probabilities\n"); //:"Output labels"
//	if (mode ==1)
//		printf("\nmode = Output labels\n");
//	if (mode ==0)
//		printf("\nmode = Vector representation\n");*/
//		
//}
//void TrainNet(int tobegin, int nEpoch)
//{
//	int n_train = 1;
//	int n_sample = 0;
//	float train_correct = 0;
//	
//	int t_start=clock();
//	for (int epoch = tobegin; epoch <= nEpoch; ++ epoch) {
//		//break;
//		float J = 0;
//		int avglen = 0;
//
//		for (int i = 0;i < num_train; ++i) {
//			ReadTrainNetwork(i,i%2);
//			
//			if (i == num_train-1) {
//				cout << "Epoch = "<< epoch<<" i = "<<i;
//				
//				predictCV(y_CV, num_CV, NULL, NULL,0);
//				predictTest(y_test, num_test, NULL,NULL,0);
//					
//				cout <<  "  trainerror : " <<  J / n_train<<"  train accuracy : "<<train_correct/n_train<< endl;
//				n_train = 0;J = 0;
//				train_correct = 0;
//
//			}
//
//			//cout<<"test and cv finished"<<endl;
//			++ n_sample;
//			++ n_train;
//			//cout<<"bbbbbbbbbbbbbbbbbb i = "<<i<<endl;
//			Layer ** Xnet;
//			int len ;
//			if(i%2 ==1)
//			{
//				Xnet = X_train[0];
//				len = len_X_train[0];
//			}
//			else
//			{
//				Xnet = X_trainB[0];
//				len = len_X_trainB[0];
//
//			}
//
//			int t = y_train[i];
//			
//			FeedForward(Xnet, len);
//			//cout<<"endFeed"<<endl;
//			int lastidx = len - 1;
//
//			float * h = Xnet[ lastidx ] -> y;
//
//			J -= log(h[t]);
//			//cout<<"J : "<<J<<endl;
//
//
//
//			avglen+=len;
//		
//			int train_predict = 0;
//		        float max_pro = 0.0;	
//			for(int x = 0; x < num_label; x++){
//				//yy = ;
//				if(h[x]>max_pro){
//				    train_predict = x;
//				    max_pro = h[x];
//				}
//				Xnet[ lastidx ]->dE_dz[x] = Xnet[ lastidx ]->y[x];
//				if(!(Xnet[ lastidx ]->dE_dz[x]>=0 || Xnet[ lastidx ]->dE_dz[x]<=0))
//					cout<<"@@@@@@@@@@@@@@@@Xnet[ lastidx ]->dE_dz is nan at x = "<<x<<endl;
//				//cout<<"!!  "<<Xnet[ lastidx ]->dE_dz[x]<<endl;;
//			}
//			if(train_predict == t){
//			    train_correct += 1;
//			}
//			//cout<<"$$$$$$$$$$$$$$$$$$$$$Xnet[ lastidx ]->dE_dz[76] = "<<Xnet[ lastidx ]->dE_dz[76]<<" and Xnet[ lastidx ]->name = "<<Xnet[ lastidx ]->name<<endl;
//			Xnet[ lastidx ] -> dE_dz[ t ] -= 1;
//			//cout<<"beginBack"<<endl;
//			CleanDerivative(Xnet, len);
//			//cout<<"begin BackPropagation i =  "<<i<<endl;
//
//			BackPropagation(Xnet, len);
//			//cout<<"end propagetion"<<endl;
//
//
//
//
//
//
//			if (n_sample % batch_size == 0){
//				learn_rate = alpha / (1 + beta * n_miniGD);
//				if(n_miniGDchange)++ n_miniGD;
//
//				float weight_decay = 1 - learn_rate * C_weights;
//				//float biases_decay = 1 - learn_rate * C_weights;
//				cblas_sscal(num_weights, weight_decay, weights, 1);
//				//cblas_sscal(num_biases, biases_decay, biases, 1);
//
//
//				GradDescent(num_weights, learn_rate / batch_size , gradWeights, weights);
//				GradDescent(num_biases,  learn_rate / batch_size, gradBiases, biases);
//
//				//cout<<"ddddddddddddddd"<<endl;
//				memset(gradWeights, 0, sizeof(float)*num_weights);
//				memset(gradBiases,  0, sizeof(float)*num_biases);
//				//cout<<"eeeeeeeeeeeeeeeeeeeeeeeeeeee"<<endl;
//			}
//
//		}
//	}
//	int t_stop =clock();
//	cout<<"\nrunning time: "<< t_stop - t_start;
//	cout << "\ndone" << endl;
//}
