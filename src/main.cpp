/*
 * main.cpp
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

//#include "global.h"
#include "global.h"
#include <string>
#include "FFNN.h"
#include <cmath>
#include"read_data.h"
#include"predict.h"
#include <stdlib.h>
#include<math.h>
#include <stdio.h>
#include"activation.h"
#include"loss.h"
#include <malloc.h>
#include <ctime>
#include "file_io.cpp"
using namespace std;

//void saveParam(int epoch);
extern char * f_train;
extern char * f_CV;
extern char * f_test;
extern char * f_yCV;
extern char * f_ytest;
extern char * f_ytrain;
extern char * buf_train;
extern char * buf_CV;
extern char * buf_test;
extern char * database;
extern char * settingfile;
int m_nEpoch;
float m_alpha;
float m_CVacc =0.0;
float * m_weights, * m_biases; // weights and biases at max accuracy of CV
//void (* dloss)( int ytrue, float * ypred, float* dEdz, int n);

Layer ** global;
int main(int argc, char* argv[]){
	if (argc>1)
		settingfile = argv[1];
	else
	{
		settingfile = (char*)malloc(20*sizeof(char)); 
		strcpy(settingfile,"settings.txt");
	}
	
	// read parameters from setting file
	var* varlist = getparameters();
	if(varlist==NULL)
	{
		cout<<"Can not read initial parameters from file: "<<settingfile<<endl;
		return 0;
	}
	
	float learn_rate;
	float K;
	//eta = 1.0;
//	float C_weights_0;
	batch_size = atoi(getvarvalue(varlist, "batch"));
	int n_miniGD = 0;
	bool n_miniGDchange = false;
//	int tmp1 = 0;

	alpha = atof(getvarvalue(varlist, "alpha")); //0.1
	float beta = atof(getvarvalue(varlist, "beta")); //0.7
	
	C_weights = 0.000001;
	

	//beta = 0;
	p_dropout1 = 0.5;
	p_dropout2 = 0;

	learn_rate = alpha / (1 + beta * n_miniGD);
	K = 0.0001;

	beta =  (1.0 / beta - 1.0) / 3000.0;
	eta = learn_rate * sqrt(K);

	C_weights /= learn_rate;

	K *= (batch_size * batch_size );

	cout << "\nalpha = " << alpha<<"; beta = "<<beta;
	cout << "\nbatch = " << batch_size << "; C_weights = " <<
			C_weights << "; learn_rate = " << learn_rate << endl;
	//"; P(c/h dropout) = " << p_dropout1 <<
		//	"; P(a/ae dropout) = " << p_dropout2<<endl;

	cout << flush;

	int n_sample = 0;
	float train_correct = 0;
    // some common parameters
	num_train =atoi(getvarvalue(varlist,"num_train"));// 31200;
	num_CV = atoi(getvarvalue(varlist,"num_cv")); //10400;
	num_test = atoi(getvarvalue(varlist,"num_test")); //10400;
	num_label = atoi(getvarvalue(varlist,"output"));//104
	printf("\nnum train = %d, num CV = %d, num test = %d", num_train, num_CV, num_test);
	printf("\nnum out = %d", num_label);
	
	int n_train = 1;

	char * fp = getvarvalue(varlist,"param_file"); // file store: embedding + weights + biases
	
	ReadParam(fp);


	//ReadIndex("samplepath");
	m_weights = (float*)malloc(num_weights*sizeof(float));
	m_biases = (float*)malloc(num_biases*sizeof(float));
	
	for(int i = 0; i< num_weights; i++){
		Wcoef_AdaGrad[i] = K;
	}
	for(int i = 0; i < num_biases; i++){
		Bcoef_AdaGrad[i] = K;
	}
	cout<<"Begin Read"<<endl;
 	
 	// data files
 	f_train = (char*) malloc(sizeof(char) * 300) ;
    snprintf( f_train, 300, "%s",getvarvalue(varlist,"x_train"));
    
	f_CV = (char*) malloc(sizeof(char) * 300) ;
	snprintf( f_CV , 300, "%s",getvarvalue(varlist,"x_cv"));
	
	f_test = (char*) malloc(sizeof(char) * 300) ;
	snprintf( f_test , 300, "%s",getvarvalue(varlist,"x_test"));
	
	f_ytrain = (char*) malloc(sizeof(char) * 300) ;
	snprintf( f_ytrain , 300, "%s",getvarvalue(varlist,"y_train"));
	
	f_yCV = (char*) malloc(sizeof(char) * 300) ;
	snprintf( f_yCV , 300, "%s", getvarvalue(varlist,"y_CV"));
	
	f_ytest = (char*) malloc(sizeof(char) * 300) ;
	snprintf( f_ytest , 300, "%s",getvarvalue(varlist,"y_test"));
	
	database = (char*) malloc(sizeof(char) * 20);
	snprintf( database , 20, "%s",getvarvalue(varlist,"database"));
	printf("\ndatabase: %s\n",database);
	
	getActiveFunction(getvarvalue(varlist,"act_func"));
	getdLoss(getvarvalue(varlist,"loss"));
	
	ReadAllData();
	cout << "INFO: Data loaded" << endl;
	// check x train, cv, test
//	getNetInfor(buf_train, num_train, "trainnet_info.txt");
//	getNetInfor(buf_CV, num_CV, "cvnet_info.txt");
//	getNetInfor(buf_test, num_test, "testnet_info.txt");

	isTraining = true;
	srand(314159);
	//ReadTrainNetwork(6);
	//ReadTrainNetwork(7);
	cout<<num_weights<<endl;
	
	int tobegin = atoi(getvarvalue(varlist,"begin"));
	char* pretrained_file = getvarvalue(varlist,"pretrained_file");  
	if(strcmp(pretrained_file,"none") == 0)
	{
		cout<<"randomly initialize weights and biases"<<endl;
	}
	else
	{
		cout<<"read pretrained weights and biases from: '"<<pretrained_file<<"'"<<endl;
//	    readTBCNNParam2(tobegin,alpha,n_miniGDchange);
		ReadNetParams(pretrained_file);
	    n_miniGD = tobegin*num_train/batch_size;
	}
	//write :
	//-0.0170019
	//0.617616
	// write 
	int nEpoch = atoi(getvarvalue(varlist,"epoch_num"));
	
	printf("\nBegin = %d, Epoch =%d\n", tobegin,nEpoch);

	
	bool markCV= 1;	// use CV or train to valid the model
	if (num_CV<=10)
	{
		printf("\nnum_CV <10, save model at maximum of training\n");
		markCV = 0;
	}
		
	int t_start=clock();
	
	char * phase = getvarvalue(varlist,"phase");
	char * model_file= getvarvalue(varlist,"model_file");
	if (strcmp(phase,"test") ==0) // test mode
	{
		cout<<"\n\nread pretrained weights and biases from: '"<<model_file<<"'"<<endl;
		ReadNetParams(model_file);
		cout<<"testing and writing results..."<<endl;
		// write to file
		FILE *f_vectest, *f_veccv, *f_vectrain,*f_probcv, *f_probtest;
		char cv_vec[300];
        snprintf( cv_vec, 300, "%s_vec_cv_r%d.txt",database,tobegin);
        
		char test_vec[300];
        snprintf( test_vec, 300, "%s_vec_test_r%d.txt",database,tobegin);
        
        char train_vec[300];
        snprintf( train_vec, 300, "%s_vec_train_r%d.txt",database,tobegin);
        
		char cv_prob[300];
        snprintf( cv_prob, 300, "%s_prob_cv_r%d.txt",database,tobegin);

		char test_prob[300];
        snprintf( test_prob, 300, "%s_prob_test_r%d.txt",database,tobegin);
        
		f_vectest = fopen(test_vec, "w");
		f_veccv = fopen(cv_vec, "w");
		f_vectrain = fopen(train_vec, "w");
		f_probcv = fopen(cv_prob, "w");
		f_probtest = fopen(test_prob, "w");
		

		predictCV(y_CV, num_CV, f_veccv, f_probcv);
		predictTest(y_test, num_test, f_vectest, f_probtest);
		predictTrain(y_train, num_train, f_vectrain);
		
		fclose(f_vectest);
		fclose(f_veccv);
		fclose(f_vectrain);
		fclose(f_probcv);
		fclose(f_probtest);
		return 0;
	}
	cout<<"training..."<<endl;
	for (int epoch = tobegin; epoch <= nEpoch; ++ epoch) {
		//break;
		float J = 0;
		int avglen = 0;



		//if(epoch%10 == 0)saveTBCNNParam2(epoch,alpha,n_miniGDchange);
		//readTBCNNParam(epoch);
		//break;
		for (int i = 0;i < num_train; ++i) {
			//cout<<"aaaaaaaaaaaaaaaaaaaaa"<<endl;
			//if(i==6)
				//continue;
			//if(i%10000 == 0)cout<<" i = "<<i<<endl;
			//cout<<" i = "<<i<<endl;
			ReadTrainNetwork(i,i%2);
			//cout<<"makeit!!!!!!!!!!!"<<endl;
			//cout<<"test and cv begin"<<endl;

			if (i == num_train-1) {
				cout << "Epoch = "<< epoch<<" i = "<<i;
				
				float CVacc = predictCV(y_CV, num_CV, NULL, NULL);
				predictTest(y_test, num_test, NULL,NULL);
				
				cout <<  "  trainerror : " <<  J / n_train<<"  train accuracy : "<<train_correct/n_train<< endl;
				n_train = 0;J = 0;
				train_correct = 0;
				
				if (markCV==0) // no CV data
					CVacc = train_correct; 
				if (CVacc>=m_CVacc) // save weights, biases at the current max point
				{
					m_CVacc = CVacc;
					m_nEpoch = epoch;
					m_alpha = alpha;
					memcpy(m_weights,weights, num_weights*sizeof(float));
					memcpy(m_biases,biases, num_biases*sizeof(float));	
				}
			}

			//cout<<"test and cv finished"<<endl;
			++ n_sample;
			++ n_train;
			//cout<<"bbbbbbbbbbbbbbbbbb i = "<<i<<endl;
			Layer ** Xnet;
			int len ;
//			if(i%2 ==1)
			{
				Xnet = X_train[0];
				len = len_X_train[0];
			}
//			else
//			{
//				Xnet = X_trainB[0];
//				len = len_X_trainB[0];
//
//			}

			int t = y_train[i];
			//cout<<"t = "<<t<<endl;


			//-----------------train_CV
			//ReadCVNetwork(i);
			//Xnet = X_CV[0];
			//len = len_X_CV[0];
	        //t = y_CV[i];
			//----------------end
			//cout<<"beginFeed"<<endl;
			FeedForward(Xnet, len);
			//cout<<"endFeed"<<endl;
			int lastidx = len - 1;

			float * h = Xnet[ lastidx ] -> y;

			J -= log(h[t]);
			//cout<<"J : "<<J<<endl;



			avglen+=len;
			//if (Xnet[ lastidx ]->dE_dz[2] == NULL)
				//cout<<"NULL!!!!!!!!"<<endl;
			//cout<<Xnet[ lastidx ]->dE_dz[2]<<endl;
			//cout<<Xnet[ lastidx ]->y[0]<<endl;
			//cout<<Xnet[ lastidx ]->y[1]<<endl;
			//float yy;
			//cout<<"inmain : lastidx = "<<lastidx<<endl;
			//printf("inmain : %4f, %4f\n", Xnet[ lastidx ]->y[0],Xnet[ lastidx ]->y[1]);
			int train_predict = 0;
		    float max_pro = 0.0;	
		    
			//compute dloss( int ytrue, float * ypred, float* dEdz, int n);
			dloss(t,h,Xnet[ lastidx ]->dE_dz,num_label);
				
			for(int x = 0; x < num_label; x++){
				//yy = ;
				if(h[x]>max_pro){
				    train_predict = x;
				    max_pro = h[x];
				}
//				Xnet[ lastidx ]->dE_dz[x] = Xnet[ lastidx ]->y[x];
				
				if(!(Xnet[ lastidx ]->dE_dz[x]>=0 || Xnet[ lastidx ]->dE_dz[x]<=0))
					cout<<"@@@@@@@@@@@@@@@@Xnet[ lastidx ]->dE_dz is nan at x = "<<x<<endl;
				//cout<<"!!  "<<Xnet[ lastidx ]->dE_dz[x]<<endl;;
			}
			if(train_predict == t){
			    train_correct += 1;
			}
			//cout<<"$$$$$$$$$$$$$$$$$$$$$Xnet[ lastidx ]->dE_dz[76] = "<<Xnet[ lastidx ]->dE_dz[76]<<" and Xnet[ lastidx ]->name = "<<Xnet[ lastidx ]->name<<endl;
//			Xnet[ lastidx ] -> dE_dz[ t ] -= 1;
			//cout<<"beginBack"<<endl;
			CleanDerivative(Xnet, len);
			//cout<<"begin BackPropagation i =  "<<i<<endl;

			BackPropagation(Xnet, len);
			//cout<<"end propagetion"<<endl;


			if (n_sample % batch_size == 0){
				learn_rate = alpha / (1 + beta * n_miniGD);
				if(n_miniGDchange)++ n_miniGD;

				float weight_decay = 1 - learn_rate * C_weights;
				//float biases_decay = 1 - learn_rate * C_weights;
				cblas_sscal(num_weights, weight_decay, weights, 1);
				//cblas_sscal(num_biases, biases_decay, biases, 1);

				GradDescent(num_weights, learn_rate / batch_size , gradWeights, weights);
				GradDescent(num_biases,  learn_rate / batch_size, gradBiases, biases);

				//cout<<"ddddddddddddddd"<<endl;
				memset(gradWeights, 0, sizeof(float)*num_weights);
				memset(gradBiases,  0, sizeof(float)*num_biases);
				//cout<<"eeeeeeeeeeeeeeeeeeeeeeeeeeee"<<endl;
			}

		}

	}
	// save parameters at the max accuracy of CV
	//	
	memcpy(weights, m_weights, num_weights*sizeof(float));
	memcpy(biases,m_biases, num_biases*sizeof(float));
//	saveNetParams(m_nEpoch,m_alpha,n_miniGDchange);

    cout<<"\n Save params at epoch: " << m_nEpoch<<", to file:"<<model_file;
	SaveParam(model_file);
	
	int t_stop =clock();
	cout<<"\nrunning time: "<< t_stop - t_start;
	cout << "\ndone" << endl;

}

