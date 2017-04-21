///*
// * main.cpp
// *
// *  Created on: Mar 7, 2015
// *      Author: mou
// */
//
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
//using namespace std;
//
////void saveParam(int epoch);
//
//Layer ** global;
//int main(int argc, char* argv[]){
//
//	float learn_rate;
//	float K;
//	//eta = 1.0;
//	float C_weights_0;
//	batch_size = 100;
//	int n_miniGD = 0;
//	bool n_miniGDchange = false;
//	int tmp1 = 0;
//
//	float beta = 0.0;
//	if (argc == 7){
//
//		batch_size = atoi(argv[1]);
//		C_weights = atof(argv[2]);
//		alpha = atof(argv[3]);
//		beta = atof(argv[4]);
//		p_dropout1 = atof(argv[5]);
//		p_dropout2 = atof(argv[6]);
//
//		beta =  (1.0 / beta - 1.0) / 3000.0;
//		learn_rate = alpha / (1 + beta * n_miniGD);
//		//p_dropout = 0.5;
//		C_weights /= learn_rate;
//		K *= (batch_size * batch_size);
//
//	}
//	else if(argc == 3){
//		alpha = atof(argv[1]);
//	        cout << "set alpha =  "<< alpha << endl;
//		tmp1 = atoi(argv[2]);
//		if(tmp1 == 0){
//		    n_miniGDchange = false;
//		    cout<<"you set learnrate don't change!" <<endl;
//		}
//		else cout<<"you set learnrate change!"<<endl;
//
//		batch_size = 100;
//		C_weights = 0.000001;
//		
//		//alpha = 0.01;
//		beta = 0.7;
//		//beta = 0;
//		p_dropout1 = 0.5;
//		p_dropout2 = 0;
//
//		learn_rate = alpha / (1 + beta * n_miniGD);
//		K = 0.0001;
//
//		beta =  (1.0 / beta - 1.0) / 3000.0;
//		eta = learn_rate * sqrt(K);
//
//		C_weights /= learn_rate;
//
//		K *= (batch_size * batch_size );
//	
//	}	    
//	else{
//		cout << "WARNING: no parameters" << endl;
//		batch_size = 100;
//		C_weights = 0.000001;
//		
//		alpha = 0.1;
//		beta = 0.7;
//		//beta = 0;
//		p_dropout1 = 0.5;
//		p_dropout2 = 0;
//
//		learn_rate = alpha / (1 + beta * n_miniGD);
//		K = 0.0001;
//
//		beta =  (1.0 / beta - 1.0) / 3000.0;
//		eta = learn_rate * sqrt(K);
//
//		C_weights /= learn_rate;
//
//		K *= (batch_size * batch_size );
//	}
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
//	int tobegin = 1;
//	if(tobegin != 1){
//	    readTBCNNParam2(tobegin,alpha,n_miniGDchange);
//	    n_miniGD = tobegin*num_train/batch_size;
//	}
//	//write :
//	//-0.0170019
//	//0.617616
//	// write 
//	
//	char* params[10] ;
//	getparameters(params);
//	int nEpoch = atoi(params[1]);
//	int pmark = atoi(params[3]);
//	int mode = atoi(params[5]);
//	printf("\nEpoch =%d\n",nEpoch);
//	printf("\nmark pos = %d\n",  pmark);
//	if (mode == 0)
//		printf("\nmode = Probabilities\n"); //:"Output labels"
//	if (mode ==1)
//		printf("\nmode = Output labels\n");
//	if (mode ==0)
//		printf("\nmode = Vector representation\n");
//		
//	printf("\nbegin test\n");	
//		
//	int t_start=clock();
//	for (int epoch = tobegin; epoch <= nEpoch; ++ epoch) {
//		//break;
//		float J = 0;
//		int avglen = 0;
//
//
//
//		//if(epoch%10 == 0)saveTBCNNParam2(epoch,alpha,n_miniGDchange);
//		//readTBCNNParam(epoch);
//		//break;
//		for (int i = 0;i < num_train; ++i) {
//			//cout<<"aaaaaaaaaaaaaaaaaaaaa"<<endl;
//			//if(i==6)
//				//continue;
//			//if(i%10000 == 0)cout<<" i = "<<i<<endl;
//			//cout<<" i = "<<i<<endl;
//			ReadTrainNetwork(i,i%2);
//			//cout<<"makeit!!!!!!!!!!!"<<endl;
//			//cout<<"test and cv begin"<<endl;
//
//			if (i == num_train-1) {
//				cout << "Epoch = "<< epoch<<" i = "<<i;
//				
//				
//				
//				if(epoch>=pmark)
//				{
//					// write to file
//					FILE *f_vectest, *f_veccv, *f_vectrain,*f_probcv, *f_probtest;
//					char cv_vec[300];
//                    snprintf( cv_vec, 300, "vec_cv_r%d.txt",epoch);
//                    
//					char test_vec[300];
//                    snprintf( test_vec, 300, "vec_test_r%d.txt",epoch);
//                    
//                    char train_vec[300];
//                    snprintf( train_vec, 300, "vec_train_r%d.txt",epoch);
//                    
//					char cv_prob[300];
//                    snprintf( cv_prob, 300, "prob_cv_r%d.txt",epoch);
//
//					char test_prob[300];
//                    snprintf( test_prob, 300, "prob_test_r%d.txt",epoch);
//                    
//					f_vectest = fopen(test_vec, "w");
//					f_veccv = fopen(cv_vec, "w");
//					f_vectrain = fopen(train_vec, "w");
//					f_probcv = fopen(cv_prob, "w");
//					f_probtest = fopen(test_prob, "w");
//					
//	
//					predictCV(y_CV, num_CV, f_veccv, f_probcv, mode);
//					predictTest(y_test, num_test, f_vectest, f_probtest,mode);
//					predictTrain(y_train, num_train, f_vectrain, mode);
//					
//					fclose(f_vectest);
//					fclose(f_veccv);
//					fclose(f_vectrain);
//					fclose(f_probcv);
//					fclose(f_probtest);
//				}
//				else
//				{
//					predictCV(y_CV, num_CV, NULL, NULL,0);
//					predictTest(y_test, num_test, NULL,NULL,0);
//				}
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
//			//cout<<"t = "<<t<<endl;
//
//
//			//-----------------train_CV
//			//ReadCVNetwork(i);
//			//Xnet = X_CV[0];
//			//len = len_X_CV[0];
//	        //t = y_CV[i];
//			//----------------end
//			//cout<<"beginFeed"<<endl;
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
//			//if (Xnet[ lastidx ]->dE_dz[2] == NULL)
//				//cout<<"NULL!!!!!!!!"<<endl;
//			//cout<<Xnet[ lastidx ]->dE_dz[2]<<endl;
//			//cout<<Xnet[ lastidx ]->y[0]<<endl;
//			//cout<<Xnet[ lastidx ]->y[1]<<endl;
//			//float yy;
//			//cout<<"inmain : lastidx = "<<lastidx<<endl;
//			//printf("inmain : %4f, %4f\n", Xnet[ lastidx ]->y[0],Xnet[ lastidx ]->y[1]);
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
//				//memset(gradWeights, 0, sizeof(float) * 180000);
//				//memset(gradBiases,  0, sizeof(float) * 62817900);
//				/*
//				for (int ii = 0;ii<1544;ii++)
//				{
//					cout<<gradWeights[ii]<<endl;
//				}
//				cout<<endl;
//				*/
//				//return 0;
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
//		//learn_rate *= 0.6;
////		char des[50] = "param_pretrain";
////		des[14] = '0' + epoch;
////		des[15] = '\0';
////		SaveParam( des );
//
//	}
//	int t_stop =clock();
//	cout<<"\nrunning time: "<< t_stop - t_start;
//	cout << "\ndone" << endl;
//
//}
//

