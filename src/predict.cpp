#include "global.h"
#include <iostream>
#include "FFNN.h"
#include <cmath>
#include <stdio.h>
#include"read_data.h"

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

int modeProb = 0;
int modeLabel = 1;
int modeVec = 0;
using namespace std;
// predict is out of time
/*void predict(Layer *** Xtest, int * ytest, int test_num, int * len_test) {
	isTraining = false;

	int correct = 0;
	float J = 0.0, avg = 0.0;
	for (int i = 0;i < test_num ; ++ i) {


		//cout<<i<<endl;
		int t = ytest[i];

		FeedForward(Xtest[i], len_test[i]);

		int lastidx = len_test[i] - 1;
		float * h = Xtest[i][ lastidx ] -> y;


		J -= log(h[t]);


		//for (int i = 0;i < 5;++ i)
		//	cout<<h[i]<<", ";
		avg += h[t];

		int predict = 0;
		float max_prob = 0.0;

		for(int j = 0; j < num_label; j++){
			//printf("%4f, ", h[j]);
			if (h[j] > max_prob) {
				max_prob = h[j];
				predict = j;
			}
		}
		//printf("\n");


//		if (i % 500 == 0){
//			cout<< "test case ("<< i<< "), h = "<< predict<< " (predicted "<< h[predict];
//			cout<< ") "<< t<< " (actural " << h[t]<<")"<<endl;
//		}
		if (predict == t)
			correct += 1;

	}

	//cout << "       average target output" << avg / test_num <<endl;
	//cout << "Cost " << J <<"; Acc " << (correct + 0.0) / test_num << endl;
	cout << "    " << (correct + 0.0) / test_num;
	isTraining = true;

}
*/
float predictCV(int * yCV,int num_CV, FILE * fvec, FILE * fprob, int mode){
	isTraining = false;
	
	
	int correct = 0;
	float J = 0.0, avg = 0.0;
	for (int i = 0;i < num_CV ; ++ i) {
		//cout<<i<<endl;
		int t = yCV[i];
		ReadCVNetwork(i);

		Layer ** Xnet = X_CV[0];

		int len = len_X_CV[0];
        FeedForward(Xnet, len);

		/* viet anh add
		for(int i=0; i< len; i++)
		{
			cout << "\n name of layer "<<i<< "=" << Xnet[i]-> name;
			cout << "len = " << Xnet[i]-> numUnit;
		}
		cout << "\n";*/
		//FeedForward(Xtest[i], len_test[i]);

		int lastidx = len - 1;
		float * h = Xnet[ lastidx ] -> y;


		J -= log(h[t]);


		//for (int i = 0;i < 5;++ i)
		//	cout<<h[i]<<", ";
		avg += h[t];

		int predict = 0;
		float max_prob = 0.0;

		for(int j = 0; j < num_label; j++){
			if(mode==modeProb&& fprob !=NULL) // print probalities
				fprintf(fprob,"%4f, ", h[j]);
			if (h[j] > max_prob) {
				max_prob = h[j];
				predict = j;
			}
		}
		if(mode==modeProb&& fprob !=NULL)
			fprintf(fprob,"\n");


	//		if (i % 500 == 0){
	//			cout<< "test case ("<< i<< "), h = "<< predict<< " (predicted "<< h[predict];
	//			cout<< ") "<< t<< " (actural " << h[t]<<")"<<endl;
	//		}
		if (predict == t)
			correct += 1;
		if (fvec !=NULL && mode ==modeLabel)// print labels
			fprintf(fvec,"%d\n", predict);
		if (fvec !=NULL && mode ==modeVec)// print vector representation
		{
			int hidid = len - 2;
			float * y = Xnet[hidid] -> y;
			int veclen = Xnet[hidid] -> numUnit;
			for (int v=0; v< veclen; v++)
				fprintf(fvec,"%4f ", y[v]);
			fprintf(fvec,"\n");
		}		

	}

	//cout << "       average target output" << avg / test_num <<endl;
	//cout << "Cost " << J <<"; Acc " << (correct + 0.0) / test_num << endl;
	float acc = (correct + 0.0) / num_CV;
	cout << "  cv-correct  " << acc;
	isTraining = true;
	
	return acc;
}

// mode =0, write detail probabilities 
// mode =1, write predicting label
//int modeProb =0;
//int modeLabel = 1;
void predictTest(int * yTest,int num_test, FILE * fvec, FILE * fprob, int mode){
	isTraining = false;

	int correct = 0;
	float J = 0.0, avg = 0.0;
	for (int i = 0;i < num_test ; ++ i) {


		//cout<<i<<endl;
		int t = yTest[i];
		ReadTestNetwork(i);

		Layer ** Xnet = X_test[0];

		int len = len_X_test[0];
        FeedForward(Xnet, len);




		//FeedForward(Xtest[i], len_test[i]);

		int lastidx = len - 1;
		float * h = Xnet[ lastidx ] -> y;


		J -= log(h[t]);


		//for (int i = 0;i < 5;++ i)
		//	cout<<h[i]<<", ";
		avg += h[t];

		int predict = 0;
		float max_prob = 0.0;

		for(int j = 0; j < num_label; j++){
			
			if(mode==modeProb&& fprob !=NULL) // print probalities
				fprintf(fprob,"%4f, ", h[j]);
			if (h[j] > max_prob) {
				max_prob = h[j];
				predict = j;
			}
		}
		if(mode==modeProb&& fprob !=NULL)
			fprintf(fprob,"\n");


	//		if (i % 500 == 0){
	//			cout<< "test case ("<< i<< "), h = "<< predict<< " (predicted "<< h[predict];
	//			cout<< ") "<< t<< " (actural " << h[t]<<")"<<endl;
	//		}
		if (predict == t)
			correct += 1;
		if (fvec !=NULL && mode ==modeLabel) // print labels
			fprintf(fvec,"%d\n", predict);
		if (fvec !=NULL && mode ==modeVec)// print vector representation
		{
			int hidid = len - 2;
			float * y = Xnet[hidid] -> y;
			int veclen = Xnet[hidid] -> numUnit;
			for (int v=0; v< veclen; v++)
				fprintf(fvec,"%4f ", y[v]);
			fprintf(fvec,"\n");
		}	
	}

	//cout << "       average target output" << avg / test_num <<endl;
	//cout << "Cost " << J <<"; Acc " << (correct + 0.0) / test_num << endl;
	cout << "  test-correct    " << (correct + 0.0) / num_test;
	isTraining = true;
}
void predictTrain(int * yTrain,int num_Train, FILE * fvec, int mode){
	if (mode !=modeVec) return;
	for (int i = 0;i < num_Train ; ++ i) {
	
			ReadTrainNetwork(i,i%2);
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
//			}
//			
			FeedForward(Xnet, len);

			if (fvec !=NULL && mode ==modeVec)// print vector representation
			{
				int hidid = len - 2;
				float * y = Xnet[hidid] -> y;
				int veclen = Xnet[hidid] -> numUnit;
				for (int v=0; v< veclen; v++)
					fprintf(fvec,"%4f ", y[v]);
				fprintf(fvec,"\n");
			}		

	}
}
