#include"loss.h"
#include"global.h"
#include"math.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// y true, y prediction, number of labels
void (* dloss)( int ytrue, float * ypred, float* dEdz, int n);
void getdLoss(char* name)
{
	dloss = &dEdz_MeanSquareError;
	
	char * loss_name = (char*)malloc(200*sizeof(char)) ;
	strcpy(loss_name, "Mean Square Error");
	
	if (strcmp(name, "WeightedMSE")==0)
	{
		dloss = &dEdz_WeightedMeanSquareError;	
		strcpy(loss_name, "Weighted MSE");
	}
	if (strcmp(name, "CatCrossEn")==0)
	{
		dloss = &dEdz_WeightedMeanSquareError;	
		strcpy(loss_name, "Categorical Cross Entropy");
	}
	printf("\nLoss function: %s\n", loss_name);
	
}	
//K.mean(K.square(y_pred - y_true))
void dEdz_MeanSquareError(int ytrue, float * ypred, float* dEdz, int n)
{
	for (int i=0; i<n; i++)
	{
		dEdz[i] = ypred[i];
	}
	dEdz[ytrue] -= 1; 
}
//K.mean(K.square(y_pred - y_true))
void dEdz_WeightedMeanSquareError(int ytrue, float * ypred, float* dEdz, int n)
{
	for (int i=0; i<n; i++)
	{
		dEdz[i] = ypred[i]* classweight[i];
	}
	dEdz[ytrue] -= classweight[ytrue]; //1.0 * classweight[ytrue]
}
//dEdz_CategoricalCrossEntropy (-1/N* sum(  yi*log(ypred_i) + (1-yi)*log(1-ypred_i) ))
//void dEdz_CategoricalCrossEntropy(int ytrue, float * ypred, float* dEdz, int n)
//{
//	for (int i=0; i<n; i++)
//	{
//		if (i != ytrue)
//			dEdz[i] = 1/(1-ypred[i]);
//	}
//	dEdz[ytrue] = -1/ ypred[ytrue];
//}
//Categorical Cross Entropy:  -1/N* sum_by_N(sum_by_C(y_n_i*log(ypred_n_i)))
void dEdz_CategoricalCrossEntropy(int ytrue, float * ypred, float* dEdz, int n)
{
	for (int i=0; i<n; i++)
	{
			dEdz[i] = 0;
	}
	dEdz[ytrue] = -1/ ypred[ytrue];
}

