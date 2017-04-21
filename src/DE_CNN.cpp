#include <stdio.h>
#include <string.h>
#include "DESolver.h"
#include <algorithm>
#include "global.h"
#include "FFNN.h"
#include"predict.h"

#define N_POP 100
#define MAX_GENERATIONS	20

#define BConvWid 0//3600 // begin index of weights of convolution layer
#define BConvBias 0//1290// begin index of biases of convolution layer

class Pair
{
	public:
		int Id;
		float Value;
		Pair(){}
		Pair(int id, float value)
		{
			this->Id = id;
			this->Value = value;
		}
		bool operator<(Pair const & b)
		{
			return this->Value < b.Value;
		}

};
// DE_CNN fitting problem
class DE_CNNSolver : public DESolver
{
public:
	DE_CNNSolver(int dim,int pop) : DESolver(dim,pop), count(0) {;}
	float EnergyFunction(float trial[],bool &bAtSolution);

private:
	int count;
};
void writedata(char*file, float*data, int length)
{
	printf("writing data");
	FILE *fp;
    
    fp=fopen(file,"w");
    if(fp==NULL)
	{
		printf("can't create a file for writing\n");
		exit(1);
	}
    
    for(int i=0; i<length; i++)
    	fprintf(fp,"\n%lf", data[i]);
    fclose(fp);
}
float DE_CNNSolver::EnergyFunction(float *trial,bool &bAtSolution)
{
	// decode copy gene to weights, biases
	memcpy(weights+BConvWid,trial, (num_weights-BConvWid)*sizeof(float));
	memcpy(biases+BConvBias, trial+(num_weights-BConvWid), (num_biases-BConvBias)*sizeof(float));
	
	float acc = predictCV(y_CV, num_CV, NULL, NULL,0);
	if (count++ % nPop == 0)
		printf("\n%d %lf\n",count,Energy());
	//float acc =0;	
	//for(int i=0; i<nDim-10; i++)
	//	acc +=trial[i];
	return(-acc);
}
