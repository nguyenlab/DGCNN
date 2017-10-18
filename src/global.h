/*
 * global.h
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include<iostream>
//#include<pthread.h>
#include<string.h>

#include "cblas.h"
#include "cblas_f77.h"

extern"C"
{
    #include<cblas.h>
}
using namespace std;
////////////////////////////////////////////////////
// hyperparameters

extern float alpha;

////////////////////////////////////////////////////
// model parameters

extern float * weights, * biases;
extern float * gradWeights, *gradBiases;
extern float * Wcoef_AdaGrad, * Bcoef_AdaGrad;
extern float eta;

extern int num_weights, num_biases;
extern int num_train;
extern int num_CV;
extern int num_test;
extern int num_label;
extern float* classweight;

extern float C_weights;
extern float p_dropout1;
extern float p_dropout2;
extern bool isTraining;
extern float momentum_n;
extern int batch_size;
////////////////////////////////////////////////////
// multi-threads

//#define NUM_THREADS 4
//extern pthread_t threads[];

// helper functions


void RandomInitParam();

void ReadParam(char* filepath);
void SaveParam(const char * filepath);
void ReadNetParams(const char * filepath);
void saveNetParams(int epoch,float alpha,bool n_miniGDchange);
void readNetParams(int epoch,float alpha,bool n_miniGDchange);
#endif /* GLOBAL_H_ */
