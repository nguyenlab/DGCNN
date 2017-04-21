/*
 * predict.h
 *
 *  Created on: Mar 13, 2015
 *      Author: mou
 */

#ifndef PREDICT_H_
#define PREDICT_H_

//void predict(Layer *** X_test, int * y_test, int test_num, int * len_test);
void predictTest(int * yTest,int num_test,  FILE * fvec, FILE * fprob, int mode);
float predictCV(int * yCV,int num_CV, FILE * fvec, FILE * fprob, int mode);
void predictTrain(int * yTrain,int num_Train, FILE * fvec, int mode);

#endif /* PREDICT_H_ */
