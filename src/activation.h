/*
 * activation.h
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

#ifndef ACTIVATION_H_
#define ACTIVATION_H_

void ReLU(float * z, float * y, int n);
void ReLUPrime(float * y, float * dy_dz, int n);

void Softmax(float * z, float * y, int n);
void tanh(float * z, float * y, int n);
void tanhPrime(float *y, float * dy_dz, int n);
void dummy(float * y, float * dy_dz, int n);



#endif /* ACTIVATION_H_ */
