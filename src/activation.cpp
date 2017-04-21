/*
 * activation.cpp
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */


#include"global.h"
#include"activation.h"
#include"math.h"

void tanh(float * z, float * y, int n){
	// in our implementation, we define
	// tanh(x) = sigmoid(x)
	// there is a scaling of 2 to the input
	n--;
	while( n>= 0 ){
		//y[n] = 2.0/(1.0+exp(-z[n]))-1;
		float t = exp(z[n] * 2.0);
		y[n] = 1.0 - 2.0 / (1.0 + t);
		n--;
	}
	return;
}


void tanhPrime(float *y, float * dy_dz, int n){
	n--;
	while( n >= 0){
		//float ysigmoid = (y[n]+1.0)/2;
		dy_dz[n] = 1- y[n] * y[n]; //2 * ysigmoid * ( 1 - ysigmoid );
		n--;
	}
	return;
}

// y = ReLU(z)
void ReLU(float * z, float * y, int n){
	n--;
	while( n >= 0 ){
		if ( z[n] > 0 )
			y[n] = z[n];
		else
			y[n] = 0.0;
		n--;
	}
	return ;
}

// dy_dz =  dy / dz | evaluated at y
void ReLUPrime(float * y, float * dy_dz, int n){
	n--;
	while( n >= 0){
		if ( y[n] > 0)
			dy_dz[n] = 1.0;
		else
			dy_dz[n] = 0.0;
		n--;
	}
	return ;
}

void Softmax(float * z, float * y, int n){
	n--;
	int num = n;
	// compute the maximum of z
	float z_max = -1e10;
	while( num >= 0){
		if ( z[num] > z_max ){
			z_max = z[num];
		}
		num--;
	}

	// extract the maximum, exponentiate, and compute the denominator

	float den = 0;
	num = n;
	while( num >= 0 ){
		float tmp = z[num] - z_max;
		tmp = exp( tmp );
		den += tmp;
		z[num] = tmp;
		num--;
	}

	// compute y
	while( n >= 0 ){
		y[n] = z[n] / den;
		n--;
	}



}

void dummy(float * x, float * y, int n){
	return ;
}
