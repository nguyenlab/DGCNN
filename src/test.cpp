/*
 * test.cpp
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

#include"FFNN.h"
#include"activation.h"
#include<math.h>
#include<malloc.h>
#include"read_data.h"

Layer ** Layers = new Layer *[3];


void print_array(char *message, float *a, int n){
	cout << message;
	for(int i = 0; i < n; i++)
		cout << a[i] << ", ";
	cout << endl;
}
void test_ReLU(){
	cout << "TEST ReLU:" << endl;
	float x[] = {-0.3, -0.2, -0.1, 0.0, 0, .1, .2, .3, .4, -0.1};
	float y[10] = {};
	float dydx[10] = {};

	RandomInitParam();

	Layer * layer = new Layer( "test", 100, 0, 0, 0, NULL, NULL);

	(* layer->f)(x, y, 10);

	(* layer->fprime)(y, dydx, 10);
	print_array("    input     ", x, 10);
	print_array("    output    ", y, 10);
	print_array("    derivative", dydx, 10);

}

void test_prop(){
	cout << "TEST forward propagation and back propagation:" << endl;

#define num1  200
#define num2  300
#define num3  3
//	float myWeights[] = {-1,2, 3,-2,-1, 1,    1,2,3,-1,-2,-2};
//	float myBiases[]  = { -2,1,   2,-1,1,    1, 30};

	Layer * layer1 = new Layer( "input", num1, 0, 2, 0, NULL, NULL);
	layer1->z = NULL;
	Layer * layer2 = new Layer( "hid",   num2, num1, 1, 1, NULL, NULL);
	Layer * layer3 = new Layer( "output",num3, num1+num2, 0, 2, Softmax, dummy);


	Connection * con1 = new Connection(layer1, layer2, num1, num2, 0, 1.0);
	Connection * con2 = new Connection(layer2, layer3, num2, num3, num1*num2, 1.0);
	Connection * con3 = new Connection(layer1, layer3, num1, num3, 2, 1.0);

	layer1->connectUp[0] = con1;
	layer1->connectUp[1] = con3;
	layer2->connectDown[0] = con1;

	layer2->connectUp[0] = con2;
	layer3->connectDown[0] = con2;
	layer3->connectDown[1] = con3;


	Layers[0] = layer1;
	Layers[1] = layer2;
	Layers[2] = layer3;

time_t t1 = time(NULL);

for(int ite = 0; ite < 10; ite++){// 10 for testing soundness, any number for testing the efficiency
	FeedForward(Layers, 3);


//	cout << "Feed forward! The result should be" << endl
//		 << "    [[ 11.  19.]]" << endl
//		 << "and the network says" << endl;

//	print_array(layer3->y, 2); // the result should be 19, 16
//	cout << "Now, back propagate! The result should be" << endl
//		 << "    [[ 16  -8 -22  11 -28  14  66   0  44 114   0  76]]" << endl
//		 << "    [[ 27 -24  -8   0  -5  11  19]]"<< endl
//		 << "and the network says" << endl;

	// First add cross entropy error
	// the target label is 1

	int target = 1;

	CleanDerivative(Layers, 3);

	for(int i = 0; i < num3; i++){
		Layers[2]->dE_dz[i] = Layers[2]->y[i];
	}
	Layers[2]->dE_dz[target] -= 1;

	BackPropagation(Layers, 3);

	GradDescent(num_weights, 0.03, gradWeights, weights);
	GradDescent(num_biases,  0.03, gradBiases, biases);

	memset(gradWeights, 0, sizeof(float)*num_weights);
	memset(gradBiases,  0, sizeof(float)*num_biases);
}
time_t t2 = time(NULL);

	cout << "time " << t2 - t1 << endl;
	print_array("    weights: ", weights, 5);
	print_array("    biases:  ", biases, 5);
//	print_array(biases, 5);

	print_array("    output (probabilities):", layer3->y, 3);

	cout << "The result should be"<< endl
		<< "[[ 0.00331745 -0.00881416  0.0185164   0.00703729  0.01413297]]" << endl
		<< "[[ 0.01161394  0.00864863 -0.00029485 -0.01346057 -0.01507052]]" << endl
		<< "[ 0.30441662, 0.39588947, 0.29969391]"<<endl;

	return ;

}

void test_maxpool(){
	cout << "TEST forward propagation and back propagation:" << endl;

	// CAUTION
	// don't use static arrays unless the function lasts til the end

	float myWeights[] = {0};
	float myBiases[]  = {1,2,3,4,  5,4,3,2};
	float gradW[] = {0};
	float gradB[8] = {0};

	num_weights = 1;
	num_biases = 8;

	weights = myWeights;
	biases  = myBiases;
	gradWeights = gradW;
	gradBiases  = gradB;

	int num = 4;

	Layer * layer1 = new Layer( "in1",   num,   0, 1,  0, NULL, NULL);
	layer1->z = NULL;
	Layer * layer2 = new Layer( "in2",   num, num, 1, 0, NULL, NULL);
	layer2->z = NULL;
	Layer * layer3 = new PositivePoolLayer( "pool", num, 0, 2);

	Connection * con1 = new PositivePoolConnection( layer1, layer3, num, 1);
	Connection * con2 = new PositivePoolConnection( layer2, layer3, num, 1);


	layer1->connectUp[0] = con1;
	layer2->connectUp[0] = con2;

	layer3->connectDown[0] = con1;
	layer3->connectDown[1] = con2;


	Layers[0] = layer1;
	Layers[1] = layer2;
	Layers[2] = layer3;


	FeedForward(Layers, 3);

	CleanDerivative(Layers, 3);

	for(int i = 0; i < num; i++){
		layer3->dE_dy[i] = i + .1;
	}


	BackPropagation(Layers, 3);

	cout << "max{[1,2,3,4], [5,4,3,2]}" << endl;
	print_array("    output:  ", layer3->y,   4);
	print_array("    weights: ", gradWeights, 1);
	print_array("    biases:  ", gradBiases,  8);


	return;
}

void test_loaddata(char *file){

	//Layers = ReadOneNet(file, );

	int target = 1;

	cout << "here" << endl;
time_t t1 = time(NULL);

for(int x = 0; x < 100000; x++){

	FeedForward(Layers, 35);

	CleanDerivative(Layers, 35);


	for(int i = 0; i < 10; i++){
		Layers[34]->dE_dz[i] = Layers[34]->y[i];
	}
	Layers[34]->dE_dz[target] -= 1;

	BackPropagation(Layers, 35);
}

time_t t2 = time(NULL);

cout << t2 - t1 << endl;
//	print_array( "    dE_dy", Layers[34]->dE_dz, 3);
//	print_array( "    output:          ", Layers[34]->y, 3);
//	print_array( "    grad of weights: ", gradWeights, 5);
//	print_array( "    grad of biases:  ", gradBiases,  5);
////	cout << "Ground truth:" << endl
////		 << "[[ 0.33795485,  0.33016045,  0.3318847 ]]" << endl
////		 << "[[ 0.          0.          0.06260947  0.03210237  0.00080014]]" << endl
////		 << "[[ 0.25536022  0.04318755  0.02116565  0.00162851  0.11008303]]" << endl;
//	cout << Layers[34]->y[0] << endl;
//	cout << "-----" << endl;
//	for (int i = 34; i>=0; i--){
//		cout << Layers[i]->dE_dy[0] << endl;
//	}
}
