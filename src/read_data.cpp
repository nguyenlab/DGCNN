/*
 * read_data.cpp
 *
 *  Created on: Mar 10, 2015
 *      Author: mou
 */


#include"read_data.h"
#include <stdio.h>
#include <exception>
#include <iostream>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include"FFNN.h"
#include"activation.h"
#include <errno.h>
#define sizeofint 4
#define sizeoffloat 4
#define sizeofchar  1
#define MAX_BUF 1500000000
char * buf_train= new char [MAX_BUF];
char * cursor_train;
char * buf_CV= new char [MAX_BUF];
char * cursor_CV;
char * buf_test= new char [MAX_BUF];
char * cursor_test;

char * f_train;
char * f_CV;
char * f_test;
char * f_yCV;
char * f_ytest;
char * f_ytrain;
void (* act_f)( float *z, float *y, int n); // activation function, y = f(z)
void (* act_fprime)( float * y, float * dy_dz, int n);// derivative
void getActiveFunction(char* name)
{
	act_f = tanh;
	act_fprime = tanhPrime;
	char * act_name = (char*)malloc(200*sizeof(char)) ;
	strcpy(act_name, "tanh");
	
	if (strcmp(name, "ReLU")==0)
	{
		act_f = ReLU;
		act_fprime = ReLUPrime;		
		strcpy(act_name, "ReLU");
	}
	printf("\nActivation function: %s\n", act_name);
	
}	
void DeleteClass(Layer ** layers, int len){
	if (layers == NULL)
			return;
	//return;
	//cout<<"deleting !! len = "<<len<<endl;
	//cout<<"real len = "<<endl;
	for (int i = 0; i< len; ++i){
		Layer * lay = layers[i];
		if (lay == NULL)
			continue;
		for (int j = 0; j< lay->numUp; ++j){
			Connection * con = lay->connectUp[j];
			if (con != NULL)
				delete con;
		}
		//if(i == 310)
			//cout<<":!!!!"<<endl;


			if (lay->dE_dy != NULL)
				delete lay->dE_dy;

			if (lay->dE_dz != NULL && lay->dE_dz != lay->dE_dy)
				delete lay->dE_dz;
			if (lay->dy_dz != NULL)
				delete lay->dy_dz;

			if (lay->y != NULL && lay->y != lay->dE_dz)
				delete lay->y;
			if (lay->z!=NULL && lay->y!=lay->z && lay->z != lay->dE_dz)
				delete lay->z;


			//cout<<"before delete connect!!"<<endl;

		//lay = layers[i];
		if (lay->connectUp != NULL)
			delete lay->connectUp;
		if (lay->connectDown != NULL)
			delete lay->connectDown;

		delete lay;

	}
	delete layers;
}

void ReadOneTrainFile( int num_sample, Layer *** &net_array, int *& len){
	// for all samples
	for( int i_sample = 0; i_sample < num_sample; ++ i_sample){
		//cout<<i_sample<<endl;
		ConstructNetFromBuf(net_array[i_sample], len[i_sample], cursor_train);

	}
	return;
}
void ReadOneCVFile( int num_sample, Layer *** &net_array, int *& len){
	// for all samples
	for( int i_sample = 0; i_sample < num_sample; ++ i_sample){
		//cout<<i_sample<<endl;
		ConstructNetFromBuf(net_array[i_sample], len[i_sample], cursor_CV);

	}
	return;
}

void ReadOneTestFile( int num_sample, Layer *** &net_array, int *& len){
	// for all samples
	for( int i_sample = 0; i_sample < num_sample; ++ i_sample){
		//cout<<i_sample<<endl;
		ConstructNetFromBuf(net_array[i_sample], len[i_sample], cursor_test);

	}
	return;
}


void ReadAllData( ){

	printf("data files (x_y - train -CV - test):\n");
	printf("%s\n%s\n%s\n%s\n%s\n%s\n", f_train, f_ytrain, f_CV, f_yCV, f_test, f_ytest);

	X_train    = new Layer **[1];
//	X_trainB    = new Layer **[1];

	X_CV    = new Layer **[1];
	X_test  = new Layer **[1];

	len_X_train    = new int [1];
//	len_X_trainB    = new int [1];
	len_X_CV    = new int [1];
	len_X_test  = new int [1];

	y_train = new int[num_train];
	y_CV    = new int[num_CV];
	y_test  = new int[num_test];

	//const char * f_train = "/home/seke/Workspace/TBCNN_Csmallnet/xy/small_train";
	//const char * f_CV = "/home/seke/Workspace/TBCNN_Csmallnet/xy/small_CV";
	//const char * f_test = "/home/seke/Workspace/TBCNN_Csmallnet/xy/small_test";

	FILE * infile = fopen(f_CV, "rb");
	
	fread( buf_CV, 1,  MAX_BUF, infile);
	cursor_CV = buf_CV;
	//ReadOneFile( num_CV,    X_CV,    len_X_CV);
	fclose(infile);

	infile = fopen(f_test, "rb");
	fread( buf_test, 1,  MAX_BUF, infile);
	cursor_test = buf_test;
	//ReadOneFile( num_test,  X_test,  len_X_test);
	fclose(infile);


	infile = fopen(f_train, "rb");
	fread( buf_train, 1,  MAX_BUF, infile);
	cursor_train = buf_train;
	fclose(infile);

	//ReadTrainNetwork(0);
//	infile = fopen(f_train, "r");
	char tmpstring[20];


	// y cv
	//infile = fopen("/home/seke/Workspace/TBCNN_Csmallnet/xy/smally_CV.txt", "r");
	//const char * f_yCV = "./xy/testdata_yCV.txt";
	infile = fopen(f_yCV, "r");
	if( infile == NULL)
		cout << "ERROR: cannot load: " << "y_CV.txt" << endl;
	for ( int i = 0; i < num_CV; ++i){
		fgets( tmpstring, 100, infile);
		y_CV[i] = atoi(tmpstring);
	}
	fclose(infile);
	// y test
	//infile = fopen("/home/seke/Workspace/TBCNN_Csmallnet/xy/smally_test.txt", "r");
	//const char * ytest = "./xy/testdata_ytest.txt";
	infile = fopen(f_ytest, "r");

	if( infile == NULL)
		cout << "ERROR: cannot load: " << "y_test.txt" << endl;
	for ( int i = 0; i < num_test; ++i){
		fgets( tmpstring, 100, infile);
		y_test[i] = atoi(tmpstring);
	}
	fclose(infile);

	// y train
	//infile = fopen("/home/seke/Workspace/TBCNN_Csmallnet/xy/smally_train.txt", "r");
	infile = fopen(f_ytrain, "r");
	if( infile == NULL)
		cout << "ERROR: cannot load: " << "y_train.txt" << endl;

	for ( int i = 0; i < num_train; ++i){
		fgets( tmpstring, 100, infile);
		y_train[i] = atoi(tmpstring);
	}
	fclose(infile);
	return;
}

void ReadTrainNetwork ( int suffix = 0 ,int i = 0) {

	if (suffix == 0) {
		cursor_train = buf_train;
	}
	//cout<<"Begin!!!!!!!!"<<endl;
//	if(i == 0){
//		DeleteClass(X_train[0], len_X_train[0]);
//		ReadOneTrainFile( 1, X_trainB, len_X_trainB);
//		//cout<<"YYYYYYYYA"<<endl;
//	}
//	else
	{
		DeleteClass(X_train[0], len_X_train[0]);
//		DeleteClass(X_trainB[0], len_X_trainB[0]);
		ReadOneTrainFile( 1, X_train, len_X_train);
		//cout<<"YYYYYYYYB"<<endl;
	}


}

void ReadCVNetwork( int suffix = 0 ) {

	if (suffix == 0) {
		cursor_CV = buf_CV;
	}
	DeleteClass(X_CV[0], len_X_CV[0]);
	ReadOneCVFile( 1, X_CV, len_X_CV);


}
void ReadTestNetwork( int suffix = 0 ) {

	if (suffix == 0) {
		cursor_test = buf_test;
	}
	DeleteClass(X_test[0], len_X_test[0]);
	ReadOneTestFile( 1, X_test, len_X_test);


}

void ConstructNetFromBuf( Layer ** & one_net, int & len, char * & cursor){
	int numlay, numcon;


	numlay = *((int *) cursor);     cursor += sizeofint;
	numcon = *((int *) cursor);     cursor += sizeofint;
//	cout<<numlay<<"numlay"<<endl;
//	cout<<numcon<<"numcon"<<endl;
//	fread( &numlay, sizeofint, 1, infile);
//	fread( &numcon, sizeofint, 1, infile);

	len = numlay;
	// read layers
	one_net = new Layer*[numlay];

	for(int i = 0; i < numlay; i++){
		int numUnit, numUp, numDown;
		char activation;

		numUnit = *((int *) cursor);  cursor += sizeofint;
		numUp   = *((int *) cursor);  cursor += sizeofint;
		numDown = *((int *) cursor);  cursor += sizeofint;
		activation = *cursor;         cursor += sizeofchar;

//		fread( &numUnit, sizeofint, 1, infile );
//		fread( &numUp,   sizeofint, 1, infile );
//		fread( &numDown, sizeofint, 1, infile );
//		fread( &activation, sizeof(char), 1, infile);
//		cout << activation <<endl;
//		cout << "numUnit " << numUnit << "; numUp " << numUp
//				<< "; numDown " << numDown << "; activate " << activation << endl;
		if ( activation == 'x' || activation == 'u'){ // max pooling
			one_net[i] = new PositivePoolLayer( "pool", numUnit, numUp, numDown);

		}
		else {	// seems to need a more parameter bidx

			int bidx = 0;


			// Hidden and con dropout, with activation function ReLU; embedding and autoencoding layer
			// do not dropout, with activation function tanh.
			// ReLU by default
			bidx = * ((int *) cursor);  cursor += sizeofint;
//			fread( & bidx, sizeofint, 1, infile );
			if( activation == 'r' ){ // relu
				one_net[i] = new Layer( "relu", numUnit, bidx, numUp, numDown, act_f, act_fprime);
			}
			else if( activation == 'c' ){ // drop and relu
				one_net[i] = new Layer( "con", numUnit, bidx, numUp, numDown, act_f, act_fprime);

			}
			else if( activation == 'v' ){ // recursive
				one_net[i] = new Layer( "recursive", numUnit, bidx, numUp, numDown, act_f, act_fprime);
			}
			else if( activation == 'a' ){ 
				one_net[i] = new Layer( "ae", numUnit, bidx, numUp, numDown, act_f, act_fprime);
			}
			else if( activation == 'b' ){ 
							one_net[i] = new Layer( "combination", numUnit, bidx, numUp, numDown, act_f, act_fprime);
						}
			else if( activation == 's') { // softmax
				one_net[i] = new Layer( "softmax", numUnit, bidx, numUp, numDown, Softmax, dummy);
			}
			else if( activation == 'e'){ 
				one_net[i] = new Layer( "embed", numUnit, bidx, numUp, numDown, act_f, act_fprime);
				delete one_net[i]->z;
				one_net[i]->z = NULL;
			}
			else if( activation == 'h'){ // dropout and relu
				one_net[i] = new Layer( "hidden", numUnit, bidx, numUp, numDown, act_f, act_fprime);
			}

		}

	}
	// read connections
	for(int i = 0; i < numcon; i++){

		int xid, yid, xupid, ydownid, Widx;


		xid = *((int *) cursor);   cursor += sizeofint;
		yid = *((int *) cursor);   cursor += sizeofint;
		xupid = *((int *) cursor); cursor += sizeofint;
		ydownid = *((int *) cursor); cursor+=sizeofint;
		Widx = *((int *) cursor);  cursor += sizeofint;

//		fread(&xid, sizeofint, 1, infile);
//		fread(&yid, sizeofint, 1, infile);
//		fread(&xupid, sizeofint, 1, infile);
//		fread(&ydownid, sizeofint, 1, infile);
//		fread(&Widx, sizeofint, 1, infile);

//		cout << cursor - sizeofint - buf << endl;
//		cout << *(int * )(buf + 260009)<<endl;
//		cout << "xid " << xid << "; yid " << yid << "; xupid " << xupid
//				<< "; ydownid " << ydownid << " Widx " << Widx << endl;
		Connection * con;

		if ( Widx < 0 ){	// pooling layer, assume max pooling
			 con = new PositivePoolConnection( one_net[xid], one_net[yid],
					                      one_net[yid]->numUnit , 1);
								// type: max pooling = 1
		} else { // linear combination

			float coef;
			coef = * ((float *)cursor); cursor += sizeoffloat;
//			cout << coef << endl;
//			fread( & coef, sizeof(float), 1, infile);
			con = new Connection(one_net[xid], one_net[yid],
								one_net[xid]->numUnit, one_net[yid]->numUnit,
								Widx, coef);

		}
		one_net[xid]->connectUp[xupid] = con;

		one_net[yid]->connectDown[ydownid] = con;
	}
	return ;
}

void saveTBCNNParam(int epoch)
{
	cout<<"save param epoch = "<<epoch<<endl;
	char s[10];
	//char * fweights = "/home/seke/Workspace/TBCNN_Csmallnet/TBCNNParam/Weights_";
	//char * fbiases = "/home/seke/Workspace/TBCNN_Csmallnet/TBCNNParam/Biases_";
	string strw = "./TBCNN_CsmallnetTBCNNParam/Weights_";
	string strb = "./TBCNN_CsmallnetTBCNNParam/Biases_";
	sprintf(s, "%d", epoch);
	string tmp;
	tmp = s;
	strw = strw+s+".txt";
	strb = strb+s+".txt";
	const char * fweights = strw.c_str();
	const char * fbiases = strb.c_str();
	//float * we;


	//strcat(fbiases,s);

	FILE * fw ;
	//cout<<"saveParam2222"<<endl;
	fw = fopen(fweights, "w+");
    if(fw == NULL)cout<<"FFFFFFFFF"<<endl;
	for(int ii = 0;ii<num_weights;ii++)
		fprintf(fw,"%f ",weights[ii]);

	fclose(fw);
	fw = fopen(fbiases, "w+");
    if(fw == NULL)cout<<"FFFFFFFFFB"<<endl;
	for(int ii = 0;ii<num_biases;ii++)
		fprintf(fw,"%f ",biases[ii]);
/*
	if(true)
	{
		cout<<"write : epoch = "<<epoch<<endl;
		cout<<weights[10]<<endl;
		cout<<biases[20]<<endl;
		cout<<weights[-1]<<endl;
		cout<<biases[-1]<<endl;
	}*/
	fclose(fw);
	//cout<<">>>>"<<endl;

}
void readTBCNNParam(int epoch)
{
	//fscanf(fp,"%d%d" ,&x,&y);
	cout<<"read param epoch = "<<epoch<<endl;
	char s[10];
	//char * fweights = "/home/seke/Workspace/TBCNN_Csmallnet/TBCNNParam/Weights_";
	//char * fbiases = "/home/seke/Workspace/TBCNN_Csmallnet/TBCNNParam/Biases_";
	string strw = "./TBCNN_Csmallnet/TBCNNParam/Weights_";
	string strb = "./TBCNN_Csmallnet/TBCNNParam/Biases_";
	sprintf(s, "%d", epoch);
	string tmp;
	tmp = s;
	strw = strw+s+".txt";
	strb = strb+s+".txt";
	const char * fweights = strw.c_str();
	const char * fbiases = strb.c_str();
	//float * we;


	//strcat(fbiases,s);

	FILE * fw ;
	//cout<<"saveParam2222"<<endl;
	fw = fopen(fweights, "r+");
    if(fw == NULL)cout<<"FFFFFFFFF"<<endl;
	for(int ii = 0;ii<num_weights;ii++)
		fscanf(fw,"%f",&weights[ii]);

	fclose(fw);
	fw = fopen(fbiases, "r+");
    if(fw == NULL)cout<<"FFFFFFFFFB"<<endl;
	for(int ii = 0;ii<num_biases;ii++)
		fscanf(fw,"%f",&biases[ii]);
/*
	if(true)
	{
		cout<<"read : epoch = "<<epoch<<endl;
		cout<<weights[10]<<endl;
		cout<<biases[20]<<endl;
		cout<<weights[-1]<<endl;
		cout<<biases[-1]<<endl;
	}*/

	fclose(fw);

}
