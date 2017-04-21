/*
 * main.cpp
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

//#include "global.h"
#include "global.h"
#include <string>
#include "FFNN.h"
#include <cmath>
#include"read_data.h"
#include"predict.h"
#include <stdlib.h>
#include<math.h>
#include <stdio.h>
#include"activation.h"
#include <malloc.h>
#include <ctime>
#include "file_io.cpp"
using namespace std;

//void saveParam(int epoch);

Layer ** global;
int main(int argc, char* argv[]){
	
	char * testnet =  "./xy/data_train"; 
	Layer ***X_net    = new Layer **[1];

	FILE *infile = fopen(testnet, "rb");
	long MAX_BUF = 1500000000;
	char * cursor_net= new char [MAX_BUF];
	fread( cursor_net, 1,  MAX_BUF, infile);
	fclose(infile);

	int len=0;
	ConstructNetFromBuf(X_net[0], len,cursor_net);

}


