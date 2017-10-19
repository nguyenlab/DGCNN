/*
 * read_data.h
 *
 *  Created on: Mar 10, 2015
 *      Author: mou
 */

#ifndef READ_DATA_H_
#define READ_DATA_H_

#include"global.h"
#include"FFNN.h"
void getActiveFunction(char* name);
Layer ** ReadOneNet(char * filepath, int& len);
void ReadIndex(char * datapath);
void ReadAllData();
void DeleteClass(Layer ** layers, int len);
void ReadTrainNetwork( int suffix,int i);
void ReadTestNetwork( int suffix) ;
void ReadCVNetwork( int suffix) ;
void ConstructNetFromBuf( Layer ** & one_net, int & len, char * & cursor);
//void ReadOneFile( char * filename, int num_sample, Layer *** &net_array, int *& len);
void ReadOneTrainFile( int num_sample, Layer *** &net_array, int *& len);
void ReadOneCVFile( int num_sample, Layer *** &net_array, int *& len);
void ReadOneTestFile( int num_sample, Layer *** &net_array, int *& len);
void saveTBCNNParam(int epoch);
void readTBCNNParam(int epoch);
#endif /* READ_DATA_H_ */
