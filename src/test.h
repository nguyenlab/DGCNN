/*
 * test.h
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

#ifndef TEST_H_
#define TEST_H_

#include"FFNN.h"

extern Layer ** Layers;
void test_ReLU();
void test_prop();
void test_maxpool();
void test_loaddata(char file[]);
#endif /* TEST_H_ */
