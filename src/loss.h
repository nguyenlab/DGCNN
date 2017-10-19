// y true, y prediction, number of labels
void getdLoss(char* name);
extern void (* dloss)( int ytrue, float * ypred, float* dEdz, int n);
void dEdz_MeanSquareError(int ytrue, float * ypred, float* dEdz, int n);
void dEdz_WeightedMeanSquareError(int ytrue, float * ypred, float* dEdz, int n);
void dEdz_CategoricalCrossEntropy(int ytrue, float * ypred, float* dEdz, int n);
