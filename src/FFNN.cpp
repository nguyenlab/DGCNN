//============================================================================
// Name        : FFNN.cpp
// Author      : 
// Version     :
// Copyright   : None
// Description : Hello World in C++, Ansi-style
//============================================================================
//#include"global.h"
#include"FFNN.h"
#include"activation.h"
#include<math.h>
#include <malloc.h>
#include"global.h"
#include<stdlib.h>
int * file_train;
int * file_CV;
int * file_test;

Layer *** X_train;
//Layer *** X_trainB;
Layer *** X_CV;
Layer *** X_test;
Layer *** X_train_origin;

int * len_X_train;
//int * len_X_trainB;
int * len_X_CV;
int * len_X_test;
int * len_X_train_origin;

int * y_train;
int * y_CV;
int * y_test;
int * y_train_origin;
float C_weights;
float p_dropout1;
float p_dropout2;
int batch_size;
bool isTraining;


//////////////////////////////////////////////////////////
// forward propagation and back propagation

void FeedForward(Layer ** layer, int n){
	//cout<<"in FF "<<n<<endl;
	for (int i = 0; i < n; i++){
		//cout << i <<"?????????"<<endl;
		//cout << layer[i]->name<<"!!!!!!!"<<endl;
		layer[i]->computeY();
		/*
		if(layer[i]->name == "softmax")
		{
			printf("output : %4f, %4f\n", layer[i]->y[0],layer[i]->y[1]);
			cout<<"in FFNN : i = "<<i<<" n =  "<<n<<endl;
		}
		if(layer[i]->name == "hidden")
		{
			printf("hidden : %4f, %4f\n", layer[i]->y[0],layer[i]->y[1]);
		}
		*/

		//cout<<layer[i]->name<<" i = "<<i<<"  n = "<<n<<"  ";
		//printf("%4f, \n", layer[i]->y[1]);
	}
	//cout<<"FF end"<<endl;
}
void BackPropagation( Layer ** layer, int n){
	for (int i = n - 1; i >= 0; i--){

		layer[i]->updateB();
	}
}
#define FLOAT_SIZE 4
void CleanDerivative( Layer ** layer, int n){
	for (int i = 0; i < n; i++){
		memset( layer[i]->dE_dy, 0 , FLOAT_SIZE * layer[i]->numUnit  );
	}
}

///////////////////////////////////////////////////////////
// layer functions
Layer::Layer(string _name, int _numUnit, int _bidx,
		int _numUp, int _numDown,
		void(* _f)(float *x, float *y, int n),
		void(* _fprime)(float *, float * dy_dz, int n) ){

//	int size = sizeof(float) * _numUnit;

	// neuron properties
	name =_name;
	numUnit = _numUnit;
	numUp = _numUp;
	numDown = _numDown;
	bidx = _bidx;

	if ( _f == NULL ){
		f = ReLU;
		fprime = ReLUPrime;
	}else{
		f = _f;
		fprime = _fprime;

	}
	isDropOut = false;
	dropout_rate = 0.0;
	// initialize states (allocate memory)
//	z = (float *) malloc( size);
//	y = (float *) malloc( size);
//	dE_dz = (float *) malloc( size );
//	dE_dy = (float *) malloc( size );
//	dy_dz = (float *) malloc( size );

	z = new float[ numUnit ];
	y = new float[ numUnit ];
	dE_dz = new float[ numUnit ];
	dE_dy = new float[ numUnit ];
	dy_dz = new float[ numUnit ];
	// connections
	connectUp = new Connection *[_numUp];
	connectDown = new Connection * [_numDown];

	// activate

}

DropoutLayer::DropoutLayer(string name, int numUnit, int bidx, int numUp, int numDown,
				void (* f)(float *z, float *y, int n),
				void (* fprime)(float *y, float * dy_dz, int n), float dr)
	: Layer(name, numUnit, bidx, numUp, numDown, f, fprime ){

	indicator = new int[numUnit];
	isDropOut = true;
	dropout_rate = dr;
};

DropoutLayer::~DropoutLayer(){
	delete indicator;
}
void DropoutLayer::computeY(){

	// Embeddings do not have input.
	// Thus, y = biases[bidx]
	int num = this->numUnit;


	if ( this->z == NULL ){
		icopy( num , biases + bidx, this->y );
		//cblas_scopy(n, x, 1, y, 1);
		return;
	}

	// otherwise, y = f(W*x+b)

	// first, compute z = b + sum_i W_i * x_i
	memset(this->z, 0, sizeof(float) * num);
	for ( int i = 0; i < numDown; i++ ){
		connectDown[i] -> computeZ();
		// z += connectDown[i].tmpZ
		//iXpY(num, connectDown[i]->tmpZ, this->z);
	}


	if (bidx > 0)
		iXpY(num, biases + bidx, this->z);
	// second, apply the activation function
	(* (this->f)) (this->z, this->y, num);

	// training
	// dropout !!!
	if( isTraining ){
		//cout<<"dropout forward"<<endl;
		for (int i = 0; i < num; i++){
			int t = rand();

			int tmp_ind = ( (t % 100) >= (dropout_rate * 100) ) ;
			this->indicator[i] = tmp_ind;

			this->y[i] *= tmp_ind;

		}
	}

	return;
}



void DropoutLayer::updateB(){

	int num = this->numUnit;
	//cout<<"Error!!!!!!!!!!!!!!!!in DropoutLayer!!!!!!!"<<endl;
	if (this->z == NULL){ // embeddings
		// gradB += dE_dy, because y = b
		iXpY( num , this->dE_dy, gradBiases + bidx );

		return;
	}

	if (fprime != dummy){

		// dy_dz = f', evaluated at y
		( * this->fprime)(this->y, this->dy_dz, num);
		// dE_dz = dE_dy .* dy_dz

		pointwise_dot(this->dE_dy, this->dy_dz, this->dE_dz, num);
		//cout<<"dropout backward"<<endl;
		for( int i = 0; i < num; ++i){

			this->dE_dz[i] *= this->indicator[i];

		}

	}// else if fprime == softmaxprime{
		// do nothing, because we assume dE_dz is given by softmax

	//}


	// dE_db = dE_dz * 1<size_data x 1> / size_data
	//       = dE_dz   if size of data = 1
	iXpY( num , this->dE_dz, gradBiases + bidx );


	for(int i = 0; i < this->numDown; i++){
		connectDown[i]->updateW();
	}


	return;
//		ReLUPrime(float * y, float * dy_dz, int n);
}




void Layer::computeY(){
	//cout << "in base" << endl;
	// Embeddings do not have input.
	// Thus, y = biases[bidx]
	int num = this->numUnit;
	//cout<<"aaaaaaaaa"<<endl;

	if ( this->z == NULL ){
		icopy( num , biases + bidx, this->y );
		//cblas_scopy(n, x, 1, y, 1);
		//cout<<"bidx = "<<bidx<<endl;
		return;
	}
	//cout<<"bbbbbbbbbbbbbb"<<endl;
	// otherwise, y = f(W*x+b)

	// first, compute z = b + sum_i W_i * x_i
	if (bidx > 0)
		icopy(num, biases + bidx, this->z);
	else
		memset(this->z, 0, sizeof(float) * num);
	//cout<<"cccccccccccc"<<endl;

	for ( int i = 0; i < numDown; i++ ){
		connectDown[i] -> computeZ();
		// z += connectDown[i].tmpZ
		//iXpY(num, connectDown[i]->tmpZ, this->z);
	}
	//cout<<"ddddddddddd"<<endl;
	// second, apply the activation function
	(* (this->f))(this->z, this->y, num);
	//cout<<"eeeeeeeeee"<<endl;
	return;
}



void Layer::updateB(){

	int num = this->numUnit;

	if (this->z == NULL){ // embeddings
		// gradB += dE_dy, because y = b
		iXpY( num , this->dE_dy, gradBiases + bidx );

		return;
	}

	if (fprime != dummy){
		// dy_dz = f', evaluated at y
		( * this->fprime)(this->y, this->dy_dz, num);
		// dE_dz = dE_dy .* dy_dz
		pointwise_dot(this->dE_dy, this->dy_dz, this->dE_dz, num);
	}// else if fprime == softmaxprime{
		// do nothing, because we assume dE_dz is given by softmax

	//}


	// dE_db = dE_dz * 1<size_data x 1> / size_data
	//       = dE_dz   if size of data = 1
	if(this->name == "combination")
	{
		//cout<<"this combination's dE_dz: "<<this->dE_dz[0]<<" "<<this->dE_dz[1]<<" "<<this->dE_dz[2]<<" "<<endl;
		//cout<<"this combination's dE_dy: "<<this->dE_dy[0]<<" "<<this->dE_dy[1]<<" "<<this->dE_dy[2]<<" "<<endl;
		//cout<<"this combination's dy_dz: "<<this->dy_dz[0]<<" "<<this->dy_dz[1]<<" "<<this->dy_dz[2]<<" "<<endl;

	}
	iXpY( num , this->dE_dz, gradBiases + bidx );


	for(int i = 0; i < this->numDown; i++){
		connectDown[i]->updateW();
	}

	//cout<<this->name<<"  gradB : "<<gradBiases[0]<<" "<<gradBiases[1]<<" "<<gradBiases[2]<<" "<<endl;
	return;
//		ReLUPrime(float * y, float * dy_dz, int n);
}



PositivePoolLayer::PositivePoolLayer(string _name, int _numUnit, int _numUp, int _numDown)
	:Layer(_name, _numUnit, -1, _numUp, _numDown, dummy, dummy){
	delete this->dE_dz;
	dE_dz = dE_dy;
	delete this->z;
	z = y;
}

void PositivePoolLayer::computeY(){
	memset(this->z, 0, sizeof(float) * this->numUnit);
	for ( int i = 0; i < numDown; i++ ){
		connectDown[i] -> computeZ();
	}
	return;
}
void PositivePoolLayer::updateB(){
	for(int i = 0; i < this->numDown; i++){
		connectDown[i]->updateW();
	}
	return ;
}
///////////////////////////////////////////////////////
// connection functions
Connection::Connection(){}// do nothing, reserved for derived classed

Connection::Connection(Layer * _x, Layer * _y,
		int _xnum, int _ynum, int _Widx, float _Wcoef){

	xlayer = _x;
	ylayer = _y;
	xnum = _xnum;
	ynum = _ynum;
	Widx = _Widx;
	Wcoef = _Wcoef;

//	tmpZ = (float *) malloc( sizeof(float) * _ynum );

}

void Connection::computeZ(){
	// z += coef * W * x
	//cout<<this->xlayer->name<<"  "<<this->ylayer->name<<endl;

	//cout<<this->ynum<<" "<< this->xnum<<" "<< this->Wcoef<<" "<<Widx<< endl;
	// not dropout
	if ( isTraining || ! this->xlayer->isDropOut )
		selfplus_matrix_dot_vector(this->ynum, this->xnum, this->Wcoef,
			          weights + this->Widx, CblasNoTrans,this->xlayer->y,
			          ylayer->z);
	else{ // testing, dropout
		selfplus_matrix_dot_vector(this->ynum, this->xnum, this->Wcoef * (1- this -> xlayer -> dropout_rate),
					  weights + this->Widx, CblasNoTrans,this->xlayer->y,
					  ylayer->z);
//		cout << "here " << xlayer->name << xlayer->numUnit << endl;
//		int *p = NULL;
//		cout << *p << endl;
	}
	//cout<<"Compute Z finish"<<endl;
	return;
}

void Connection::updateW(){
	// dE_dW += coef * dE_dz * x.T
	//    note that x.T is exactly x, provided that x is a column vector

	selfplus_matrix_dot_matrix(this->ynum, this->xnum, 1, this->Wcoef,
					  this->ylayer->dE_dz, CblasNoTrans,
					  this->xlayer->y, CblasNoTrans,
					  gradWeights + this->Widx
	);


	//dE_dx += coef * W.T * dE_dz
	selfplus_matrix_dot_vector( this->ynum, this->xnum, this->Wcoef,
						weights + Widx, CblasTrans,
						this->ylayer->dE_dz,
						this->xlayer->dE_dy
	);
	/*
	for(int ii = 0;ii<this->xnum;ii++)
	{
		//cout<<this->xlayer->dE_dy[ii]<<" ";
		if (!(xlayer->dE_dy[ii]>=0 || xlayer->dE_dy[ii]<=0))
			cout<<"xlayer->dE_dy is nan at i = "<<ii<<" and xlayer is "<<xlayer->name<<endl;
	}
	*/
	for(int ii = 0;ii<this->ynum;ii++)
	{
		if (!(ylayer->dE_dz[ii]>=0 || ylayer->dE_dz[ii]<=0))
		{
			cout<<"ylayer->dE_dz is "<<ylayer->dE_dz[ii]<<" at i = "<<ii<<" and ylayer is "<<ylayer->name<<endl;
			cout<<" and xlayer is"<<xlayer->name<<endl;
			break;
		}
	}
/*
	if(this->xlayer->name == "combination")
	{
		cout<<"ynum : "<<this->ynum<<endl;
		cout<<"xnum : "<<this->xnum<<endl;
		cout<<"combination layer's dE_dy!!!!!!!!!!!!!!!!!!!!!  ";
		for(int ii = 0;ii<this->xnum;ii++)
			cout<<this->xlayer->dE_dy[ii]<<" ";
		if (!(xlayer->dE_dy[3]>=0 || xlayer->dE_dy[3]<=0))cout<<endl<<"Come On!!!!!!!!";
		cout<<endl<<"combination UPlayer's dE_dz!!!!!!!!!!!!!!!!!!!!!  ";
		for(int ii = 0;ii<this->ynum;ii++)
			cout<<ylayer->dE_dz[ii]<<" ";
		cout<<endl;

	}*/
	//if(this->xlayer->name == "hidden")
		//cout<<" UPlayer's dE_dz!!!!!!!!!!!!!!!!!!!!!  "<<this->ylayer->dE_dz[0]<<" "<<this->ylayer->dE_dz[1]<<endl;

	//	inline void matrix_dot_vector(int M, int N, float coef, float * A, float * x, float * y){

}

PositivePoolConnection::PositivePoolConnection(Layer * _x, Layer* _y, int _num, int _type){
	xlayer = _x;
	ylayer = _y;
	xnum = _num;
	ynum = _num;
	type = _type;
}

void PositivePoolConnection::computeZ(){
	// ylayer.y = max( ylayer.y, xlayer.y )
	// In this simplified version, we assume x is always greater than 0,
	// which is true for ReLU outputs
	for (int i = 0; i < this->xnum; i++){
		if ( xlayer->y[i] > ylayer->z[i]){
			ylayer->z[i] = xlayer->y[i];
		}
	}
	return;
}
void PositivePoolConnection::updateW(){
	// Actually, no weight is associated with a pooling layer

	// dE_dx = dE_dy if x = y
	// otherwise, dE_dx = 0
	for(int i = 0; i < this->xnum; i++){
		if( xlayer->y[i] == ylayer->y[i]  ){
			xlayer->dE_dy[i] += ylayer->dE_dy[i];
 		}
	}
}
///////////////////////////////////////////////////////
// blas wrapper
// y <- x
inline void icopy(int n, float * x, float * y){
	cblas_scopy(n, x, 1, y, 1);
	return;
}
// y += x
inline void iXpY(int n, float * x, float * y){
	cblas_saxpy(n, 1.0, x, 1, y, 1);
}
// y = coef * A * x
inline void matrix_dot_vector(int M, int N, float coef, float * A, enum CBLAS_TRANSPOSE trans, float * x, float * y){
	cblas_sgemv(CblasRowMajor, trans, M, N, coef, A, N, x, 1, 0, y, 1);
}
inline void selfplus_matrix_dot_vector(int M, int N, float coef, float * A, enum CBLAS_TRANSPOSE trans, float * x, float * y){
	cblas_sgemv(CblasRowMajor, trans, M, N, coef, A, N, x, 1, 1.0, y, 1);
}
// z = x .* y
inline void pointwise_dot(float * x, float * y, float *z, int n){
	// x .* y -> z
	// TODO use blas, multiplying with diagnal matrix
	for(int i = 0; i < n; i++)
		z[i] = x[i] * y[i];
}
// C += A * B, where A: <M x K>
//                  B: <K x N>
//                  C: <M x N>
inline void selfplus_matrix_dot_matrix(int M, int N, int K, float coef, float * A, enum CBLAS_TRANSPOSE transA,
								float * B, enum CBLAS_TRANSPOSE transB, float * C){
	cblas_sgemm(CblasRowMajor, transA, transB,
			M, N, K,
	   coef, A, K,
			B, N,
	1.0, C, N
	);
}

void GradDescent(int n, float learn_rate, float * grad, float * param){
	cblas_saxpy(n, -learn_rate, grad, 1, param, 1);

}
