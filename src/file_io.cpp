#include <stdlib.h>
#include <stdio.h> 
#include <string.h>
#include "initparams.h"
#include"FFNN.h"
#include"activation.h"
#define sizeofint 4
#define sizeoffloat 4
#define sizeofchar  1
int max_line_len = 1024;
char* readline(FILE *input);
char* settingfile;
const int nP = 10;
void ReadNetFromBuf( char *& cursor, FILE *fp);
var* getparameters()
{
    FILE *fp;
    fp=fopen(settingfile,"r");
    if(fp==NULL)
	{
		fprintf(stderr,"can't open file dataset information file\n");
		return NULL;
	}
    char * line;
    char delims[10] =":";
    var** varlist =(var**)malloc(sizeof(var*));
	while((line=readline(fp))!=NULL)
	{		
        if(line[0]=='/')
            continue; 
        var* newvar = getvarfromstring(line, delims);
		Insert(varlist, newvar);    
	}
    fclose(fp);
    return *varlist;
}
char* readline(FILE *input)
{
	int len;
	char *line = (char*)malloc(200 * sizeof(char));
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	len = (int)strlen(line);
	if(len>0)
	{
		if(line[len-1]=='\n')
			line[len-1] ='\0';
	}
	return line;
}
void getNetInfor(char * buff, int numNet, const char* out)
{
	char * ptr = buff;
	FILE * fp = fopen(out, "w");
	for (int i=0; i< numNet; i++)
	{
		ReadNetFromBuf(ptr, fp);
	}
	fclose(fp);	
}
void ReadNetFromBuf( char *& cursor, FILE *fp){
	int numlay, numcon;
//	int len;
	Layer ** one_net;
	numlay = *((int *) cursor);     cursor += sizeofint;
	numcon = *((int *) cursor);     cursor += sizeofint;
	fprintf(fp,"%d - %d\n",numlay, numcon);

//	len = numlay;
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
				one_net[i] = new Layer( "relu", numUnit, bidx, numUp, numDown, tanh, tanhPrime);
			}
			else if( activation == 'c' ){ // drop and relu
				one_net[i] = new Layer( "con", numUnit, bidx, numUp, numDown, tanh, tanhPrime);

			}
			else if( activation == 'v' ){ // recursive
				one_net[i] = new Layer( "recursive", numUnit, bidx, numUp, numDown, tanh, tanhPrime);
			}
			else if( activation == 'a' ){ // tanh
				one_net[i] = new Layer( "ae", numUnit, bidx, numUp, numDown, tanh, tanhPrime);
			}
			else if( activation == 'b' ){ // tanh
							one_net[i] = new Layer( "combination", numUnit, bidx, numUp, numDown, tanh, tanhPrime);
						}
			else if( activation == 's') { // softmax
				one_net[i] = new Layer( "softmax", numUnit, bidx, numUp, numDown, Softmax, dummy);
			}
			else if( activation == 'e'){ // tanh
				one_net[i] = new Layer( "embed", numUnit, bidx, numUp, numDown, tanh, tanhPrime);
				delete one_net[i]->z;
				one_net[i]->z = NULL;
			}
			else if( activation == 'h'){ // dropout and relu
				one_net[i] = new Layer( "hidden", numUnit, bidx, numUp, numDown, tanh, tanhPrime);
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

