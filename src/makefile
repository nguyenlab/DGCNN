CPP      = g++.exe
CC       = gcc.exe
WINDRES  = windres.exe
OBJ      = activation.o FFNN.o global.o main.o predict.o read_data.o file_io.o DESolver.o main_cnn.o main_decnn.o DE_CNN.o test_loadnet.o loss.o initparams.o
LINKOBJ  = activation.o FFNN.o global.o main.o predict.o read_data.o file_io.o DESolver.o main_cnn.o main_decnn.o DE_CNN.o test_loadnet.o loss.o initparams.o
LIBS     = -L"C:/Dev-Cpp/MinGW64/lib" -L"C:/Dev-Cpp/MinGW64/x86_64-w64-mingw32/lib" -static-libgcc
INCS     = -I"C:/Dev-Cpp/MinGW64/include" -I"C:/Dev-Cpp/MinGW64/x86_64-w64-mingw32/include" -I"C:/Dev-Cpp/MinGW64/lib/gcc/x86_64-w64-mingw32/4.9.2/include" -I"C:/BLAS/CBLAS/include"
CXXINCS  = -I"C:/Dev-Cpp/MinGW64/include" -I"C:/Dev-Cpp/MinGW64/x86_64-w64-mingw32/include" -I"C:/Dev-Cpp/MinGW64/lib/gcc/x86_64-w64-mingw32/4.9.2/include" -I"C:/Dev-Cpp/MinGW64/lib/gcc/x86_64-w64-mingw32/4.9.2/include/c++" -I"C:/BLAS/CBLAS/include"
BIN      = CNN.exe
CXXFLAGS = $(CXXINCS) 
CFLAGS   = $(INCS) 
RM       = rm.exe -f

.PHONY: all all-before all-after clean clean-custom

all: all-before $(BIN) all-after

clean: clean-custom
	${RM} $(OBJ) $(BIN)

$(BIN): $(OBJ)
	$(CPP) $(LINKOBJ) -o $(BIN) $(LIBS)

activation.o: activation.cpp
	$(CPP) -c activation.cpp -o activation.o $(CXXFLAGS)

FFNN.o: FFNN.cpp
	$(CPP) -c FFNN.cpp -o FFNN.o $(CXXFLAGS)

global.o: global.cpp
	$(CPP) -c global.cpp -o global.o $(CXXFLAGS)

main.o: main.cpp
	$(CPP) -c main.cpp -o main.o $(CXXFLAGS)

predict.o: predict.cpp
	$(CPP) -c predict.cpp -o predict.o $(CXXFLAGS)

read_data.o: read_data.cpp
	$(CPP) -c read_data.cpp -o read_data.o $(CXXFLAGS)

file_io.o: file_io.cpp
	$(CPP) -c file_io.cpp -o file_io.o $(CXXFLAGS)

DESolver.o: DESolver.cpp
	$(CPP) -c DESolver.cpp -o DESolver.o $(CXXFLAGS)

main_cnn.o: main_cnn.cpp
	$(CPP) -c main_cnn.cpp -o main_cnn.o $(CXXFLAGS)

main_decnn.o: main_decnn.cpp
	$(CPP) -c main_decnn.cpp -o main_decnn.o $(CXXFLAGS)

DE_CNN.o: DE_CNN.cpp
	$(CPP) -c DE_CNN.cpp -o DE_CNN.o $(CXXFLAGS)

test_loadnet.o: test_loadnet.cpp
	$(CPP) -c test_loadnet.cpp -o test_loadnet.o $(CXXFLAGS)

loss.o: loss.cpp
	$(CPP) -c loss.cpp -o loss.o $(CXXFLAGS)

initparams.o: initparams.cpp
	$(CPP) -c initparams.cpp -o initparams.o $(CXXFLAGS)