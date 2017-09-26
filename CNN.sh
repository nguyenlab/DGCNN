g++ -I../CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/FFNN.d" -MT"src/FFNN.d" -o "src/FFNN.o" "src/FFNN.cpp"
g++ -I../CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/activation.d" -MT"src/activation.d" -o "src/activation.o" "src/activation.cpp"
g++ -I../CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/global.d" -MT"src/global.d" -o "src/global.o" "src/global.cpp"
g++ -I../CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/main.d" -MT"src/main.d" -o "src/main.o" "src/main.cpp"
g++ -I../CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/read_data.d" -MT"src/read_data.d" -o "src/read_data.o" "src/read_data.cpp"
g++ -I../CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/predict.d" -MT"src/predict.d" -o "src/predict.o" "src/predict.cpp"
g++ -o "CNNtest" ./src/FFNN.o ./src/activation.o ./src/global.o ./src/main.o ./src/read_data.o ./src/predict.o ../CBLAS/lib/cblas_LINUX.a ../BLAS/BLAS-3.6.0/blas_LINUX.a -lgfortran -lpthread
