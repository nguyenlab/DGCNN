################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/FFNN.cpp \
../src/activation.cpp \
../src/global.cpp \
../src/main.cpp \
../src/predict.cpp \
../src/read_data.cpp \
../src/test.cpp 

O_SRCS += \
../src/FFNN.o \
../src/activation.o \
../src/global.o \
../src/main.o \
../src/predict.o \
../src/read_data.o \
../src/test.o 

OBJS += \
./src/FFNN.o \
./src/activation.o \
./src/global.o \
./src/main.o \
./src/predict.o \
./src/read_data.o \
./src/test.o 

CPP_DEPS += \
./src/FFNN.d \
./src/activation.d \
./src/global.d \
./src/main.d \
./src/predict.d \
./src/read_data.d \
./src/test.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/seke/Workspace/CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


