#!/bin/sh
#PBS -q SINGLE
#PBS -l select=1
#PBS -j oe
#PBS -N 1V_Ops
cd $PBS_O_WORKDIR 
python writeXY_Train_CV_Test.py


