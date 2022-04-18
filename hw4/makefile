all: ip MVmult jacobi2D

ip: ip.cu
	nvcc -arch=sm_61 ip.cu -o ip -Xcompiler -fopenmp

MVmult: MVmult.cu
	nvcc -arch=sm_61 MVmult.cu -o MVmult -Xcompiler -fopenmp

jacobi2D: jacobi2D.cu
	nvcc -arch=sm_61 jacobi2D.cu -o jacobi2D -Xcompiler -fopenmp