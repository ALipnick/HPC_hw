// $ nvcc -arch=sm_61 MVmult.cu -o MVmult -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <random>


// we will assume that m and n are divisible by THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#define m (THREADS_PER_BLOCK * 10)
#define n (THREADS_PER_BLOCK * 5)

void MVmult(double* b, const double* A, const double* x) {
  #pragma omp parallel for schedule(static)
  for ( long i = 0; i < m; i++ ) {
    double sum = 0;
    for ( long j = 0; j < n; j++ ) {
      sum += A[i*n + j] * x[j];
    }
    b[i] = sum;
  }
}


__global__ 
void MVmult_kernel(double* b, const double* A, const double* x) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  double sum = 0.0;
  if ( idx < m ) {
    for ( long i = 0; i < n; i++ ) {
      sum += A[idx*n + i] * x[i];
    }
    b[idx] = sum;
  }
}


// second idea where all products are calculated in parallel, this ended up being slower.
__global__ 
void MVmult_kernel2(double* b, const double* A, const double* x) {
  __shared__ double prods[THREADS_PER_BLOCK]; //shared var for all products
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int j = idx%n; //column
  int i = idx / n; //row
  prods[threadIdx.x] = A[idx]*x[j];
  __syncthreads(); //sync to make sure all relavent prods are calculated
  if (0 == threadIdx.x ) {
    double sum = 0;
    for ( int k = 0; k < THREADS_PER_BLOCK; k++ ) {
      sum += prods[k];
    }
    atomicAdd(&b[i],sum);
  }
}


void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}


int main(void) {
  // notation: Ax = b with A m by n
  double* A = (double*) malloc( m * n * sizeof(double));
  double* x = (double*) malloc( n * sizeof(double));
  double* b = (double*) malloc( m * sizeof(double));
  double* b_ref = (double*) malloc( m * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < m*n; i++) {
    A[i]   = rand() % 10 -10;
  }
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < n; i++) {
    x[i]   = rand() % 10 -10;
  }

  double tt = omp_get_wtime();
  MVmult(b_ref, A, x);
  printf("CPU %f s\n", omp_get_wtime()-tt);

  double *A_d, *x_d, *b_d;
  cudaMalloc(&A_d, m * n *sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&x_d, n *sizeof(double));
  cudaMalloc(&b_d, m *sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(A_d, A, m * n *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x, n *sizeof(double), cudaMemcpyHostToDevice);

  double ttinner = omp_get_wtime();
  MVmult_kernel<<< m/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(b_d, A_d, x_d);
  // MVmult_kernel2<<<n*m/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(b_d, A_d, x_d);
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(b, b_d, m *sizeof(double), cudaMemcpyDeviceToHost);
  printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

  double err = 0;
  for (long i = 0; i < m; i++) err += fabs(b[i]-b_ref[i]);
  printf("Error = %f\n", err);

  //printf("b[0] = %f\n", b[0]);

  cudaFree(A_d);
  cudaFree(x_d);
  cudaFree(b_d);

  free(A);
  free(x);
  free(b);
  free(b_ref);

  return 0;
}
