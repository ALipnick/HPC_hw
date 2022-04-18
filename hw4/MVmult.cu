// $ nvcc -arch=sm_61 iMVmult.cu -o MVmult -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define THREADS_PER_BLOCK 1024

void MVmult(double* A, const double* x, const double* b, long n, long m){
  #pragma omp parallel for schedule (static)
  for (long i = 0; i < m; i++ ) {
    double sum = 0.0;
    for ( long j = 0; j < n; j++ ) {
      sum += A[i*n + j] * x[j];
    }
    b[i] = sum;
  }
}

__global__
void MVmult_kernel(double* A, const double* x, const double* b, long n, long m){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int eoq = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0.0;
  if (id )
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
  //use Ax = b for matrix vector mult
  long n = 1024;
  long m = 1024;
  double* A = (double*) malloc(n * m * sizeof(double));
  double* x = (double*) malloc(n * sizeof(double));
  double* b = (double*) malloc(m * sizeof(double));
  double* b_ref = (double*) malloc(m * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < n*m; i++) {
    A[i] = 1;
  }
  #pragma omp parallel for schedule(static)
  for (long i = 0; i< n; i++) {
    x[i] = 1;
  }
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < m; i++) {
    b_ref[i];
  }

  double tt = omp_get_wtime();
  vec_dot(b_ref, A, x, n,m);
  printf("CPU %f s\n", omp_get_wtime()-tt);

  double *A_d, *x_d, *b_d;
  cudaMalloc(&A_d, m*n*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&x_d, n*sizeof(double));
  cudaMalloc(&b_d, m*sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(A_d, A, n*m*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x, n*sizeof(double), cudaMemcpyHostToDevice);
  double ttinner = omp_get_wtime();
  MVmult_kernel<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(A_d, x_d, b_d, n,m);
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(b, b_d, sizeof(double), cudaMemcpyDeviceToHost);
  printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

  double err = 0;
  for (long i = 0; i < N; i++) err += fabs(b[i]-b_ref[i]);
  printf("Error = %f\n", err);

  cudaFree(A_d);
  cudaFree(x_d);
  cudaFree(b_d);

  free(A);
  free(x);
  free(b);
  free(b_ref);

  return 0;
}
