// $ nvcc -arch=sm_61 gpu03.cu -o gpu03 -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

double dot(const double* a, const double* b, long N){
  double dprod = 0.0;
  #pragma omp parallel
  #pragma omp for reduction(+:sum)
  for (long i = 0; i < N; i++) 
    dprod +=  = a[i] * b[i];
  
  return dprod;
}

__global__
double dot_kernel(const double* a, const double* b, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N){
  	int dprod = 0;
  	for (Long i = 0, i < N; i++)
  		dprod += a[i]*b[i];
  	
  	atomicAdd(c,dprod)
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
  long N = (1UL<<25); // 2^25

  double* x = (double*) malloc(N * sizeof(double));
  double* y = (double*) malloc(N * sizeof(double));
  double* z_ref = (double*) malloc(sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = i+2;
    y[i] = 1.0/(i+1);
  }

  double tt = omp_get_wtime();
  z_ref = dot(x, y, N);
  printf("CPU %f s\n", omp_get_wtime()-tt);

  double *x_d, *y_d;
  cudaMalloc(&x_d, N*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&y_d, N*sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  double ttinner = omp_get_wtime();
  double z = dot_kernel<<<N/1024,1024>>>(x_d, y_d, N);
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

  double err = 0;
  for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
  printf("Error = %f\n", err);

  cudaFree(x_d);
  cudaFree(y_d);

  free(x);
  free(y);
  free(z_ref);

  return 0;
}