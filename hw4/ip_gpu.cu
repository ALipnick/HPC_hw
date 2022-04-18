// $ nvcc -arch=sm_61 ip_gpu.cu -o gpu03 -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define THREADS_PER_BLOCK 1024

void vec_dot(double* c, const double* a, const double* b, long N){
  double sum = 0.0;
  #pragma omp parallel
  #pragma omp for reduction(+:sum) //reduction so can do sum in parallel
  for (long i = 0; i < N; i++) {
    sum += a[i]*b[i];
  }
  *c = sum;
}

__global__
void vec_dot_kernel(double* c, const double* a, const double* b, long N){
  __shared__ double prods[THREADS_PER_BLOCK]; //shared var for all producs
  int idx = blockIdx.x * blockDim.x + threadIdx.x; //get thread id
  prods[threadIdx.x] = a[idx] * b[idx]; //each thread calcs a product
  __syncthreads(); //sync to make sure all relavent prods are calculated
  if (0 == threadIdx.x ) {
  	double sum = 0.0;
  	for ( int i = 0; i < THREADS_PER_BLOCK; i++ ) {
  		sum += prods[i];
  	}
  	atomicAdd(c,sum);
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
  double* z = (double*) malloc(sizeof(double));
  double* z_ref = (double*) malloc(sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = 1;
    y[i] = 1;
  }
  *z = 0.0;
  *z_ref = 0.0;
  double tt = omp_get_wtime();
  vec_dot(z_ref, x, y, N);
  printf("CPU %f s\n", omp_get_wtime()-tt);

  double *x_d, *y_d, *z_d;
  cudaMalloc(&x_d, N*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&y_d, N*sizeof(double));
  cudaMalloc(&z_d, sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  double ttinner = omp_get_wtime();
  vec_dot_kernel<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(z_d, x_d, y_d, N);
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(z, z_d, sizeof(double), cudaMemcpyDeviceToHost);
  printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

  printf("z_ref = %f\n", *z_ref );
  printf("z = %f\n", *z );
  printf("Error = %f\n", fabs(*z - *z_ref));

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);

  free(x);
  free(y);
  free(z);
  free(z_ref);

  return 0;
}
