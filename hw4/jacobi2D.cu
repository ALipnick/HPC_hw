// $ nvcc -arch=sm_61 jacobi2D.cu -o jacobi2D -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler

#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <random>

#define BLOCK_SIZE 32

void jacobi2D_step(double* u_new, const double* u, double h, long N) {
  #pragma omp parallel for collapse(2)
  for (int i = 1; i<N+1; i++){
  	for (int j = 1; j<N+1; j++){
  		u_new[i+j*(N+2)] = 0.25*(h*h 
			+ u[(i-1)+(j)*(N+2)] + u[(i)+(j-1)*(N+2)]
			+ u[(i+1)+(j)*(N+2)] + u[(i)+(j+1)*(N+2)] );
			}
		}
}


__global__ 
void jacobi2D_step_kernel(double* u_new, const double* u, double h, long N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ( (i > 0) && (i < N+1) && (j > 0) && (j < N+1) ) {
  	u_new[i+j*(N+2)] = 0.25*(h*h 
			+ u[(i-1)+(j)*(N+2)] + u[(i)+(j-1)*(N+2)]
			+ u[(i+1)+(j)*(N+2)] + u[(i)+(j+1)*(N+2)] );
  }
}

//decided to just run for full maximum iterations because just comparing two ways
// void jacobi2D(double* u_new, double* u,  double* temp,double h, long N, long max_iter) {
// 	for (long iter = 0; iter < max_iter; iter++) {
// 		jacobi2D_step(u_new,  u,  h, N);
// 		*temp = *u;
//     	*u = *u_new;
//     	*u_new = *temp;
// 	}
// }


// void jacobi2D_cuda(double* u_new, double* u, double* temp, double h, long N, long max_iter) {
//   dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE);
//   dim3 GridDim(N/BLOCK_SIZE, N/BLOCK_SIZE);
//   for (long iter = 0; iter < max_iter; iter++) {
//   	jacobi2D_step_kernel<<<GridDim, BlockDim>>>(u_new,  u,  h, N);
//   	*temp = *u;
//     *u = *u_new;
//     *u_new = *temp;
// 	}
// }



void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}


int main(void) {
  int max_iter = 1000;
  int N = 500;
 
  double* u = (double*) malloc((N+2) * (N+2) * sizeof(double)); // N+2 x N+2 grid with 0 boundary
  double* u_new = (double*) malloc((N+2) * (N+2) * sizeof(double));
  double* temp = (double*) malloc((N+2) * (N+2) * sizeof(double));
  double* u_ref = (double*) malloc((N+2) * (N+2) * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (N+2)*(N+2); i++){ //initialize values
		u[i] = 0;
	}
  //calc necessary info
  double h = ((double)1)/((N+1)*(N+1)); // h^2
  


  double tt = omp_get_wtime();
  //jacobi2D(u_new, u, temp, h, N,max_iter);
  for (long iter = 0; iter < max_iter; iter++) {
	jacobi2D_step(u_new,  u,  h, N);
	*temp = *u;
    *u = *u_new;
    *u_new = *temp;
}
#pragma omp parallel for schedule(static)  
for ( long i = 0; i < (N+2)*(N+2); i++ ) {
   u_ref[i] = u[i];
 }
  printf("CPU %f s\n", omp_get_wtime()-tt);

  double *u_d, *u_new_d;
  cudaMalloc(&u_d, (N+2) * (N+2) *sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&u_new_d, (N+2) * (N+2) *sizeof(double));

  // reset u values
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (N+2)*(N+2); i++){ //initialize values
		u[i] = 0;
		u_new[i] = 0;
  }

  tt = omp_get_wtime();
  cudaMemcpy(u_d, u, (N+2) * (N+2) *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(u_new_d, u_new, (N+2) * (N+2) * sizeof(double), cudaMemcpyHostToDevice);

  double ttinner = omp_get_wtime();
  //jacobi2D_cuda(u_new, u, temp, h, N,max_iter);
  dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 GridDim(N/BLOCK_SIZE, N/BLOCK_SIZE);
  for (long iter = 0; iter < max_iter; iter++) {
  	jacobi2D_step_kernel<<<GridDim, BlockDim>>>(u_new,  u,  h, N);
  	*temp = *u;
    *u = *u_new;
    *u_new = *temp;
}
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(u_new, u_new_d, (N+2) * (N+2) * sizeof(double), cudaMemcpyDeviceToHost);
  printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

  double err = 0;
  for (long i = 0; i < (N+2) * (N+2); i++) err += fabs(u_ref[i]-u_new[i]);
  printf("Error = %f\n", err);

  //printf("b[0] = %f\n", b[0]);

  cudaFree(u_d);
  cudaFree(u_new_d);

  free(u);
  free(u_new);
  free(u_ref);
  free(temp);

  return 0;
}