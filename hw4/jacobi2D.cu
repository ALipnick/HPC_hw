// $ nvcc -arch=sm_61 jacobi2D.cu -o jacobi2D -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler

#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <random>

#define THREADS_PER_BLOCK 32 //32 = sqrt(1024) which is max size

void jacobi2D_step(double* u_new, double* u, double h, long N) {
  #pragma omp parallel for collapse(2)
  for (int i = 1; i<N+1; i++){
  	for (int j = 1; j<N+1; j++){
  		u_new[i+j*(N+2)] = 0.25*(h*h 
			+ u[(i-1)+(j)*(N+2)] + u[(i)+(j-1)*(N+2)]
			+ u[(i+1)+(j)*(N+2)] + u[(i)+(j+1)*(N+2)] );

			}
		}
  #pragma omp parallel for collapse(2)
  for (int i = 1; i<N+1; i++){
  	for (int j = 1; j<N+1; j++){
  		u[i+j*(N+2)] = u_new[i+j*(N+2)];
			}
		}
}


__global__ 
void jacobi2D_step_kernel(double* u_new, double* u, double h, long N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ( (i > 0) && (i < N+1) && (j > 0) && (j < N+1) ) {
  	u_new[i+j*(N+2)] = 0.25*(h*h 
			+ u[(i-1)+(j)*(N+2)] + u[(i)+(j-1)*(N+2)]
			+ u[(i+1)+(j)*(N+2)] + u[(i)+(j+1)*(N+2)] );
  }
  __syncthreads();
  u[i+j*(N+2)] = u_new[i+j*(N+2)];  	
}

//decided to just run for full maximum iterations because just comparing two ways
void jacobi2D(double* u_new, double* u, double h, long N, long max_iter) {
	for (long iter = 0; iter < max_iter; iter++) {
		jacobi2D_step(u_new,  u,  h, N);
	}
}


void jacobi2D_cuda(double* u_new_d, double* u_d, double h, long N, long max_iter) {
  dim3 GridDim(N/THREADS_PER_BLOCK, N/THREADS_PER_BLOCK);
  dim3 BlockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  for (long iter = 0; iter < max_iter; iter++) {
  	jacobi2D_step_kernel<<<GridDim, BlockDim>>>(u_new_d, u_d, h, N);
	}
}

double calc_residual(double* u, double h, long N) {
	double residual = 0;
	#pragma omp parallel for collapse(2) reduction(+:residual)
	for (int i = 1; i<N+1; i++){
		for (int j = 1; j<N+1; j++){
			residual += pow(1 -(((double)4*u[(i)+(j)*(N+2)]
				- u[(i-1)+(j)*(N+2)] - u[(i)+(j-1)*(N+2)]
				- u[(i+1)+(j)*(N+2)] - u[(i)+(j+1)*(N+2)])/(h*h)) ,2);
			}
		}
	residual = pow(residual,0.5);
	return residual; 
}


void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}


int main(void) {
  int max_iter = 1000;
  int N = THREADS_PER_BLOCK*16;
 
  double* u = (double*) malloc((N+2) * (N+2) * sizeof(double)); // N+2 x N+2 grid with 0 boundary
  double* u_new = (double*) malloc((N+2) * (N+2) * sizeof(double));
  double* u_ref = (double*) malloc((N+2) * (N+2) * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (N+2)*(N+2); i++){ //initialize values
		u[i] = 0;
	}

  //calc necessary info
  double h = ((double)1)/((N+1)*(N+1)); // h^2
  double residual = calc_residual( u, h, N);
  printf("residual = %f\n",residual);
  
  double tt = omp_get_wtime();
  jacobi2D(u_new, u, h, N,max_iter);

  #pragma omp parallel for schedule(static)  
  for ( long i = 0; i < (N+2)*(N+2); i++ ) {
   u_ref[i] = u[i];
  }
  printf("CPU %f s\n", omp_get_wtime()-tt);
  residual = calc_residual( u_ref, h, N);
  printf("residual = %f\n",residual);

  double *u_d, *u_new_d;
  cudaMalloc(&u_d, (N+2) * (N+2) *sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&u_new_d, (N+2) * (N+2) *sizeof(double));

  // reset u values
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (N+2)*(N+2); i++){ //initialize values
		u[i] = 0;
  }

  tt = omp_get_wtime();
  cudaMemcpy(u_d, u, (N+2) * (N+2) *sizeof(double), cudaMemcpyHostToDevice);
  double ttinner = omp_get_wtime();
  jacobi2D_cuda(u_new_d, u_d, h, N, max_iter);
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(u, u_d, (N+2) * (N+2) * sizeof(double), cudaMemcpyDeviceToHost);
  printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);
  
  residual = calc_residual(u, h, N);
  printf("residual = %f\n",residual);

  double err = 0;
  for (long i = 0; i < (N+2) * (N+2); i++) {
  	err += fabs(u_ref[i]-u[i]);
  }
  printf("Error = %f\n", err);

  cudaFree(u_d);
  cudaFree(u_new_d);

  free(u);
  free(u_new);
  free(u_ref);

  return 0;
}