//things to include
#include "utils.h"
#include <cmath>
#include <omp.h>

int main(int argc, char** argv){
	int max_iter = 5000;
	int res_factor = 10000;

	//int N = read_option<long>("-N", argc, argv);
	//int num_of_threads = read_option<long>("-T", argc, argv);
	int N = 500;
	int num_of_threads = 16;

	int max_num_threads = omp_get_max_threads();
	num_of_threads = std::min(num_of_threads, max_num_threads);
	
	double* u = (double*) malloc((N+2) * (N+2) * sizeof(double)); // N+2 x N+2 grid with 0 boundary
	double* u_new = (double*) malloc((N+2) * (N+2) * sizeof(double));	
	for (long i = 0; i < (N+2)*(N+2); i++){ //initialize values
		u[i] = 0;
		u_new[i] = 0;
	}
	
	//calc necessary info
	double h_2 = ((double)1)/((N+1)*(N+1)); // h^2
	double residual = N; //calc starting residual
	double tolerance = residual/res_factor; //calc tolerance to compare against
	printf("\ninitial residual = %f so tolerance = %f\n",residual, tolerance); 
	int iter;
	
	Timer t; // so we can time how long everything takes
	t.tic(); //start timer
	for (iter = 0; iter < max_iter; iter++) {//iterate
		//update u using jacobi
		//loop through and solve for u_new with u
		#ifdef _OPENMP
		#pragma omp parallel for collapse(2) num_threads(num_of_threads)
		#endif
		for (int i = 1; i<N+1; i++){
			for (int j = 1; j<N+1; j++){
				u_new[i+j*(N+2)] = 0.25*(h_2 
					+ u[(i-1)+(j)*(N+2)] + u[(i)+(j-1)*(N+2)]
					+ u[(i+1)+(j)*(N+2)] + u[(i)+(j+1)*(N+2)] );
			}
			
		}
		
		//now loop through and calc residual and set u to u_new
		residual = 0;
		#ifdef _OPENMP
		#pragma omp parallel for collapse(2) reduction(+:residual) num_threads(num_of_threads)
		#endif
		for (int i = 1; i<N+1; i++){
			for (int j = 1; j<N+1; j++){
				u[i+j*(N+2)] = u_new[i+j*(N+2)];
				residual += pow(1 -(((double)4*u_new[(i)+(j)*(N+2)]
					- u_new[(i-1)+(j)*(N+2)] - u_new[(i)+(j-1)*(N+2)]
				 	- u_new[(i+1)+(j)*(N+2)] - u_new[(i)+(j+1)*(N+2)])/h_2) ,2);
			}
		}
		residual = pow(residual,0.5);
		if (residual < tolerance){ //if below tolerance stop iterating
			break;
		}
	}
	double final_time = t.toc(); //store final time
	//print results
   	printf("after %d iterations the residual is %f\n", iter, residual);
   	printf("Total time was %fs with a total of %d threads\n",final_time,num_of_threads);
   	free(u_new); //free because malloc
   	free(u);
}