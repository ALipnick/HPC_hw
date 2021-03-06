/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

//a was too big to be a doulbe array and have each thread make a copy so malloc'ed it and put in loop to prevent error, also added line to free it

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid)
  { //change a to local variable because each will have its own copy
  double* a = (double*) malloc(N * N * sizeof(double)); // m x k

  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i+N*j] = tid + i + j;

  /* For confirmation */
  printf("Thread %d done. Last element= %f\n",tid,a[(N-1)+N*(N-1)]);

  free(a); //free because malloc

  }  /* All threads join master thread and disband */

}

