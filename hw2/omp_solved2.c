/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

//tid was shared to it would get overwritten by each thread, each thread needs its own private copy 
//made tid private
//total was different each time because race condition
//got rid of nested parallel region

int main (int argc, char *argv[]) 
{
int nthreads, i, tid; 
float total;

/*** Spawn parallel region ***/
#pragma omp parallel private(tid) //make tid private for each thread
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  total = 0.0;
  //#pragma omp for schedule(dynamic, 10) //doing this messes up the total
  for (i=0; i<1000000; i++) 
    total = total + i*1.0;
  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
