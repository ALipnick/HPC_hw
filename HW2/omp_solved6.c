/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

//issue with sum and loop
//moved parallel region to inside dotprod with reduction. created global sum variable and added output to dotprod

float a[VECLEN], b[VECLEN];

float dotprod ()
{
int i,tid;
float sum;

tid = omp_get_thread_num();
#pragma omp parallel reduction(+:sum) //need reduction here
  for (i=0; i < VECLEN; i++)
    {
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }
    return sum; //return sum so value is accessable outside of function
}


int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

sum = dotprod(); //global sum variable needs to be set to output from dotprod func

printf("Sum = %f\n",sum);

}

