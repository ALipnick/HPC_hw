#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  // each thread sums its section
  // then waits until the section before it is done with its last value
  // and adds that section's last value to all of it values
  // starting with it's last value so the next section can start sooner
  int num_of_threads = 16; //assume N is dividible by num_threads
  //                      so possible values are 1,2,4,5,8,10,16
  //otherwise would have to do some work to separate
  long k = floor(n/num_of_threads); //how many terms each thread takes
  #pragma omp parallel num_threads(num_of_threads)
  {
    int thread_num = omp_get_thread_num();
    if (thread_num == 0){
      prefix_sum[0] = 0;
    }
    else{
      prefix_sum[thread_num * k] = A[thread_num * k -1]; //set prefix_sum[thread_num * k] = A[thread_num*k]
    }
    for (long i = thread_num*k+1; i < (thread_num+1)*k; ++i) {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1]; //add A[i-1]
    }
    #pragma omp barrier //don't add results to next thread's values unil all are done

    if (thread_num == 0){
      for (long i = 1; i<num_of_threads; ++i){
            prefix_sum[(i+1)*k-1] = prefix_sum[(i+1)*k-1] + prefix_sum[i*k-1];
      }
    }
    #pragma omp barrier //after adding to last value we can do the  rest in parallel

    if (thread_num > 0){
         for (long i = thread_num*k; i < (thread_num+1)*k - 1; ++i) {
                  prefix_sum[i] = prefix_sum[i] + prefix_sum[thread_num*k-1];
         }
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}