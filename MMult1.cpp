// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 16

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void matrix_mult(double *A, double *B, double *C) {
  // C + A*B
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  // TODO: See instructions below
//   for ( long i = 0; i < m; i+= BLOCK_SIZE) {
//     for (long j = 0; j < n; j+= BLOCK_SIZE) {
//       //I = ((i-1)*b + 1): i*b
//       //J = ((j-1)*b + 1): j*b
//       //double C_ij = c[I,J]
//       double C_ij[BLOCK_SIZE][BLOCK_SIZE]
//       for (long p = 0; p < k; p+= BLOCK_SIZE) {
//         //K = ((k-1)*b+1):k*b
//         double A_ip = a[I,K]
//         double B_pj = b[K,j]
//         C_ij = matrix_mult(A_ip, B_pj, C_ij);
//         c[I,J] = C_ij
//       }
//     }
//   }
// }
  for (int ii = 0; ii < m; ii+=BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj+=BLOCK_SIZE) {
      for (int pp = 0; pp < k; pp+=BLOCK_SIZE) {
        for (int i = ii; i < ii+BLOCK_SIZE; i++) {
          for (int j = jj; j < jj+BLOCK_SIZE; j++) {
            for (int p = pp; p < pp+BLOCK_SIZE; p++) {
              c[i+j*m] = c[i+j*m] + a[i+p*m] * b[p+j*k];
            }
          }
        }
      }
    }
  }
}


int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = 2*(m*n*1e-9)*k*NREPEATS/time; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth = 0; // TODO: calculate from m, n, k, NREPEATS, time
    printf("%10ld %10f %10f %10f", p, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
