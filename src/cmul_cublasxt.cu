#include <stdio.h>
#include "cublasXt.h"
#include <curand.h>

void fill(double* &x, long m, long n, double val) {
  x = new double[m * n];
  for (long i = 0; i < m; ++i) {
    for (long j = 0; j < n; ++j) {
      x[i * n + j] = val;
    }
  }
}

int main() {
  cublasXtHandle_t xt_;
  
  if(cublasXtCreate(&xt_) != CUBLAS_STATUS_SUCCESS) {
    printf("handle create fail\n"); 
    return 1;
  }
  int devices[1] = { 0 };  // add this line
  if(cublasXtDeviceSelect(xt_, 1, devices) != CUBLAS_STATUS_SUCCESS) {
    printf("set devices fail\n");
    return 1;
  } // add this line


  double *A, *B, *C;
  long m = 10, n = 10, k = 20;

  fill(A, m, k, 0.2);
  fill(B, k, n, 0.3);
  fill(C, m, n, 0.0);

  double alpha = 1.0;
  double beta = 0.0;

  cublasXtDgemm(xt_, CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k, &alpha, A, m, B, k, &beta, C, m
  );

  cudaDeviceSynchronize();

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      printf ("%lf ", C[i *n + j]);
    }
    printf ("\n");
  }

  cublasXtDestroy(xt_);
  return 0;
}
