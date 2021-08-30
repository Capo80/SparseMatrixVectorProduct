#include<stdio.h>
#include<stdlib.h>

extern "C" {
#include "../include/formats.h"
#include "../include/product_gpu.h"
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void print_from_gpu(void) {
    printf("Hello World! from thread [%d,%d] \
        From device\n", threadIdx.x,blockIdx.x);
}

extern "C"
double cuda_product_csr(csr_matrix* matrix, double* array, double* result) {

    printf("Hello World from host!\n");
    print_from_gpu<<<1,1>>>();
    cudaError_t cudaerr = cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    return 0.0;

}

extern "C"
double cuda_product_ellpack(ellpack_matrix* matrix, double* array, double* result) {

	return 0.0;

}