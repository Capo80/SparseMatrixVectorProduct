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

//dumb
template <unsigned int THD> __global__ void product_one_row_one_block_csr(csr_matrix* matrix, double* array, double* result) {

   //block sum
	__shared__ double block_sum[THD];
   
   int tid = threadIdx.x;
   int row = blockIdx.x;
   int i;

   //check row
   if (row < matrix->M) {

      //check thread
      int limit = matrix->irp[row+1];
      int row_elements = matrix->irp[row] - limit;
      if (tid < row_elements) {
         
         // block sums row
         for (i = matrix->irp[row] + tid; i < limit; i += blockDim.x)
            block_sum[tid] += matrix->as[i] * array[matrix->ja[i]];

         //wait for all to finish
         __syncthreads();

         //block reduction (max efficency)
         if (THD >= 1024) { if (tid < 512) { block_sum[tid] += block_sum[tid + 512]; } __syncthreads(); }

         if (THD >= 512) { if (tid < 256) { block_sum[tid] += block_sum[tid + 256]; } __syncthreads(); }

         if (THD >= 256) { if (tid < 128) { block_sum[tid] += block_sum[tid + 128]; } __syncthreads(); }

         if (THD >= 128) { if (tid < 64) { block_sum[tid] += block_sum[tid + 64]; } __syncthreads(); }

         if (THD >= 64) { if (tid < 32) { block_sum[tid] += block_sum[tid + 32]; } __syncthreads(); }

         //last warp
         if (tid < 32) { block_sum[tid] += block_sum[tid + 32]; }
         if (tid < 16) { block_sum[tid] += block_sum[tid + 16]; }
         if (tid < 8) { block_sum[tid] += block_sum[tid + 8]; }
         if (tid < 4) { block_sum[tid] += block_sum[tid + 4]; }
         if (tid < 2) { block_sum[tid] += block_sum[tid + 2]; }
         if (tid < 1) { block_sum[tid] += block_sum[tid + 1]; }

         //reduction over, save result in final array
         if (tid == 0)
            result[row] = block_sum[tid];
      }

   }

}

__global__ void product_one_row_one_warp_csr(csr_matrix* matrix, double* array, double* result) {

}

extern "C"
double cuda_product_csr(csr_matrix* matrix, double* array, double* result) {

   printf("block_size: %d\n", matrix->M);

   csr_matrix* matrix_gpu;
   csr_matrix* matrix_inter = (csr_matrix*) malloc(sizeof(csr_matrix));
   double* array_gpu;
   double* result_gpu;
   //alloc arguments
   printf("allocation finished\n");
   gpuErrchk( cudaMalloc((void**) &matrix_gpu, sizeof(csr_matrix)) );
   gpuErrchk( cudaMalloc((void**) &matrix_inter->irp, sizeof(int)*matrix->M) );
   gpuErrchk( cudaMalloc((void**) &matrix_inter->ja, sizeof(int)*matrix->nz) );
   gpuErrchk( cudaMalloc((void**) &matrix_inter->as, sizeof(double)*matrix->nz) );
   gpuErrchk( cudaMalloc((void**) &array_gpu, sizeof(double)*matrix->M) );
   gpuErrchk( cudaMalloc((void**) &result_gpu, sizeof(double)*matrix->M) );

   printf("allocation finished\n");

   //copy arguments
   gpuErrchk( cudaMemcpy(matrix_gpu, matrix_inter, sizeof(csr_matrix), cudaMemcpyHostToDevice) );
   gpuErrchk( cudaMemcpy(array_gpu, array, sizeof(double)*matrix->M, cudaMemcpyHostToDevice) );
   gpuErrchk( cudaMemcpy(result_gpu, matrix, sizeof(double)*matrix->M, cudaMemcpyHostToDevice) );
   
   printf("copy finished\n");

   product_one_row_one_block_csr<1024><<<10,1024, sizeof(double)*1024>>>(matrix_gpu, array_gpu, result_gpu);      
   cudaError_t cudaerr = cudaDeviceSynchronize();
   gpuErrchk( cudaPeekAtLastError() );
   gpuErrchk( cudaDeviceSynchronize() );

   //free everything
   gpuErrchk( cudaFree(matrix_gpu) );
   gpuErrchk( cudaFree(matrix_gpu->irp) );
   gpuErrchk( cudaFree(matrix_gpu->ja) );
   gpuErrchk( cudaFree(matrix_gpu->as) );
   gpuErrchk( cudaFree(result_gpu) );
   gpuErrchk( cudaFree(array_gpu) );

   free(matrix_inter);
   return 0.0;

}

extern "C"
double cuda_product_ellpack(ellpack_matrix* matrix, double* array, double* result) {

	return 0.0;

}