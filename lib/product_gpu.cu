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

// ######################################## csr ############################################

/*
template <unsigned int THD> __global__ void product_one_row_one_block_csr(int M, int* irp, int* ja, double* as, double* array, double* result) {

   //block sum
	__shared__ double block_sum[THD];
   
   int tid = threadIdx.x;
   int row = blockIdx.x;
   int i;

   block_sum[tid] = 0;
   //check row
   if (row < M) {

      //check thread
      int limit = irp[row+1];

      // block sums row
      double sum = 0;
      for (i = irp[row] + tid; i < limit; i += blockDim.x)
         sum += as[i] * array[ja[i]];
      
      block_sum[tid] = sum;
      //wait for all to finish
      __syncthreads();

      //block reduction (max efficency)
      if (THD >= 1024) { if (tid < 512) { block_sum[tid] += block_sum[tid + 512]; } __syncthreads(); }

      if (THD >= 512) { if (tid < 256) { block_sum[tid] += block_sum[tid + 256]; } __syncthreads(); }

      if (THD >= 256) { if (tid < 128) { block_sum[tid] += block_sum[tid + 128]; } __syncthreads(); }

      if (THD >= 128) { if (tid < 64) { block_sum[tid] += block_sum[tid + 64]; } __syncthreads(); }

      if (THD >= 64) { if (tid < 32) { block_sum[tid] += block_sum[tid + 32]; } __syncthreads(); }

      //last warp
      if (tid < 16) { block_sum[tid] += block_sum[tid + 16]; __syncwarp(); }
      if (tid < 8) { block_sum[tid] += block_sum[tid + 8]; __syncwarp(); }
      if (tid < 4) { block_sum[tid] += block_sum[tid + 4]; __syncwarp(); }
      if (tid < 2) { block_sum[tid] += block_sum[tid + 2]; __syncwarp(); }
      if (tid < 1) { block_sum[tid] += block_sum[tid + 1]; __syncwarp(); }

      //reduction over, save result in final array
      if (tid == 0)
         result[row] = block_sum[tid];

   }

}
*/
__global__ void product_one_row_one_warp_csr(unsigned int M, unsigned int* __restrict__ irp, unsigned int* __restrict__ ja, double*  __restrict__ as, double* __restrict__ array, double* __restrict__ result) {

   __shared__ double warp_sum[BLOCK_SIZE];

   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   int b_tid = threadIdx.x;
   int warp_id = tid % WARP_SIZE;
   int row = tid / WARP_SIZE;
   int i;

   warp_sum[b_tid] = 0;
   if (row < M) {

      // warp sums row
      int limit = irp[row+1];
      double sum = 0;
      for (i = irp[row] + warp_id; i < limit; i += WARP_SIZE) {
         sum += as[i] * array[ja[i]];
      }

      warp_sum[b_tid] = sum;
      
      __syncwarp();

      // we have only one warp, no need to sync entire block
      if (warp_id < 16) { warp_sum[b_tid] += warp_sum[b_tid + 16]; __syncwarp();}
      
      if (warp_id < 8) { warp_sum[b_tid] += warp_sum[b_tid + 8]; __syncwarp();}

      if (warp_id < 4) { warp_sum[b_tid] += warp_sum[b_tid + 4]; __syncwarp();}
      
      if (warp_id < 2) { warp_sum[b_tid] += warp_sum[b_tid + 2]; __syncwarp();}
      
      if (warp_id < 1) { warp_sum[b_tid] += warp_sum[b_tid + 1]; __syncwarp();}
      
      //reduction over, save result in final array
      if (warp_id == 0) {
         result[row] = warp_sum[b_tid];
      }

   }


}
/*
template <unsigned int N> __global__ void product_one_row_N_warp_csr(unsigned int M, unsigned int* __restrict__ irp, unsigned int* __restrict__ ja, double*  __restrict__ as, double* __restrict__ array, double* __restrict__ result) {

   __shared__ double warp_sum[BLOCK_SIZE];

   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   int b_tid = threadIdx.x;
   int warp_id = tid % (WARP_SIZE*2);
   int row = tid / (WARP_SIZE*N);
   int i;

   warp_sum[b_tid] = 0;
   if (row < M) {

      // warp sums row
      int limit = irp[row+1];
      double sum = 0;
      for (i = irp[row] + warp_id; i < limit; i += WARP_SIZE) {
         sum += as[i] * array[ja[i]];
      }

      warp_sum[b_tid] = sum;
      
      __syncthreads();

      //block reduction (max efficency)
      if (N >= 32) { if (tid < 512) { warp_sum[b_tid] += warp_sum[b_tid + 512]; } __syncthreads(); }

      if (N >= 16) { if (tid < 256) { warp_sum[b_tid] += warp_sum[b_tid + 256]; } __syncthreads(); }

      if (N >= 8) { if (tid < 128) { warp_sum[b_tid] += warp_sum[b_tid + 128]; } __syncthreads(); }

      if (N >= 4) { if (tid < 64) { warp_sum[b_tid] += warp_sum[b_tid + 64]; } __syncthreads(); }

      if (N >= 2) { if (tid < 32) { warp_sum[b_tid] += warp_sum[b_tid + 32]; } __syncthreads(); }


      // we have only one warp, no need to sync entire block
      if (warp_id < 16) { warp_sum[b_tid] += warp_sum[b_tid + 16]; __syncwarp();}
      
      if (warp_id < 8) { warp_sum[b_tid] += warp_sum[b_tid + 8]; __syncwarp();}

      if (warp_id < 4) { warp_sum[b_tid] += warp_sum[b_tid + 4]; __syncwarp();}
      
      if (warp_id < 2) { warp_sum[b_tid] += warp_sum[b_tid + 2]; __syncwarp();}
      
      if (warp_id < 1) { warp_sum[b_tid] += warp_sum[b_tid + 1]; __syncwarp();}
      
      //reduction over, save result in final array
      if (warp_id == 0) {
         result[row] = warp_sum[b_tid];
      }

   }


} */

/*
__global__ void product_one_row_one_thread_csr(int M, int* irp, int* ja, double* as, double* array, double* result) {

   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   int row = tid;
   int i;
   //printf("%d\n", tid);

   if (row < M) {

      // warp sums row
      int limit = irp[row+1];
      double sum = 0;
      for (i = irp[row]; i < limit; i += 1) {
         sum += as[i] * array[ja[i]];
      }

      result[row] = sum;
      
   }


}

template <unsigned int N> __global__ void product_N_row_one_thread_csr(int M, int* irp, int* ja, double* as, double* array, double* result) {

   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   int row_start = tid*N;
   int row_end = tid*N + N;
   int i;
   //printf("%d\n", tid);

   // warp sums row
   for (; row_start < row_end && row_start < M; row_start++) {
      double sum = 0;
      int limit = irp[row_start+1];
      for (i = irp[row_start]; i < limit; i += 1) {
         sum += as[i] * array[ja[i]];
      }

      result[row_start] = sum;
   }

}
*/

extern "C"
float cuda_product_csr(csr_matrix* matrix, double* array, double* result) {

   unsigned int* irp_gpu;
   unsigned int* ja_gpu;
   double* as_gpu;
   double* array_gpu;
   double* result_gpu;

   //alloc arguments
   gpuErrchk( cudaMalloc((void**) &irp_gpu, sizeof(int)*(matrix->M+1)) );
   gpuErrchk( cudaMalloc((void**) &ja_gpu, sizeof(int)*matrix->nz) );
   gpuErrchk( cudaMalloc((void**) &as_gpu, sizeof(double)*matrix->nz) );
   gpuErrchk( cudaMalloc((void**) &array_gpu, sizeof(double)*matrix->M) );
   gpuErrchk( cudaMalloc((void**) &result_gpu, sizeof(double)*matrix->M) );

   //copy arguments
   gpuErrchk( cudaMemcpy(irp_gpu, matrix->irp, sizeof(int)*(matrix->M+1), cudaMemcpyHostToDevice) );
   gpuErrchk( cudaMemcpy(ja_gpu, matrix->ja, sizeof(int)*matrix->nz, cudaMemcpyHostToDevice) );
   gpuErrchk( cudaMemcpy(as_gpu, matrix->as, sizeof(double)*matrix->nz, cudaMemcpyHostToDevice) );
   gpuErrchk( cudaMemcpy(array_gpu, array, sizeof(double)*matrix->M, cudaMemcpyHostToDevice) );
   
   //set up timer
   cudaEvent_t start, stop;
   gpuErrchk( cudaEventCreate(&start) );
   gpuErrchk( cudaEventCreate(&stop) );

   gpuErrchk( cudaEventRecord(start, 0) );
   product_one_row_one_warp_csr<<<(matrix->M*WARP_SIZE) / BLOCK_SIZE + 1, BLOCK_SIZE , sizeof(double)*BLOCK_SIZE>>>(matrix->M, irp_gpu, ja_gpu, as_gpu, array_gpu, result_gpu);      
   gpuErrchk( cudaEventRecord(stop, 0) );

   gpuErrchk( cudaEventSynchronize(stop) );

   //calculate time of computation (in ms)
   float time;
   gpuErrchk( cudaEventElapsedTime(&time, start, stop) );
   
   //copy result back from gpu
   gpuErrchk( cudaMemcpy(result, result_gpu, sizeof(double)*matrix->M, cudaMemcpyDeviceToHost) );
   
   //free everything
   gpuErrchk( cudaFree(irp_gpu) );
   gpuErrchk( cudaFree(ja_gpu) );
   gpuErrchk( cudaFree(as_gpu) );
   gpuErrchk( cudaFree(result_gpu) );
   gpuErrchk( cudaFree(array_gpu) );
   gpuErrchk( cudaEventDestroy(start) );
   gpuErrchk( cudaEventDestroy(stop) );

   return time;

}

// ################################ ellpack #########################################

__global__ void product_one_row_one_warp_ellpack(unsigned int M, unsigned int maxnz, unsigned int* __restrict__ ja, double* __restrict__ as, double* __restrict__ array, double* __restrict__ result) {

   __shared__ double warp_sum[BLOCK_SIZE];

   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   int b_tid = threadIdx.x;
   int warp_id = tid % WARP_SIZE;
   unsigned int row = tid / WARP_SIZE;
   int i;

   warp_sum[b_tid] = 0;
   if (row < M) {

      // warp sums row
      double sum = 0;
      int index = row*maxnz;
      for (i = warp_id; i < maxnz; i += WARP_SIZE) {
         sum += as[index + i] * array[ja[index + i]];
      }

      warp_sum[b_tid] = sum;
      
      __syncwarp();

      // we have only one warp, no need to sync entire block
      if (warp_id < 16) { warp_sum[b_tid] += warp_sum[b_tid + 16]; __syncwarp();}
      
      if (warp_id < 8) { warp_sum[b_tid] += warp_sum[b_tid + 8]; __syncwarp();}

      if (warp_id < 4) { warp_sum[b_tid] += warp_sum[b_tid + 4]; __syncwarp();}
      
      if (warp_id < 2) { warp_sum[b_tid] += warp_sum[b_tid + 2]; __syncwarp();}
      
      if (warp_id < 1) { warp_sum[b_tid] += warp_sum[b_tid + 1]; __syncwarp();}
      
      //reduction over, save result in final array
      if (warp_id == 0) {
         result[row] = warp_sum[b_tid];
      }

   }


}

extern "C"
float cuda_product_ellpack(ellpack_matrix* matrix, double* array, double* result) {

   unsigned int* ja_gpu;
   double* as_gpu;
   double* array_gpu;
   double* result_gpu;

   //alloc arguments
   gpuErrchk( cudaMalloc((void**) &ja_gpu, (unsigned long)sizeof(unsigned int)*matrix->M*matrix->maxnz) );
   gpuErrchk( cudaMalloc((void**) &as_gpu, (unsigned long)sizeof(double)*matrix->M*matrix->maxnz) );
   gpuErrchk( cudaMalloc((void**) &array_gpu, sizeof(double)*matrix->M) );
   gpuErrchk( cudaMalloc((void**) &result_gpu, sizeof(double)*matrix->M) );

   //copy arguments
   gpuErrchk( cudaMemcpy(ja_gpu, matrix->ja, (unsigned long) sizeof(int)*matrix->M*matrix->maxnz, cudaMemcpyHostToDevice) );
   gpuErrchk( cudaMemcpy(as_gpu, matrix->as, (unsigned long) sizeof(double)*matrix->M*matrix->maxnz, cudaMemcpyHostToDevice) );
   gpuErrchk( cudaMemcpy(array_gpu, array, sizeof(double)*matrix->M, cudaMemcpyHostToDevice) );
   
   //set up timer
   cudaEvent_t start, stop;
   gpuErrchk( cudaEventCreate(&start) );
   gpuErrchk( cudaEventCreate(&stop) );

   gpuErrchk( cudaEventRecord(start, 0) );
   product_one_row_one_warp_ellpack<<<(matrix->M*WARP_SIZE) / BLOCK_SIZE + 1, BLOCK_SIZE , sizeof(double)*BLOCK_SIZE>>>(matrix->M, matrix->maxnz, ja_gpu, as_gpu, array_gpu, result_gpu);      
   gpuErrchk( cudaEventRecord(stop, 0) );

   gpuErrchk( cudaEventSynchronize(stop) );

   //calculate time of computation (in ms)
   float time;
   gpuErrchk( cudaEventElapsedTime(&time, start, stop) );
   
   //copy result back from gpu
   gpuErrchk( cudaMemcpy(result, result_gpu, sizeof(double)*matrix->M, cudaMemcpyDeviceToHost) );
   
   //free everything
   gpuErrchk( cudaFree(ja_gpu) );
   gpuErrchk( cudaFree(as_gpu) );
   gpuErrchk( cudaFree(result_gpu) );
   gpuErrchk( cudaFree(array_gpu) );
   gpuErrchk( cudaEventDestroy(start) );
   gpuErrchk( cudaEventDestroy(stop) );

	return time;

}