#ifndef PRODUCT_GPU_H
#define PRODUCT_GPU_H

#define WARP_SIZE 32
#define BLOCK_SIZE 128

float cuda_product_csr(csr_matrix* matrix, double* array, double* result);
float cuda_product_ellpack(ellpack_matrix* matrix, double* array, double* result);

#endif