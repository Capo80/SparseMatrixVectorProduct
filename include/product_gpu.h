#ifndef PRODUCT_GPU_H
#define PRODUCT_GPU_H

double cuda_product_csr(csr_matrix* matrix, double* array, double* result);
double cuda_product_ellpack(ellpack_matrix* matrix, double* array, double* result);

#endif