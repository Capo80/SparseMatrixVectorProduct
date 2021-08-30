#ifndef PRODUCT_CPU_H
#define PRODUCT_CPU_H

double serial_product_csr(csr_matrix* matrix, double* array, double* result);
double serial_product_ellpack(ellpack_matrix* matrix, double* array, double* result);
double omp_product_csr(csr_matrix* matrix, double* array, double* result);
double omp_product_ellpack(ellpack_matrix* matrix, double* array, double* result);


#endif