#include <time.h>
#include <omp.h>
#include <stdio.h>

#include "../include/formats.h"
#include "../include/product_cpu.h"

double serial_product_csr(csr_matrix* matrix, double* array, double* result) {

	unsigned int M = matrix->M;
	unsigned int* irp = matrix->irp;
	double* as = matrix->as;
	unsigned int* ja = matrix->ja;

	double start = omp_get_wtime();

	unsigned int i,j,limit;
	double sum;
	for (i = 0; i < M; i++) {
		sum = 0;
		limit = irp[i+1];
		for (j = irp[i]; j < limit; j++)
			sum += as[j]*array[ja[j]];

		result[i] = sum;
	}
	return omp_get_wtime() - start;

}


double serial_product_ellpack(ellpack_matrix* matrix, double* array, double* result) {

	unsigned int M = matrix->M;
	double* as = matrix->as;
	unsigned int* ja = matrix->ja;
	unsigned int maxnz = matrix->maxnz;
	
	double start = omp_get_wtime();

	unsigned int i,j,row;
	double sum;
	for (i = 0; i < M; i++) {
		sum = 0;
		row = i*maxnz;
		for (j = 0; j < maxnz; j++)
			sum += as[row+j]*array[ja[row+j]];

		result[i] = sum;
	}
	return omp_get_wtime() - start;

}

double omp_product_csr(csr_matrix* matrix, double* array, double* result) {

	unsigned int M = matrix->M;
	unsigned int* irp = matrix->irp;
	double* as = matrix->as;
	unsigned int* ja = matrix->ja;

	double start = omp_get_wtime();

	unsigned int i, j, limit;
	double sum;
	#pragma omp parallel for schedule(static) default(none) private(i, j, sum, limit) shared(M, ja, irp, as, array, result)
	for (i = 0; i < M; i++) {
        sum = 0;
        limit = irp[i+1];
		for (j = irp[i]; j < limit; j++) 
			sum += as[j]*array[ja[j]];

		result[i] = sum;
	}
	return omp_get_wtime() - start;

}

double omp_product_ellpack(ellpack_matrix* matrix, double* array, double* result) {

	unsigned int M = matrix->M;
	double* as = matrix->as;
	int* ja = matrix->ja;
	int maxnz = matrix->maxnz;

	double start = omp_get_wtime();

	unsigned int i, j, row;
	double sum;
	#pragma omp parallel for schedule(static) default(none) private(i, j, sum, row) shared(matrix, M, ja, as, maxnz, array, result)
	for (i = 0; i < M; i++) {
		sum = 0;
		row = i*maxnz;
		for (j = 0; j < maxnz; j++)
			sum += as[row+j]*array[ja[row+j]];

		result[i] = sum;
	}
	return omp_get_wtime() - start;

}