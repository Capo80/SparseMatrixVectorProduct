#include <time.h>
#include <omp.h>

#include "../include/formats.h"
#include "../include/product.h"

double serial_product_csr(csr_matrix* matrix, double* array, double* result) {

	double start = omp_get_wtime();

	int sum;
	for (int i = 0; i < matrix->M; i++) {
		sum = 0;
		for (int j = matrix->irp[i]; j < matrix->irp[i+1]; j++)
			sum += matrix->as[j]*array[matrix->ja[j]];

		result[i] = sum;
	}
	return omp_get_wtime() - start;

}


double serial_product_ellpack(ellpack_matrix* matrix, double* array, double* result) {


	double start = omp_get_wtime();

	int sum;
	for (int i = 0; i < matrix->M; i++) {
		sum = 0;
		for (int j = 0; j < matrix->maxnz; j++)
			sum += matrix->as[i][j]*array[matrix->ja[i][j]];

		result[i] = sum;
	}
	return omp_get_wtime() - start;

}