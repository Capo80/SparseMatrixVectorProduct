#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <omp.h>

#include "include/formats.h"
#include "include/product_cpu.h"
#include "include/product_gpu.h"

#define EXECUTION_PER_MATRIX 10
#define MAX_THREADS 4

double* get_random_array(double min, double max, int size) {

	if (min == max) {
		min -= 10;
		max += 10;
	}

	double random_value;

    srand(1234);

    double* array = malloc(size*sizeof(double));

    for (int i = 0; i < size; i++)
    	array[i] = (double)rand()/RAND_MAX*2*max-min;

    return array;
}

int main(int argc, char *argv[]) {

    int format, product;
    DIR* fd;
    FILE* results_csv;
    struct dirent* in_file;
    char* matrix_folder = "matrices";
    double time, flops = 0;

    results_csv = fopen("results.csv","a");

	if (argc >= 2)
	{	
		//csr_matrix* matrix = read_mm_csr(argv[1]);

		//print_csr_matrix(*matrix);
		
		//print_csr_matrix_market(*matrix);

		ellpack_matrix* matrix = read_mm_ellpack(argv[1]);

		double min, max;
        get_max_min_ellpack(matrix, &max, &min);
		print_ellpack_matrix(*matrix);

		//print_ellpack_matrix_market(*matrix);

		return 0;
	} 

    printf("Choose matrix format:\n");
    printf("1) ELLPACK\n");
    printf("2) CSR\n");

    scanf("%d", &format);

    omp_set_num_threads(MAX_THREADS);

    /* Scanning the in directory */
    if (NULL == (fd = opendir (matrix_folder))) 
    {
        fprintf(stderr, "Error : Failed to open input directory - %s\n", strerror(errno));
        return 1;
    }

    while ((in_file = readdir(fd))) 
    {
        if (!strcmp (in_file->d_name, "."))
            continue;
        if (!strcmp (in_file->d_name, ".."))    
            continue;

        flops = 0;
        char path[280];
        sprintf(path, "matrices/%s", in_file->d_name);

        if (format == 1) {

            //read matrix from file
        	ellpack_matrix* matrix = read_mm_ellpack(path);
        	if ((long) matrix < 0){
                fprintf(results_csv, ",-1");
      			continue;
            }

            //generate random array for product
        	double min, max;
        	get_max_min_ellpack(matrix, &max, &min);
        	double* vector = get_random_array(min, max, matrix->M);
 			double* result_serial = malloc(matrix->M*sizeof(double));
 			double* result_omp = malloc(matrix->M*sizeof(double));
            double* result_cuda = malloc(matrix->M*sizeof(double));
            
            //Do product
            serial_product_ellpack(matrix, vector, result_serial);
            omp_product_ellpack(matrix, vector, result_omp);
            cuda_product_ellpack(matrix, vector, result_cuda);    

            printf("Results: \n");
            for (int i = 0; i < matrix->M; i++)
                printf("%lg %lg %lg\n", result_serial[i], result_omp[i], result_cuda[i]);

        	free(result_serial);
            free(result_omp);
            free(result_cuda);
        	free(vector);
        	free_matrix_ellpack(matrix);
        } else if (format == 2) {

            // read matrxi from file
        	csr_matrix* matrix = read_mm_csr(path);

            //generate random array for product
        	double min, max;
        	get_max_min_csr(matrix, &max, &min);
        	double* vector = get_random_array(min, max, matrix->M);
            double* result_serial = malloc(matrix->M*sizeof(double));
            double* result_omp = malloc(matrix->M*sizeof(double));
            double* result_cuda = malloc(matrix->M*sizeof(double));
            
            //Do product
            serial_product_csr(matrix, vector, result_serial);
            omp_product_csr(matrix, vector, result_omp);
            cuda_product_csr(matrix, vector, result_cuda);    

            printf("Results (%s): \n", in_file->d_name);
            for (int i = 0; i < matrix->M; i++)
                printf("%lg %lg %lg\n", result_serial[i], result_omp[i], result_cuda[i]);

            free(result_serial);
            free(result_omp);
            free(result_cuda);
        	free(vector);
        	free_matrix_csr(matrix);
        } else {
        	fprintf(stderr, "Error: format is not correct\n");
            return -1;
        }

        break;
    }

    closedir(fd);

    return 0;

}