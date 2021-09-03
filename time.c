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
#define MAX_THREADS 40

double* get_random_array(double min, double max, int size) {

	if (min == max) {
		min -= 10;
		max += 10;
	}

	double random_value;

    srand(1234);

    double* array = (double*) malloc(size*sizeof(double));

    for (int i = 0; i < size; i++)
    	array[i] = (double)rand()/RAND_MAX*2*max-min;

    return array;
}

int main(int argc, char *argv[]) {

    int format, product, c = 0;
    DIR* fd;
    FILE* results_csv;
    struct dirent* in_file;
    unsigned int M, nz;
    char* matrix_folder = "matrices";
    double time = 0, flops = 0;
    float time_single = 0, flops_single = 0;
    csr_matrix* matrix_csr;
    ellpack_matrix* matrix_ellpack;

    results_csv = fopen("results.csv","a");

    //small menu
    printf("Choose matrix format:\n");
    printf("1) ELLPACK\n");
    printf("2) CSR\n");

    scanf("%d", &format);

    printf("Choose type of product:\n");
    printf("1) Serial\n");
    printf("2) Omp parallel\n");
    printf("3) CUDA parallel\n");
    
    scanf("%d", &product);

    printf("Starting calculations...\n");

    if (product != 2)
        fprintf(results_csv, "%d, %d, 1", product, format);

    //open directory
    if (NULL == (fd = opendir (matrix_folder))) 
    {
        fprintf(stderr, "Error : Failed to open input directory - %s\n", strerror(errno));
        return 1;
    }

    double saved_flops[29][MAX_THREADS]; //to save times in omp_timing
    while ((in_file = readdir(fd))) 
    {
        if (!strcmp (in_file->d_name, "."))
            continue;
        if (!strcmp (in_file->d_name, ".."))    
            continue;

        printf("Matrix: %s\n", in_file->d_name);

        flops = 0;
        char path[280];
        sprintf(path, "matrices/%s", in_file->d_name);

        //read matrix from file
        if (format == 1) {
        	matrix_ellpack = read_mm_ellpack(path);
        	if ((long) matrix_ellpack < 0){
                if (product != 2)    
                    fprintf(results_csv, ",-1");
      			for (int i = 1; i <= MAX_THREADS; i++)
                    saved_flops[c][i-1] = -1;
                c++;

                printf("--------------------\n");
                continue;
            }
            M = matrix_ellpack->M;
            nz = matrix_ellpack->nz;
        } else {
            matrix_csr = read_mm_csr(path);
            M = matrix_csr->M;
            nz = matrix_csr->nz;
        }

        //generate random array for product
    	double min, max;
        if (format == 1)
            get_max_min_ellpack(matrix_ellpack, &max, &min);
    	else
            get_max_min_csr(matrix_csr, &max, &min);
        
        double* vector = get_random_array(min, max, M);
		double* result = (double*) malloc(M*sizeof(double));
			
        //Do product
        switch (product) {
            case 1:
                for (int i = 0; i < EXECUTION_PER_MATRIX; i++) {
                    if (format == 1)    
                        time += serial_product_ellpack(matrix_ellpack, vector, result);
                    else
                        time += serial_product_csr(matrix_csr, vector, result);
                }
                time /= EXECUTION_PER_MATRIX;
                flops = 2*nz / time;
        
                fprintf(results_csv, ",%lg", flops);
                           
                printf("Time: %lf\nFLOPS: %lg\n", time, flops);
                break;
            case 2:
                for (int i = 1; i <= MAX_THREADS; i++) {
                    omp_set_num_threads(i);
                    for (int j = 0; j < EXECUTION_PER_MATRIX; j++) {
                    if (format == 1)
                        time += omp_product_ellpack(matrix_ellpack, vector, result);
                    else
                        time += omp_product_csr(matrix_csr, vector, result);
                    }
                    time /= EXECUTION_PER_MATRIX;
                    flops = 2*nz / time;
                    
                    printf("Threads: %d\nTime: %lg\nFLOPS: %lg\n", i, time, flops);

                    saved_flops[c][i-1] = flops;
                    printf("----------------------\n");
                }
                break;
            case 3:
                for (int i = 0; i < EXECUTION_PER_MATRIX; i++)    
                    if (format == 1)    
                        time_single += cuda_product_ellpack(matrix_ellpack, vector, result);    
                    else
                        time_single += cuda_product_csr(matrix_csr, vector, result);    
                        
                time_single /=EXECUTION_PER_MATRIX;
                flops_single = 2*nz / time_single * 1000; //time from gpu is in milliseconds
                
                fprintf(results_csv, ",%g", flops_single);
                           
                printf("Time: %f\nFLOPS: %g\n", time_single, flops_single);
                break;

        }

    	free(result);
    	free(vector);
    	if (format == 1)   
           free_matrix_ellpack(matrix_ellpack);
        else
           free_matrix_csr(matrix_csr);
            
        c++;

        if (product != 2)
            printf("--------------------\n");
    }
        
    if (product != 2)
        fprintf(results_csv, "\n");
    else    
        for (int i = 1; i <= MAX_THREADS; i++) {
            fprintf(results_csv, "%d, %d, %d", product, format, i);
            for (int j = 0; j < 29; j++) {

                fprintf(results_csv, ", %lg", saved_flops[j][i-1]);

            }
            fprintf(results_csv, "\n");

        }


    closedir(fd);
    fclose(results_csv);

    return 0;

}