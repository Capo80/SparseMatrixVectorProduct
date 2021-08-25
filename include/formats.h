#ifndef FORMATS_H
#define FORMATS_H

typedef struct 
{
	unsigned int M;
	unsigned int N;
	unsigned int nz;
	unsigned int* irp;
	unsigned int* ja;
	double* as;
} csr_matrix;

void print_csr_matrix(csr_matrix matrix);
void print_csr_matrix_market(csr_matrix matrix);
csr_matrix* read_mm_csr(char* filename);

#endif