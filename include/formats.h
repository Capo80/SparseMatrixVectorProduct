#ifndef FORMATS_H
#define FORMATS_H

#define ELLPACK_NZ_RATIO 3

typedef struct 
{
	unsigned int M;
	unsigned int N;
	unsigned int nz;
	unsigned int* irp;
	unsigned int* ja;
	double* as;
} csr_matrix;

typedef struct
{
	unsigned int M;
	unsigned int N;
	unsigned int nz;
	unsigned int maxnz;
	unsigned int** ja;
	double** as;
} ellpack_matrix;

typedef struct value_node {

	int column;
	double value;
	struct value_node* next;

} value_node;

void get_max_min_csr(csr_matrix* matrix, double* max, double* min);
void print_csr_matrix(csr_matrix matrix);
void print_csr_matrix_market(csr_matrix matrix);
csr_matrix* read_mm_csr(char* filename);
void free_matrix_csr(csr_matrix* matrix);

void get_max_min_ellpack(ellpack_matrix* matrix, double* max, double* min);
void print_ellpack_matrix(ellpack_matrix matrix);
void print_ellpack_matrix_market(ellpack_matrix matrix);
ellpack_matrix* read_mm_ellpack(char* filename);
void free_matrix_ellpack(ellpack_matrix* matrix);

value_node* create_new_node(int column, double value);
void print_list(value_node* list);

#endif