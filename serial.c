#include <stdlib.h>
#include <stdio.h>

#include "include/formats.h"

int main(int argc, char *argv[]) {

	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		return -1;
	} 

	csr_matrix* matrix = read_mm_csr(argv[1]);

	print_csr_matrix(*matrix);
	
	print_csr_matrix_market(*matrix);

	return 0;
}