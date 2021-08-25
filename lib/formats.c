#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/formats.h"
#include "../include/mmio.h"

//reads a symmetryc matrix
//computational cost is O(nz)
csr_matrix* read_symmetric(FILE* f) {

	int last_row = 0;
	csr_matrix* to_return = malloc(sizeof(csr_matrix));

    //find matrix size
    if (mm_read_mtx_crd_size(f, &to_return->M, &to_return->N, &to_return->nz) !=0)
        return (csr_matrix*)-4;

    //alloc memory
    to_return->irp = malloc(to_return->M*sizeof(unsigned int));
    to_return->ja = malloc(to_return->nz*sizeof(unsigned int));
    to_return->as = malloc(to_return->nz*sizeof(double));

    to_return->irp[0] = 0;
    for (int i=0; i < to_return->nz; i++)
    {	
    	int temp_row;
        fscanf(f, "%d %d %lg\n", to_return->ja+i, &temp_row, to_return->as + i);
        temp_row--;
        to_return->ja[i]--;

        if (temp_row != last_row) {
        	for (int c = last_row+1; c <= temp_row; c++)
        		to_return->irp[c] = i;
            last_row = temp_row;
        }

    }

    return to_return;
}

//reading stardard matrix (non symmetrical, non pattern)
//computational cost is O(nz*(m+nz))
csr_matrix* read_normal(FILE* f) {

    int valid_as = 0;
    csr_matrix* to_return = malloc(sizeof(csr_matrix));

    //find matrix size
    if (mm_read_mtx_crd_size(f, &to_return->M, &to_return->N, &to_return->nz) !=0)
        return (csr_matrix*)-4;

    //alloc memory
    to_return->irp = malloc(to_return->M*sizeof(unsigned int));
    to_return->ja = malloc(to_return->nz*sizeof(unsigned int));
    to_return->as = malloc(to_return->nz*sizeof(double));
    

    char active_rows[to_return->M];
    memset(to_return->irp, 0, to_return->M);
    memset(active_rows, 0, to_return->M);
    for (int i=0; i < to_return->nz; i++)
    {	
    	int temp_row, temp_col;
    	double temp_val;

        fscanf(f, "%d %d %lg\n", &temp_row, &temp_col, &temp_val);
        temp_col--;
        temp_row--;
   
        //find next row pos
        int c = temp_row+1;
        while (c < to_return->M && to_return->irp[c] == 0)
        	c++;

        //print_csr_matrix(*to_return);
        //printf("c %d row %d val %lf valid %d\n", c, temp_row, temp_val, valid_as);

        //check if its he end of the array
        int numb_pos;
        if (c == to_return->M) {
       		numb_pos = valid_as;
        } else {
        	numb_pos = to_return->irp[c];
        	//shift array;
        	memmove(&to_return->as[numb_pos+1], &to_return->as[numb_pos], (valid_as-numb_pos)*sizeof(double));
        	memmove(&to_return->ja[numb_pos+1], &to_return->ja[numb_pos], (valid_as-numb_pos)*sizeof(int));
        	//update irp valid entries
			for (int m = c; m < to_return->M; m++)
				if (active_rows[m])
					to_return->irp[m]++;
        }
        //insert new
        to_return->as[numb_pos] = temp_val;
        to_return->ja[numb_pos] = temp_col;
        valid_as++;

        //update irp if necessary
    	if (!active_rows[temp_row]) {
    		active_rows[temp_row] = 1;
    		to_return->irp[temp_row] = numb_pos;
    	}


    }

    return to_return;

}

//read a matrix market and returns it in a csr model
//this just identifies the structure of the file and calls the appropriate function
csr_matrix* read_mm_csr(char* filename) {

    MM_typecode matcode;
    csr_matrix* to_return;
    FILE *f;

	if ((f = fopen(filename, "r")) == NULL){
		return (csr_matrix*) -1;
	}

    if (mm_read_banner(f, &matcode) != 0)
    {
        return (csr_matrix*) -2;
    }

    //check if correct
    if (mm_is_complex(matcode) && mm_is_matrix(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return (csr_matrix*)-3;
    }

    if (mm_is_symmetric(matcode)) {
    	to_return = read_symmetric(f);
    } else {
    	to_return = read_normal(f);
    }
    if (f != stdin) fclose(f);

	return to_return;

}

//print matrix in csr format (in the format, not the full rectangle)
void print_csr_matrix(csr_matrix matrix) {

	printf("AS: ");
	for (int i = 0; i < matrix.nz; i++)
		printf("%lg ", matrix.as[i]);
	printf("\n");

	printf("IRP: ");
	for (int i = 0; i < matrix.M; i++)
		printf("%d ", matrix.irp[i]);
	printf("\n");

	printf("JA: ");
	for (int i = 0; i < matrix.nz; i++)
		printf("%d ", matrix.ja[i]);
	printf("\n");

	printf("Rows: %d\t Columns: %d\t NZ: %d\n", matrix.M, matrix.N, matrix.nz);

}

//print matrix in csr format (in the matrix market format)
void print_csr_matrix_market(csr_matrix matrix) {

	int curr_row = 0;
	for (int i = 0; i < matrix.nz; i++) {
		if (curr_row+1 < matrix.M && i == matrix.irp[curr_row+1])
			curr_row++;
		printf("%d %d %lg\n", curr_row+1, matrix.ja[i]+1, matrix.as[i]);
	}


}
