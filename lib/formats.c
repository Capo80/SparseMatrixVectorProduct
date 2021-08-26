#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/formats.h"
#include "../include/mmio.h"

//ALL THE FUNCTIONS HERE ASSUME THAT THE MATRIX MARKET FILE IS ORDERED TO REDUCE COMPUTATION TIME
//ORDER MUST BE FIRST BY COLUMN AND THEN BY ROW
//All the matrix downloaded from sparse.tamu.edu are in this order

//############################ Row List helpers #################################

int* fill_lists(FILE* f, value_node* row_lists[], int size, char pattern, char symmetric, int nz) {

    //lists used for initial reading
    value_node* temp_nodes[size];

    //used for ELLPACK, saves the number of nz per row
    int* row_nz = malloc(size*sizeof(int));
    for (int i = 0; i < size; i++) {
        row_lists[i] = NULL;
        row_nz[i] = 0;   
    }

    for (int i=0; i < nz; i++)
    {   
        int temp_row, temp_col;
        double temp_val;

        //if pattern value is always 1.0
        if (!pattern)
            fscanf(f, "%d %d %lg\n", &temp_row, &temp_col, &temp_val);
        else {
            fscanf(f, "%d %d\n", &temp_row, &temp_col);
            temp_val = 1.0;
        }
        temp_col--;
        temp_row--;
        
        value_node* new_node = create_new_node(temp_col, temp_val);
            
        //insert in list, the value is already ordered per column, even in the symmetrical case
        if (row_lists[temp_row] == NULL) {
            row_lists[temp_row] = new_node;
            temp_nodes[temp_row] = row_lists[temp_row];
        } else {
            temp_nodes[temp_row]->next = new_node;
            temp_nodes[temp_row] = temp_nodes[temp_row]->next;
        }
        row_nz[temp_row]++;

        if (symmetric) {

            value_node* new_node = create_new_node(temp_row, temp_val);
            if (row_lists[temp_col] == NULL) {
                row_lists[temp_col] = new_node;
                temp_nodes[temp_col] = row_lists[temp_col];
            } else {
                temp_nodes[temp_col]->next = new_node;
                temp_nodes[temp_col] = temp_nodes[temp_col]->next;
            }
            row_nz[temp_col]++;
        }
    }
    
    return row_nz;
}

value_node* create_new_node(int column, double value) {

    value_node* new_node = malloc(sizeof(value_node));
    new_node->column = column;
    new_node->value = value;
    new_node->next = NULL;

    return new_node;
}

void print_list(value_node* list) {

    printf("List:\n");
    for(value_node* temp = list; temp != NULL; temp = temp->next)
        printf("col %d value %lg\n", temp->column, temp->value);

}

//############################ CSR functions ####################################

//reading matrix and returning a csr struct
//Complexity O(nz)
//Memory O(nz)
csr_matrix* read_matrix_csr(FILE* f, char pattern, char symmetric) {

    csr_matrix* to_return = malloc(sizeof(csr_matrix));
    

    //find matrix size
    if (mm_read_mtx_crd_size(f, &to_return->M, &to_return->N, &to_return->nz) !=0)
        return (csr_matrix*)-4;

    //alloc memory
    to_return->irp = malloc((to_return->M+1)*sizeof(unsigned int));
    if (!symmetric) {
        to_return->ja = malloc(to_return->nz*sizeof(unsigned int));
        to_return->as = malloc(to_return->nz*sizeof(double));
    } else {
        to_return->ja = malloc(to_return->nz*2*sizeof(unsigned int));
        to_return->as = malloc(to_return->nz*2*sizeof(double));
    }
    //read file into lists, then combine them into the final format
    value_node* row_lists[to_return->M];
    fill_lists(f, row_lists, to_return->M, pattern, symmetric, to_return->nz);

    int c = 0;
    for (int i = 0; i < to_return->M; i++) {
        to_return->irp[i] = c;

        value_node* p = row_lists[i];
        while(p != NULL) {
            to_return->ja[c] = p->column;
            to_return->as[c] = p->value;
            c++;
            value_node* temp = p;
            p = p->next;
            free(temp);
        }
    }
    to_return->irp[to_return->M] = to_return->nz;
    
    return to_return;

}

//read a matrix market and returns it in a csr model
//this just identifies the structure of the file and calls the actual read function
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

    to_return = read_matrix_csr(f, mm_is_pattern(matcode), mm_is_symmetric(matcode));

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
	for (int i = 0; i < matrix.M+1; i++)
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

	for (int i = 0; i < matrix.M; i++) {	
        for(int c = matrix.irp[i]; c < matrix.irp[i+1]; c++)
		  printf("%d %d %lg\n", i+1, matrix.ja[c]+1, matrix.as[c]);
	}


}



// ########################### ELLPACK functions ######################################

//reading matrix and returning a ellpack struct
//very similar to the csr version
//Complexity O(MAX{M*maxnzm, nz})
//Memory O(MAX{M*maxnzm, nz})
ellpack_matrix* read_matrix_ellpack(FILE* f, char pattern, char symmetric) {

    ellpack_matrix* to_return = malloc(sizeof(ellpack_matrix));
    
    //find matrix size
    if (mm_read_mtx_crd_size(f, &to_return->M, &to_return->N, &to_return->nz) !=0)
        return (ellpack_matrix*)-4;

    //read file into lists
    value_node* row_lists[to_return->M];
    int* row_nz = fill_lists(f, row_lists, to_return->M, pattern, symmetric, to_return->nz);

    printf("Done with lists\n");

    //find max number of non-zero
    to_return->maxnz = 0;
    for (int i = 0; i < to_return->M; i++) {
        if (row_nz[i] > to_return->maxnz)
            to_return->maxnz = row_nz[i];
    }

    free(row_nz);
    printf("Max found: %d\n", to_return->maxnz);
    
#ifndef ELLPACK_ALWAYS

    //check if its worth to use ellpack
    if (to_return->maxnz*to_return->M > ELLPACK_NZ_RATIO*to_return->nz) {
        printf("Too many zeroes, ELLAPACK is not worth\n");
        return (ellpack_matrix*) -5;
    }

#endif

    //alloc memory
    to_return->ja = malloc(to_return->M*sizeof(unsigned int*));
    to_return->as = malloc(to_return->M*sizeof(double*));
    for (int i = 0; i < to_return->M; i++) {
        to_return->ja[i] = malloc(to_return->maxnz*sizeof(unsigned int));
        to_return->as[i] = malloc(to_return->maxnz*sizeof(double));
    }

    //fill matrices
    for (int i = 0; i < to_return->M; i++) {
        value_node* p = row_lists[i];
        int last_col = 0;
        for (int j = 0; j < to_return->maxnz; j++) {
            //printf("%d %d %d\n", i, j, p->column);
            if (p != NULL) {
                to_return->ja[i][j] = p->column;
                to_return->as[i][j] = p->value;
                last_col = p->column;
                value_node* temp = p;
                p = p->next;        
                free(temp);
            } else {
                to_return->ja[i][j] = last_col;
                to_return->as[i][j] = 0;
            }
        }
    }

    return to_return;

}

//read a matrix market and returns it in a ELLPACK model
//the same as the csr one
ellpack_matrix* read_mm_ellpack(char* filename) {

    MM_typecode matcode;
    ellpack_matrix* to_return;
    FILE *f;

    if ((f = fopen(filename, "r")) == NULL){
        return (ellpack_matrix*) -1;
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        return (ellpack_matrix*) -2;
    }

    //check if correct
    if (mm_is_complex(matcode) && mm_is_matrix(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return (ellpack_matrix*)-3;
    }

    to_return = read_matrix_ellpack(f, mm_is_pattern(matcode), mm_is_symmetric(matcode));

    if (f != stdin) fclose(f);

    return to_return;

}


//print matrix in ellpack format (in the format, not the full rectangle)
void print_ellpack_matrix(ellpack_matrix matrix) {

    printf("AS:\n");
    for (int i = 0; i < matrix.M; i++) {
        for (int j = 0; j < matrix.maxnz; j++) {
            printf("%lg  ", matrix.as[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("JA:\n");
    for (int i = 0; i < matrix.M; i++) {
        for (int j = 0; j < matrix.maxnz; j++) {
            printf("%d  ", matrix.ja[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Rows: %d\t Columns: %d\t NZ: %d MAXNZ: %d\n", matrix.M, matrix.N, matrix.nz, matrix.maxnz);

}

//print matrix in ellpack format (in the matrix market format)
void print_ellpack_matrix_market(ellpack_matrix matrix) {

    for (int i = 0; i < matrix.M; i++) {    
        for(int j = 0; j < matrix.maxnz; j++) {
            printf("%d %d %lg\n", i+1, matrix.ja[i][j]+1, matrix.as[i][j]);
        }
    }


}