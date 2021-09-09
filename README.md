# SparseMatrixVectorProduct
An efficent parallelized code for SpMV for openmp and CUDA, a project for the SCPA course

## Usage

The module has been build for Compute Capability 7.5 and it is compiled only for that architecture, there is a high probability that the code will not work on any other.

To compile the code run "make" in the root directory of the project.

Before running the code add a "matrices" folder in the root directory of the project and put inside all the matrices to multiply. Then run the "fill_csv" to fill the head of the csv file that will contain the results.

```
ATTENTION! The matrices must saved be in MatrixMarket format and ordered by COLUMN! The code will not work otherwise.
```

To test the correctness of the result use the "correctness" binary, to time the speed of the code use the "time" binary.

Both binaries will ask about the format to save the matrix in and the type of product to execute. 
