# SparseMatrixVectorProduct
An efficent parallelized code for SpMV for openmp and CUDA, a project for SCPA course

## Notes

tried csr
- dynamic 1
- dynamic 100
- static
in this order on csv, static is better

possible improvements
- make ellpack ja & as a real matrix
- loop unroll

### GPU

c'è una reduction su ogni riga, le righe tra di loro sono indipendenti

idea? mantenere riga all'interno di un solo blocco? evitare problematiche di reduction intra-blocco

possibilità
- una riga per blocco
- più righe in un blocco (ragionare in warp?)

- una riga per warp
- più righe per warp NO, righe sono sufficentemente grandi per occupare un warp da sole
- una riga per più warp
