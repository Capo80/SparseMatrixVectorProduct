all:
	gcc -c lib/mmio.c -o lib/mmio.o
	gcc -c lib/formats.c -o lib/formats.o
	gcc -c lib/product_cpu.c -o lib/product_cpu.o -fopenmp
	nvcc -Xptxas="-v" -c lib/product_gpu.cu -o lib/product_gpu.o -arch=sm_50 
	gcc -o main main.c lib/mmio.o lib/formats.o lib/product_cpu.o -lgomp lib/product_gpu.o  -L/usr/local/cuda-11.4/lib64 -lcuda -lcudart 
	gcc -o correctness correctness.c lib/mmio.o lib/formats.o lib/product_cpu.o -lgomp lib/product_gpu.o  -L/usr/local/cuda-11.4/lib64 -lcuda -lcudart 

always_ellpack:	
	gcc -c lib/mmio.c -o lib/mmio.o
	gcc -c lib/formats.c -o lib/formats.o -DELLPACK_ALWAYS=1
	gcc -c lib/product_cpu.c -o lib/product_cpu.o -fopenmp
	nvcc -c lib/product_gpu.cu -o lib/product_gpu.o -arch=sm_50
	gcc -o main main.c lib/mmio.o lib/formats.o lib/product_cpu.o lib/product_gpu.o -L/usr/local/cuda/lib64 -lcuda -lcudart -lgomp 
	gcc -o correctness correctness.c lib/mmio.o lib/formats.o lib/product_cpu.o -lgomp lib/product_gpu.o  -L/usr/local/cuda-11.4/lib64 -lcuda -lcudart 