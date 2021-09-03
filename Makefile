all:
	gcc -c lib/mmio.c -o lib/mmio.o
	gcc -c lib/formats.c -o lib/formats.o
	gcc -c lib/product_cpu.c -o lib/product_cpu.o -fopenmp
	nvcc -c lib/product_gpu.cu -o lib/product_gpu.o -arch=sm_75
	gcc -o fill_csv_head fill_csv_head.c
	gcc -o correctness correctness.c lib/mmio.o lib/formats.o lib/product_cpu.o -lgomp lib/product_gpu.o  -L/usr/local/cuda_save_11.2/lib64 -lcuda -lcudart -lstdc++ 
	gcc -o time time.c lib/mmio.o lib/formats.o lib/product_cpu.o -lgomp lib/product_gpu.o  -L/usr/local/cuda_save_11.2/lib64 -lcuda -lcudart -lstdc++ 

always_ellpack:	
	gcc -c lib/mmio.c -o lib/mmio.o
	gcc -c lib/formats.c -o lib/formats.o -DELLPACK_ALWAYS=1
	gcc -c lib/product_cpu.c -o lib/product_cpu.o -fopenmp
	nvcc -c lib/product_gpu.cu -o lib/product_gpu.o -arch=sm_75
	gcc -o fill_csv_head fill_csv_head.c
	gcc -o time time.c lib/mmio.o lib/formats.o lib/product_cpu.o -lgomp lib/product_gpu.o  -L/usr/local/cuda_save_11.2/lib64 -lcuda -lcudart -lstdc++ 
	gcc -o correctness correctness.c lib/mmio.o lib/formats.o lib/product_cpu.o -lgomp lib/product_gpu.o  -L/usr/local/cuda_save_11.2/lib64 -lcuda -lcudart -lstdc++ 

clean:
	rm lib/*.o
	rm correctness
	rm fill_csv_head
	rm time	