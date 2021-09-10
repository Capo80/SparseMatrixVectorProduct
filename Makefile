all:
	gcc -c lib/mmio.c -o lib/mmio.o -O3
	gcc -c lib/formats.c -o lib/formats.o -O3
	gcc -c lib/product_cpu.c -o lib/product_cpu.o -fopenmp -O3
	nvcc -c lib/product_gpu.cu -o lib/product_gpu.o -arch=sm_75 -O3
	gcc -o fill_csv_head fill_csv_head.c -O3
	gcc -o correctness correctness.c lib/mmio.o lib/formats.o lib/product_cpu.o -lgomp lib/product_gpu.o  -L/usr/local/cuda_save_11.2/lib64 -lcuda -lcudart -lstdc++ -O3 
	gcc -o time time.c lib/mmio.o lib/formats.o lib/product_cpu.o -lgomp lib/product_gpu.o  -L/usr/local/cuda_save_11.2/lib64 -lcuda -lcudart -lstdc++ -O3

always_ellpack:	
	gcc -c lib/mmio.c -o lib/mmio.o -O3
	gcc -c lib/formats.c -o lib/formats.o -DELLPACK_ALWAYS=1 -O3
	gcc -c lib/product_cpu.c -o lib/product_cpu.o -fopenmp -O3
	nvcc -c lib/product_gpu.cu -o lib/product_gpu.o -arch=sm_75 -O3
	gcc -o fill_csv_head fill_csv_head.c -O3
	gcc -o time time.c lib/mmio.o lib/formats.o lib/product_cpu.o -lgomp lib/product_gpu.o  -L/usr/local/cuda_save_11.2/lib64 -lcuda -lcudart -lstdc++ -O3
	gcc -o correctness correctness.c lib/mmio.o lib/formats.o lib/product_cpu.o -lgomp lib/product_gpu.o  -L/usr/local/cuda_save_11.2/lib64 -lcuda -lcudart -lstdc++ -O3

clean:
	rm lib/*.o
	rm correctness
	rm fill_csv_head
	rm time	