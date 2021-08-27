all:
	gcc -c lib/mmio.c -o lib/mmio.o
	gcc -c lib/formats.c -o lib/formats.o
	gcc -c lib/product.c -o lib/product.o
	gcc -o main main.c lib/mmio.o lib/formats.o lib/product.o -lgomp

always_ellpack:	
	gcc -c lib/mmio.c -o lib/mmio.o
	gcc -c lib/formats.c -o lib/formats.o -DELLPACK_ALWAYS=1
	gcc -c lib/product.c -o lib/product.o 
	gcc -o main main.c lib/mmio.o lib/formats.o lib/product.o -lgomp
