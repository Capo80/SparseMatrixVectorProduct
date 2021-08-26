all:
	gcc -c lib/mmio.c -o lib/mmio.o
	gcc -c lib/formats.c -o lib/formats.o
	gcc -o serial serial.c lib/mmio.o lib/formats.o 

always_ellpack:	
	gcc -c lib/mmio.c -o lib/mmio.o
	gcc -c lib/formats.c -o lib/formats.o -DELLPACK_ALWAYS=1 
	gcc -o serial serial.c lib/mmio.o lib/formats.o 
