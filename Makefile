all:
	gcc -c lib/mmio.c -o lib/mmio.o
	gcc -c lib/formats.c -o lib/formats.o
	gcc -o serial serial.c lib/mmio.o lib/formats.o 