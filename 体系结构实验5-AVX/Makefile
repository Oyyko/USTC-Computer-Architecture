CC= gcc -mavx -mavx2 -mfma -msse -msse2 -msse3

1.out : 1.c
	$(CC) 1.c -o 1.out	

1-1.out : 1-1.c
	$(CC) 1-1.c -o 1-1.out

2.out : 2.c
	$(CC) 2.c -o 2.out

3.out : 3.c
	$(CC) 3.c -o 3.out

.PHONY: all
all:
	$(CC) 1.c -o 1.out
	$(CC) 1-1.c -o 1-1.out
	$(CC) 2.c -o 2.out
	$(CC) 3.c -o 3.out

