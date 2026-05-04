all: assignment_1

assignment_1: assignment_1.c
	mpicc -Wall -Wextra -o assignment_1 assignment_1.c -llapacke -llapack -lblas -lm

.PHONY: all clean

clean:
	rm -f assignment_1
