/**
* popmat.c
* * Program for creating LP input files.
* */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MAX 100
int main(int argc, char* argv[]) {
	FILE *file;
	int i, j, m, n;
	if(argc < 3) {
		fprintf(stderr, "usage: popmat filename m n\n");
		exit(1);
	}
	file = fopen(argv[1], "w");
	m = atoi(argv[2]);
	n = atoi(argv[3]);
	fprintf(file, "%d %d ", m, m+n);
	fprintf(stderr, "m n written.\n");

	time_t t;
   
        srand((unsigned) time(&t));

	for(i = 0; i < n; i++ ) {
		fprintf(file, "%f ", (float)(rand()%MAX));
	}
	for(i = 0; i < m; i++ ) {
		fprintf(file, "%f ", (float)0);
	}
	fprintf(stderr, "c written.\n");
	for(i = 0; i < m; i++ ) {
		fprintf(file, "%f ", (float)(rand()%MAX));
	}
	fprintf(stderr, "b written.\n");
	for(i = 0; i < m; i++ ) {
		for(j = 0; j < n; j++ ) {
			fprintf(file, "%f ", (float)(rand()%MAX));
		}
		for(j = 0; j < m; j++ ) {
			fprintf(file, "%f ", (float)(i==j));
		}
	}
	fprintf(stderr, "A written.\n");
	fclose(file);
	return 0; 
}
