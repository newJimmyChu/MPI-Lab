EXECS=part2
MPICC?=mpicc

all: ${EXECS}

part2: part2_zpc5077.c
	${MPICC} -o part2 part2_zpc5077.c -O3 -g -lm

clean:
	rm ${EXECS}
