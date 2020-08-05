EXECS=part1
MPICC?=mpicc

all: ${EXECS}

part1: part1_zpc5077.c
	${MPICC} -o part1 part1_zpc5077.c -O3 -g

clean:
	rm ${EXECS}
