#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include "mpi.h"

static double get_walltime() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}

// size = m / p
void Do_Calculation(int *grid_current, int *grid_next, int rank, int p, int m) {
    for (int i=1; i<(m/p)-1; i++) {
        for (int j=1; j<m-1; j++) {
            /* avoiding conditionals inside inner loop */
            int prev_state = grid_current[i*m+j];
            int num_alive  = 
                            grid_current[(i  )*m+j-1] + 
                            grid_current[(i  )*m+j+1] + 
                            grid_current[(i-1)*m+j-1] + 
                            grid_current[(i-1)*m+j  ] + 
                            grid_current[(i-1)*m+j+1] + 
                            grid_current[(i+1)*m+j-1] + 
                            grid_current[(i+1)*m+j  ] + 
                            grid_current[(i+1)*m+j+1];

            grid_next[i*m+j] = prev_state * ((num_alive == 2) + (num_alive == 3)) + (1 - prev_state) * (num_alive == 3);
        }
    }
    int *grid_tmp  = grid_next;
    grid_next = grid_current;
    grid_current = grid_tmp;

}


int main(int argc, char **argv) {

    if (argc != 3) {
        printf("%s <m> <k>\n", argv[0]);
        printf("Program for parallel Game of Life\n");
        printf("with 1D grid partitioning\n");
        printf("<m>: grid dimension (an mxm grid is created)\n");
        printf("<k>: number of time steps\n");
        printf("(initial pattern specified inside code)\n");
		exit(1);
    }

    unsigned long m, k;

    m = atol(argv[1]);
    k = atol(argv[2]);

    int i, j, t;

    double d_startTime = 0.0, d_endTime = 0.0;
	d_startTime = get_walltime();

    int rank = 0;        // Rank 0 is the root
    int p = 100;  // The number of processors
    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
    MPI_Status stat;

    /* MPI code region*/
    int count = 100;
    int *proc_grid_current, *proc_grid_next;
    int SIZE = m * (m / p);  
    int *recv_buf = (int *) malloc(m * p * sizeof(int));
    // Receive buffer for each process
    recv_buf = (int *) malloc(m * sizeof(int));
    proc_grid_current = (int *) malloc((((m / p) + 2) * m) * sizeof(int));
    proc_grid_next = (int *) malloc((((m / p) + 2) * m) * sizeof(int));

    int a = m / p;
    // Initialize the matrix
    for(int i = 0; i < m; i++){
        for(int j = 0; j < a + 2; j++){
            proc_grid_current[((a + 2)*i)+j] = 0;
            proc_grid_next[((a + 2)*i)+j] = 0;
        }
    }


    // Initialize the receive buffer
    for(int i = 0; i < p; i++){
        for(int j = 0; j < m; j++){
            recv_buf[(i*m)+j] = 0;
        }
    }


    // Begin doing the function for each process
    for(int i = 0; i < k; i++){
        if(rank != 0 && rank != p - 1){
            Do_Calculation(proc_grid_current, proc_grid_next, rank, p, m);
        }
        // Update shared region between processes
        MPI_Barrier(MPI_COMM_WORLD);

        // First we can do 0->1->2->3->4->5->6 (forwarding)
        if(rank != (p-1)){
            MPI_Sendrecv(&proc_grid_current[m*((m/p)+1)], m, MPI_INT,
                         rank+1, MPI_ANY_TAG, &recv_buf[(rank+1)*m], 
                         m, MPI_INT, rank, MPI_COMM_WORLD, &stat);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // Sharing blocks with two neighbors (except rank #0)
        if(rank != 0){
            for(int i = 0; i < m; i++){
                proc_grid_current[i] = recv_buf[(rank*m)+i];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        //Second we can do 6->5->4->3->2->1->0 (backwarding)
        if(rank != 0){
            MPI_Sendrecv(&proc_grid_current[m*((m/p)+1)], m, MPI_INT,
                         rank-1, MPI_ANY_TAG, &recv_buf[(rank+1)*m],
                         m, MPI_INT, rank, MPI_COMM_WORLD, &stat);            
        }
        if(rank != p-1){
            for(int i = 0; i < m; i++){
                proc_grid_current[(m*(a+1))+i] = recv_buf[(rank*m)+i];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

    }

    /* serial code */
    /* considering only internal cells */

    d_endTime = get_walltime();

    /* Verify */
    int verify_failed = 0;
    for (i=0; i<m; i++) {
        for (j=0; j<m; j++) {
            /* Add verification code here */
        }
    }

    printf("Time taken: %3.3lf s.\n", d_endTime - d_startTime);
    printf("Performance: %3.3lf billion cell updates/s\n", 
                (1.0*m*m)*k/((d_endTime - d_startTime)*1e9));

    /* free memory */
    free(grid_current); free(grid_next);

    return 0;
}