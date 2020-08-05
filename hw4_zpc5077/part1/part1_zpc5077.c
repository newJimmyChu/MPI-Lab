#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include "mpi.h"


int error_sum;

static double get_walltime() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}

// size = m / p
void Do_Calculation(int **grid_current, int **grid_next, int rank, int p, int m) {
    int a = m/p;
    for (int i=1; i<a+1; i++) {
        for (int j=1; j<m-1; j++) {
            /* avoiding conditionals inside inner loop */
            int prev_state = grid_current[i][j];
            int num_alive  = 
                            grid_current[i  ][j-1] + 
                            grid_current[i  ][j+1] + 
                            grid_current[i-1][j-1] + 
                            grid_current[i-1][j  ] + 
                            grid_current[i-1][j+1] + 
                            grid_current[i+1][j-1] + 
                            grid_current[i+1][j  ] + 
                            grid_current[i+1][j+1];

            grid_next[i][j] = prev_state * ((num_alive == 2) + (num_alive == 3)) + (1 - prev_state) * (num_alive == 3);
        }
    }

    int **grid_tmp  = grid_next;
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
    if(m < 500){
        printf("The Testing code does not support when m < 500\n");
        printf("Please Try another m that m > 500\n");
        exit(1);
    }

    double d_startTime = 0.0, d_endTime = 0.0;
	d_startTime = get_walltime();

    int rank = 0; 
    int left, right;
    int p = 0;  // The number of processors
    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
    MPI_Status stat;


    /* MPI code region*/
    int count = 100;   
    int a = m / p;   
    // Initialize 2d array with size m * ((m/p) + 2)
    int **proc_grid_current = (int **)malloc((a+2) * sizeof(int *));
    int **proc_grid_next = (int **)malloc((a+2) * sizeof(int *));

    int *recv_buf_right = (int *)malloc(m * sizeof(int));
    int *recv_buf_left = (int *)malloc(m * sizeof(int));
    int *send_buf_right = (int *)malloc(m * sizeof(int));
    int *send_buf_left = (int *)malloc(m * sizeof(int));
    for(int i = 0; i < (a + 2); i++){
        proc_grid_current[i] = (int *)malloc(m * sizeof(int));
        proc_grid_next[i] = (int *)malloc(m * sizeof(int));
    }
    // Initialize the matrix

    for(int i = 0; i < a + 2; i++){
        for(int j = 0; j < m; j++){
            proc_grid_current[i][j] = 0;
            proc_grid_next[i][j] = 0;
        }
    }

    if(rank == 1){
        proc_grid_current[1][2] = 1;
        proc_grid_current[1][3] = 1;
        proc_grid_current[2][3] = 1;
        proc_grid_current[1][4] = 1;
    }
    if(rank == 0){
        proc_grid_current[a][3] = 1;
        proc_grid_current[a][4] = 1;
    }


    if(rank == 0){
        proc_grid_current[2][2] = 1;
        proc_grid_current[3][3] = 1;
        proc_grid_current[4][1] = 1;
        proc_grid_current[4][2] = 1;
        proc_grid_current[4][3] = 1;
    }
    // Initialize the send and receive buffer
        for(int i = 0; i < m; i++){
            recv_buf_right[i] = 0;
            recv_buf_left[i] = 0;
            send_buf_right[i] = 0;
            send_buf_left[i] = 0;
        }


    if(rank > 0)
        left = rank - 1;
    else
        left = MPI_PROC_NULL;
    if(rank < p-1)
        right = rank + 1;
    else 
        right = MPI_PROC_NULL;

    // Begin doing the function for each process
    for(int i = 0; i < k; i++){

        // 1.Assign updated value to send buffer
        for(int i = 0; i < m; i++){
            send_buf_right[i] = proc_grid_current[a][i];
            send_buf_left[i] = proc_grid_current[1][i];
        }
        // 2.Connecting left and right nodes
        // First we can do 0->1->2->3->4->5->6 (forwarding)
        MPI_Sendrecv(send_buf_right, m, MPI_INT, right, 0, recv_buf_left, 
                     m, MPI_INT, left, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
        
        //Second we can do 6->5->4->3->2->1->0 (backwarding)
        MPI_Sendrecv(send_buf_left, m, MPI_INT, left, 0, recv_buf_right,
                     m, MPI_INT, right, MPI_ANY_TAG, MPI_COMM_WORLD, &stat); 
        
        
        // 3.Updating neighbor blocks
        for(int i = 0; i < m; i++){
            proc_grid_current[0][i] = recv_buf_left[i];
            proc_grid_current[a+1][i] = recv_buf_right[i];
        }

        //MPI_Barrier(MPI_COMM_WORLD)   


        for (int i=1; i<a+1; i++) {
            for (int j=1; j<m-1; j++) {
             //avoiding conditionals inside inner loop 
                int prev_state = proc_grid_current[i][j];
                int num_alive  = 
                            proc_grid_current[i  ][j-1] + 
                            proc_grid_current[i  ][j+1] + 
                            proc_grid_current[i-1][j-1] + 
                            proc_grid_current[i-1][j  ] + 
                            proc_grid_current[i-1][j+1] + 
                            proc_grid_current[i+1][j-1] + 
                            proc_grid_current[i+1][j  ] + 
                            proc_grid_current[i+1][j+1];

                proc_grid_next[i][j] = prev_state * ((num_alive == 2) + (num_alive == 3)) + (1 - prev_state) * (num_alive == 3);
            }
        }
        int **grid_tmp  = proc_grid_next;
        proc_grid_next = proc_grid_current;
        proc_grid_current = grid_tmp;

        // Update shared region between processes


    }

    /* serial code */
    /* considering only internal cells */
    d_endTime = get_walltime();

    /* Verify */
    int verify_failed = 0;

    if(rank == 0 || rank == 1)
    {
        int error = 0;
        int **matrix = (int **)malloc(22 * sizeof(int *));
        int **matrix_next = (int **)malloc(22 * sizeof(int *));
        for(int i = 0; i < 22; i++){
            matrix[i] = (int *)malloc(22 * sizeof(int));
            matrix_next[i] = (int *)malloc(22 * sizeof(int));
        }
        for(int i = 0; i < 22; i++){
            for(int j = 0; j < 22; j++){
                matrix[i][j] = 0;
                matrix_next[i][j] = 0;
            }
        }
        matrix[11][2] = 1;
        matrix[11][3] = 1;
        matrix[12][3] = 1;
        matrix[10][3] = 1;
        matrix[10][4] = 1;
        matrix[11][4] = 1;

        for(int i = 0; i < k; i++){
            for (int i=1; i < 21; i++) {
                for (int j=1; j< 21; j++) {
                    int prev_state = matrix[i][j];
                    int num_alive  = 
                            matrix[i  ][j-1] + 
                            matrix[i  ][j+1] + 
                            matrix[i-1][j-1] + 
                            matrix[i-1][j  ] + 
                            matrix[i-1][j+1] + 
                            matrix[i+1][j-1] + 
                            matrix[i+1][j  ] + 
                            matrix[i+1][j+1];

                matrix_next[i][j] = prev_state * ((num_alive == 2) + (num_alive == 3)) + (1 - prev_state) * (num_alive == 3);
            }
        }
        int **tmp = matrix_next;
        matrix_next = matrix;
        matrix = tmp;
        }
        if(rank == 0){
            for(int i = 0; i < 11; i++){
                for(int j = 0; j < 22; j++){
                    if(proc_grid_current[i+(a-10)][j] != matrix[i][j]){
                        error++;
                        error_sum++;
                    }
                }
            }
        }
        else if(rank == 1){
            for(int i = 11; i < 22; i++){
                for(int j = 0; j < 22; j++){
                    if(proc_grid_current[i-10][j] != matrix[i][j]){
                        error++;
                        error_sum++;
                    }
                }
            }
        }
        printf("Starting Test ------------------------------------------\n");
        printf("Checking if the adjacent processes sharing data properly \n");
        printf("%s\n", error == 0? "The Test Case is Correct!!" : "The Test Case is incorrect!");

        for(int i = 0; i < 22; i++){
            free(matrix[i]);
            free(matrix_next[i]);
        }
        free(matrix); free(matrix_next);
    }

    MPI_Get_count(&stat, MPI_CHAR, &count);
    printf("Task %d: Received %d char(s) from task %d with tag %d \n",
          rank, count, stat.MPI_SOURCE, stat.MPI_TAG);



    printf("Time taken: %3.3lf s.\n", d_endTime - d_startTime);
    printf("Performance: %3.3lf billion cell updates/s\n", 
                (1.0*m*m)*k/((d_endTime - d_startTime)*1e9));
    
    for(int i = 0; i < (a + 2); i++){
        free(proc_grid_current[i]);
        free(proc_grid_next[i]);
    }
    free(proc_grid_current); free(proc_grid_next);
    free(recv_buf_right); free(recv_buf_left);
    free(send_buf_right); free(send_buf_left);
    

    MPI_Finalize();
    if(rank == 0){
        printf("\n");
        printf("-------------------------\n");
        printf("Test Result:\n");
        printf("Number of errors: %d\n", error_sum);
        printf("%s\n", error_sum == 0? "All Test Cases are Correct!!" : "Not all Test Cases Passed");
    }
    return 0;
}