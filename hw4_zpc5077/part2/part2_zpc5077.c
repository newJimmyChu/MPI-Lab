#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include "mpi.h"
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3
// Define macro in clockwise
#define LU    0
#define RU    1
#define RD    2
#define LD    3 

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
    int n = sqrt(m);
    int p = 0;
    if(m < 500){
        printf("The Testing code does not support when m < 500\n");
        printf("Please Try another m that m > 500\n");
        exit(1);
    }

    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int sub_p = sqrt(p);
    if(n * n != m && sub_p* sub_p != p){
        printf("the sqrt of m should be an integer\n");
        exit(1);
    }



    int i, j, t, count;

    double d_startTime = 0.0, d_endTime = 0.0;
	d_startTime = get_walltime();

    printf("p: %d n: %d number of processes per row and col: %d\n", p, n, sub_p);
    // Initializing Cart
    int dims[2];
    dims[0] = sub_p; dims[1] = sub_p;
    int periods[2] = {0, 0};
    int coords[2];
    int reorder = 0;
    int nbrs[4];
    int rank = 0; 
    int left, right;
    MPI_Comm cartcomm;
    MPI_Status stat;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims ,periods ,reorder ,&cartcomm); 
    MPI_Cart_coords(cartcomm, rank, 2, coords);
    MPI_Cart_shift(cartcomm, 0, 1, &nbrs[UP], &nbrs[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);


    /* MPI code region*/  
    // Initialize 2d array with size m * ((m/p) + 2)
    // All of the processes have (n+2) * (n+2) blocks
    int **proc_grid_current = (int **)malloc((n+2) * sizeof(int *));
    int **proc_grid_next = (int **)malloc((n+2) * sizeof(int *));



    // Initialize four recv buffers and four receive buffers
    //int *recv_buf_up = (int *)malloc(n * sizeof(int));
    int *send_buf[4]; int *recv_buf[4];
    for(int i = 0; i < 4; i++){
        send_buf[i] = (int *)malloc(n * sizeof(int));
        recv_buf[i] = (int *)malloc(n * sizeof(int));
    }

    // Initialize four corner blocks
    int send_corner[4];
    int recv_corner[4];
    for(int i = 0; i < 4; i++){
        send_corner[i] = 0;
        recv_corner[i] = 0;
    }

    for(int i = 0; i < (n + 2); i++){
        proc_grid_current[i] = (int *)malloc((n+2) * sizeof(int));
        proc_grid_next[i] = (int *)malloc((n+2) * sizeof(int));
    }
    // Initialize the matrix

    for(int i = 0; i < n + 2; i++){
        for(int j = 0; j < n + 2; j++){
            proc_grid_current[i][j] = 0;
            proc_grid_next[i][j] = 0;
        }
    }

    if(rank == 1){
        proc_grid_current[2][1] = 1;
        proc_grid_current[3][1] = 1;
        proc_grid_current[3][2] = 1;
        proc_grid_current[4][1] = 1;
    }

    // Preset Testing blocks for the ghost region of four process
    if(rank == 0)
        proc_grid_current[n][n] = 1;
    if(rank == 1)
        proc_grid_current[n][1] = 1;
    if(rank == sub_p)
        proc_grid_current[1][n] = 1;
    if(rank == sub_p + 1)
        proc_grid_current[1][1] = 1;

    if(rank == 0){
        proc_grid_current[3][n] = 1;
        proc_grid_current[4][n] = 1;
    }
    // Initialize the send and receive buffer
        for(int i = 0; i < n; i++){
            send_buf[0][i] = 0; send_buf[1][i] = 0;
            send_buf[2][i] = 0; send_buf[3][i] = 0;
            recv_buf[0][i] = 0; recv_buf[1][i] = 0;
            recv_buf[2][i] = 0; recv_buf[3][i] = 0;
        }


    // Begin doing the function for each process
    for(int i = 0; i < k; i++){
        // Update shared region between processes
        
        // 1.Assign updated value to send buffer
        for(int i = 1; i < n+1; i++){
            send_buf[UP][i-1] = proc_grid_current[1][i];
            send_buf[DOWN][i-1] = proc_grid_current[n][i];
            send_buf[LEFT][i-1] = proc_grid_current[i][1];
            send_buf[RIGHT][i-1] = proc_grid_current[i][n];
        }
        send_corner[LU] = proc_grid_current[1][1];
        send_corner[RU] = proc_grid_current[1][n];
        send_corner[LD] = proc_grid_current[n][1];
        send_corner[RD] = proc_grid_current[n][n];

        // 2.Connecting left and right nodes
        // First we can do 0->1->2->3->4->5->6 (forwarding)
        MPI_Sendrecv(send_buf[RIGHT], n, MPI_INT, nbrs[RIGHT], 0, recv_buf[LEFT], 
                     n, MPI_INT, nbrs[LEFT], MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

        //Second we can do 6->5->4->3->2->1->0 (backwarding)
        MPI_Sendrecv(send_buf[LEFT], n, MPI_INT, nbrs[LEFT], 0, recv_buf[RIGHT],
                     n, MPI_INT, nbrs[RIGHT], MPI_ANY_TAG, MPI_COMM_WORLD, &stat); 
        
        MPI_Sendrecv(send_buf[UP], n, MPI_INT, nbrs[UP], 0, recv_buf[DOWN],
                     n, MPI_INT, nbrs[DOWN], MPI_ANY_TAG, MPI_COMM_WORLD, &stat); 

        MPI_Sendrecv(send_buf[DOWN], n, MPI_INT, nbrs[DOWN], 0, recv_buf[UP],
                     n, MPI_INT, nbrs[UP], MPI_ANY_TAG, MPI_COMM_WORLD, &stat);                          

        // Send to left up corner

        if(nbrs[UP] > 0 && nbrs[UP] % sub_p != 0)
            MPI_Sendrecv(&send_corner[LU], 1, MPI_INT, nbrs[UP]-1, 0, &recv_corner[LU],
                         1, MPI_INT, nbrs[UP]-1, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

        if(nbrs[UP] >= 0 && nbrs[UP] % sub_p != sub_p - 1 )
            MPI_Sendrecv(&send_corner[RU], 1, MPI_INT, nbrs[UP]+1, 0, &recv_corner[RU],
                         1, MPI_INT, nbrs[UP]+1, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

        // Left Down
        if(nbrs[DOWN] > 0 && nbrs[DOWN] % sub_p != 0)
            MPI_Sendrecv(&send_corner[LD], 1, MPI_INT, nbrs[DOWN]-1, 0, &recv_corner[LD],
                         1, MPI_INT, nbrs[DOWN]-1, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

        if(nbrs[DOWN] >= 0 && nbrs[DOWN] % sub_p != sub_p - 1)
            MPI_Sendrecv(&send_corner[RD], 1, MPI_INT, nbrs[DOWN]+1, 0, &recv_corner[RD],
                         1, MPI_INT, nbrs[DOWN]+1, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

        proc_grid_current[0][0] = recv_corner[LU];
        proc_grid_current[0][n+1] = recv_corner[RU];
        proc_grid_current[n+1][0] = recv_corner[LD];
        proc_grid_current[n+1][n+1] = recv_corner[RD];


        //proc_grid_current[][];

        // 3.Updating neighbor blocks
        for(int i = 1; i < n+1; i++){
            proc_grid_current[0][i] = recv_buf[UP][i-1];
            proc_grid_current[n+1][i] = recv_buf[DOWN][i-1];
            proc_grid_current[i][0] = recv_buf[LEFT][i-1];
            proc_grid_current[i][n+1] = recv_buf[RIGHT][i-1];
            //printf("%d\n", recv_buf[LEFT][i]);
        }     

        for (int i=1; i < n+1; i++) {
            for (int j=1; j< n+1; j++) {
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



    }

    /* serial code */
    /* considering only internal cells */
    d_endTime = get_walltime();


    /* Verify */
    // Verify if the code can go through the ghost region
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
        matrix[2][11] = 1;
        matrix[3][11] = 1;
        matrix[3][12] = 1;
        matrix[3][10] = 1;
        matrix[4][10] = 1;
        matrix[4][11] = 1;

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
            for(int i = 0; i < 22; i++){
                for(int j = 0; j < 11; j++){
                    if(proc_grid_current[i][j+(n-10)] != matrix[i][j])
                        error++;
                }
            }
        }
        else if(rank == 1){
            for(int i = 0; i < 22; i++){
                for(int j = 11; j < 22; j++){
                    if(proc_grid_current[i][j-10] != matrix[i][j])
                        error++;
                }
            }
        }
        printf("Checking if the adjacent processes sharing data properly \n");
        printf("%s\n", error == 0? "The Test Case is Correct!!" : "The Test Case is incorrect!");

        for(int i = 0; i < 22; i++){
            free(matrix[i]);
            free(matrix_next[i]);
        }
        free(matrix); free(matrix_next);
    }
    // Testing if the block located at the center of four connected processes
    // still exists.
    if(rank == 0||rank == 1||rank==sub_p||rank==sub_p+1){
        int block_error = 0;
        if(rank == 0){
            block_error += proc_grid_current[n][n] == 1? 0 : 1; 
            error_sum += proc_grid_current[n][n] == 1? 0 : 1; 
        }
        if(rank == 1){
            block_error += proc_grid_current[n][1] == 1? 0 : 1;
            error_sum += proc_grid_current[n][1] == 1? 0 : 1;  
        }
        if(rank == sub_p){
            block_error += proc_grid_current[1][n] == 1? 0 : 1;
            error_sum += proc_grid_current[1][n] == 1? 0 : 1;   
        }
        if(rank == sub_p + 1){
            block_error += proc_grid_current[1][1] == 1? 0 : 1; 
            error_sum += proc_grid_current[1][1] == 1? 0 : 1;  
        }
        printf("Checking if the corner blocks shared by processes properly \n");
        printf("%s\n", block_error == 0? "The Test Case is Correct!!" : "The Test Case is incorrect!");
    }

    MPI_Barrier(MPI_COMM_WORLD);


    MPI_Get_count(&stat, MPI_CHAR, &count);
    printf("Task %d: Received %d char(s) from task %d with tag %d \n",
          rank, count, stat.MPI_SOURCE, stat.MPI_TAG);

    printf("Time taken: %3.3lf s.\n", d_endTime - d_startTime);
    printf("Performance: %3.3lf billion cell updates/s\n", 
                (1.0*n*n)*k/((d_endTime - d_startTime)*1e9));
    
    for(int i = 0; i < (n + 2); i++){
        free(proc_grid_current[i]);
        free(proc_grid_next[i]);
    }

    free(send_buf[0]); free(send_buf[1]); free(send_buf[2]); free(send_buf[3]);
    free(recv_buf[0]); free(recv_buf[1]); free(recv_buf[2]); free(recv_buf[3]);
    free(proc_grid_current); free(proc_grid_next);


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