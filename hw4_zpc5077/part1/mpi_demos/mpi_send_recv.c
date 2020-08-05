#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
   // Initialize the MPI environment. The two arguments to MPI Init are not
   // currently used by MPI implementations, but are there in case future
   // implementations might need the arguments.
   MPI_Init(NULL, NULL);

   // Get the number of processes
   int world_size;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

   // Get the rank of the process
   int world_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

   // Get the name of the processor
   char processor_name[MPI_MAX_PROCESSOR_NAME];
   int name_len;
   MPI_Get_processor_name(processor_name, &name_len);

   int i_number;
   if (world_rank == 0)
   {
      i_number = -1;
      MPI_Send(&i_number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
   }
   else if (world_rank == 1)
   {
      MPI_Recv(&i_number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("Process 1 received number %d from process 0\n", i_number);
   }

   // Finalize the MPI environment. No more MPI calls can be made after this
   MPI_Finalize();
}
