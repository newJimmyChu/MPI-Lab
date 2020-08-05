#include <mpi.h>
#include <stdio.h>

/* Run with 12 processes */ 
int main(int argc, char *argv[]) 
{ 
   int rank; 
   MPI_Comm vu; 
   int dim[2],period[2],reorder; 
   int coord[2],id; 

   // Initialize the MPI environment
   MPI_Init(&argc, &argv);     

   // Get the rank of the process
   MPI_Comm_rank(MPI_COMM_WORLD,&rank); 

   dim[0]=4; dim[1]=3; 
   period[0]=true; period[1]=false;   
   reorder=true;   

   // Create a cartesian virtual topology
   MPI_Cart_create(MPI_COMM_WORLD,2,
      dim,period,reorder,&vu); 


   if(rank==5)
   { 
      MPI_Cart_coords(vu,rank,2,coord); 
      printf("P:%d My coordinates are %d   %d\n",rank,coord[0],coord[1]);
   } 

   if(rank==0) 
   {
       coord[0]=3; coord[1]=1; 
       MPI_Cart_rank(vu,coord,&id); 
       printf("The processor at position (%d, %d) has rank %d\n",coord[0],coord[1],id); 
   } 

   // Finalize our MPI environment
   MPI_Finalize(); 
}

