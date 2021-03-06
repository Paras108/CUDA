%%cu
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define LOG_INPUT if(0)
#define LOG_OUTPUT if(1)
#define LOG if(0)


__global__
void hadamard(float *A, float *B, float *C, int M, int N)
{
	int i= blockIdx.y * blockDim.y+ threadIdx.y;
	int j= blockIdx.x * blockDim.x+ threadIdx.x;
	
	if ((i<N) && (j<N)) {
		for (int k = 0; k < N; ++k) {
      C[i*N+k] = A[i*N+k] * B[k*N+k];
	}
	}
}

/**
 * Host main routine
 */
void print_matrix(float *A,int m,int n)
{
    for(int i =0;i<m;i++)
    {
        for(int j=0;j<n;j++)
            printf("%.2f ",A[i*n+j]);
        printf("\n");
    }

}
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    
    int t; //number of test cases
    scanf("%d",&t);
    while(t--)
    {
        int m,n;
        scanf("%d %d",&m,&n);
        size_t size = m*n*sizeof(float);
        LOG printf("[Hadamard product of two matrices ]\n");

        
        // Allocate the host input vector A
        float *h_A = NULL;
        h_A =( float *) malloc (size);

        // Allocate the host input vector B
        float *h_B = NULL;
        h_B =( float *) malloc (size);
        
        // Allocate the host output vector C
        float *h_C = NULL;
        h_C =( float *) malloc (size);

        // Verify that allocations succeeded
        if (h_A == NULL || h_B == NULL || h_C == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        // Initialize the host input vectors
        
        for (int i = 0; i < n*m; ++i)
        {
            scanf("%f",&h_A[i]);
            scanf("%f",&h_B[i]);
        }
        
        
        // Allocate the device input vector A
        float *d_A = NULL;
        cudaMalloc (( void **) &d_A , size );

        // Allocate the device input vector B
        float *d_B = NULL;
        cudaMalloc (( void **) &d_B , size );

        // Allocate the device output vector C
        float *d_C = NULL;
        cudaMalloc (( void **) &d_C , size );

        // Copy the host input vectors A and B in host memory to the device input vectors in
        // device memory
        cudaMemcpy (d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy (d_B, h_B, size, cudaMemcpyHostToDevice);

        
        // initialize blocksPerGrid and threads Per Block
        dim3 grid = (2,2,1);
        dim3 block =(2,2,1) ;
        //printf ("CUDA kernel launch with %d grid of %d blocks \n",threadsPerBlock, blocksPerGrid );


        hadamard<<<grid, block>>>(d_A, d_B, d_C, m, n);
        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch hadamard kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the device result vector in device memory to the host result vector
        // in host memory.
        err = cudaMemcpy (h_C , d_C , size , cudaMemcpyDeviceToHost );
        if ( err != cudaSuccess){
          fprintf ( stderr , " Failed to copy vector C from device to host ( error code %s)!\n", cudaGetErrorString ( err));
          exit ( EXIT_FAILURE );
        }
        
        // Verify that the result vector is correct
        for (int i = 0; i < n*m; ++i)
        {
            if (fabs(h_A[i] * h_B[i] - h_C[i]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }
        }

        LOG printf("Test PASSED\n");

        // Free device global memory
        cudaFree (d_A ); cudaFree ( d_B); cudaFree ( d_C );

        // Free host memory
        free(h_A); free(h_B); free(h_C);
        
        
        err = cudaDeviceReset();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        print_matrix(h_C,m,n);
        
        LOG printf("Done\n");
    }
    return 0;
}