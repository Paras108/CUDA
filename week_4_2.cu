#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


void print_array(float *A, int n, int m)
{
    for(int i=0; i<n; i++)
	{
		for (int j=0; j<m; j++)
		{
			printf("%.1f ", A[i*m+j]);
		}
    printf("\n");
	}
}


__global__ void
process_kernel1(float *input, float *output, int n, int m)
{
    // Code for i
	int i= blockIdx.y * blockDim .y+ threadIdx .y;
	int j= blockIdx.x * blockDim.x+ threadIdx.x;

	if ((i<n) && (j<m)) {
	for(int l=0; l<n; l++){
		for (int k = 0; k < m; k+=2) {
			output[i*l+k] = input[i*l+k+1];
			output[i*l+k+1] = input[i*l+k];
			}
	}
	}
}


int main(void)
{
    cudaError_t err = cudaSuccess;

	int test_cases;
    scanf("%d",&test_cases);
	
	int m, n;
	scanf("%d %d", &m, &n);
	
    size_t size = m*n*sizeof(float);

    float *h_input = (float *)malloc(size);
	float *h_output = (float *)malloc(size);

    if (h_input == NULL || h_output == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n*m; ++i)
    {
        scanf("%f",&h_input[i]);
        
    }

    float *d_input = NULL;
    err = cudaMalloc((void **)&d_input, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    float *d_output = NULL;
    err = cudaMalloc((void **)&d_output, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


   err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector h_input from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	
   //launching process_kernel1
     int threadsPerBlock = 16;
     int blocksPerGrid = ((m*n)+threadsPerBlock-1)/threadsPerBlock; 
    
    process_kernel1<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, m);
	err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel1 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

     /*
	 // Verify that the result vectors are as expected

    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(sinf(h_input1[i]) + cosf(h_input2[i]) - h_output1[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification for h_output1 failed at element %d! value \n", i, h_input1[i]);
            exit(EXIT_FAILURE);
        }
    }
	*/

     print_array(h_output,n,m);
    

    err = cudaFree(d_input);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(h_input);
    free(h_output);

    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   
    return 0;
}

