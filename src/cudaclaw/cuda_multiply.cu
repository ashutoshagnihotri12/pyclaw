#include <stdio.h>
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * computes A = A*x
 *
 * where A is a vector of length numElements
 */
__global__ void
cudaMultiply(double *A, const double *x, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        A[i] = A[i]*x[0];
    }
}

extern "C" {
void cuda_multiply (double* array, double multiplier, int m, int n) {

    cudaError_t err = cudaSuccess;

    // Allocate the device vector A
    double *d_A = NULL;

    // Allocate the device multiplier x
    double *d_x = NULL;

    int numElements = m*n;

    size_t size = numElements * sizeof(double);

    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    err = cudaMalloc((void **)&d_x, sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device multiplier x (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors array and host memory to the 
    // device input vectors in device memory
    printf("Copy input from the numpy array in host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, array, size, cudaMemcpyHostToDevice);

    printf("Copy multiplier to the CUDA device\n");
    err = cudaMemcpy(d_x, &multiplier, sizeof(double), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy numpy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", 
    	   blocksPerGrid, threadsPerBlock);
    cudaMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_x, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch cudaMultiply kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the numpy vector
    // in host memory.
    printf("Copy output data from the CUDA device to the numpy array in host memory\n");
    err = cudaMemcpy(array, d_A, size, cudaMemcpyDeviceToHost);

    return ;
}
};
