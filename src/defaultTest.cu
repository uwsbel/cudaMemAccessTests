#include <iostream>
#include "oddsEnds.h"


__global__ void sequence_gpu(int *d_ptr, int length)
{
    int elemID = blockIdx.x * blockDim.x + threadIdx.x;

    if (elemID < length)
    {
        d_ptr[elemID] = elemID;
    }
}



int default_test()
{
    using namespace std;
    
    cout << "CUDA Runtime API template" << endl;
    cout << "=========================" << endl;
    cout << "Self-test started" << endl;

    const int N = 100;

    int *d_ptr;
    ASSERT(cudaSuccess == cudaMalloc(&d_ptr, N * sizeof(int)), "Device allocation of " << N << " ints failed", -1);

    int *h_ptr;
    ASSERT(cudaSuccess == cudaMallocHost(&h_ptr, N * sizeof(int)), "Host allocation of "   << N << " ints failed", -1);

    cout << "Memory allocated successfully" << endl;

    dim3 cudaBlockSize(32,1,1);
    dim3 cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);
    sequence_gpu<<<cudaGridSize, cudaBlockSize>>>(d_ptr, N);
    ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
    ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);

    sequence_cpu(h_ptr, N);

    cout << "CUDA and CPU algorithm implementations finished" << endl;

    int *h_d_ptr;
    ASSERT(cudaSuccess == cudaMallocHost(&h_d_ptr, N *sizeof(int)), "Host allocation of " << N << " ints failed", -1);
    ASSERT(cudaSuccess == cudaMemcpy(h_d_ptr, d_ptr, N *sizeof(int), cudaMemcpyDeviceToHost), "Copy of " << N << " ints from device to host failed", -1);
    bool bValid = true;

    for (int i=0; i<N && bValid; i++)
    {
        if (h_ptr[i] != h_d_ptr[i])
        {
            bValid = false;
        }
    }

    ASSERT(cudaSuccess == cudaFree(d_ptr),       "Device deallocation failed", -1);
    ASSERT(cudaSuccess == cudaFreeHost(h_ptr),   "Host deallocation failed",   -1);
    ASSERT(cudaSuccess == cudaFreeHost(h_d_ptr), "Host deallocation failed",   -1);

    cout << "Memory deallocated successfully" << endl;
    cout << "TEST Results " << endl;

    return bValid;
}