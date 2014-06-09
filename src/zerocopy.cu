#include <string>
#include <iostream>
#include "oddsEnds.h"

extern __global__ void sequence_gpu(int *d_ptr, int length);

int test_zerocopy()
{
	using namespace std;

    cout << "Running zero-copy test..." << endl;

    const int N = 100;

    int *d_ptr;

    int *h_ptr;
    ASSERT(cudaSuccess == cudaHostAlloc(&h_ptr, N * sizeof(int), cudaHostAllocMapped || cudaHostAllocWriteCombined), "Host allocation of "   << N << " ints failed", -1);

    cout << "Memory allocated successfully" << endl;

    ASSERT(cudaSuccess == cudaHostGetDevicePointer(&d_ptr, h_ptr, 0), "Get device pointer failed", -1);

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
        if (h_ptr[i] != h_d_ptr[i])
        {
            bValid = false;
			break;
        }

    ASSERT(cudaSuccess == cudaFreeHost(h_ptr),   "Host deallocation failed",   -1);
    ASSERT(cudaSuccess == cudaFreeHost(h_d_ptr), "Host deallocation failed",   -1);

    cout << "Memory deallocated successfully" << endl;
    cout << "TEST Results: " << bValid << endl;

    return bValid;
}
