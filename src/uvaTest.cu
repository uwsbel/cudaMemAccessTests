#include <string>
#include <iostream>
#include "oddsEnds.h"

extern __global__ void sequence_gpu(int *d_ptr, int length);

int test_UVA()
{
	using namespace std;

    cout << "Running Unified Virtual Addressing test..." << endl;

    const int N = 100;

    int *h_d_ptr, *h_ptr;
    ASSERT(cudaSuccess == cudaMallocHost(&h_d_ptr, N * sizeof(int)), "Host allocation of "   << N << " ints failed", -1);
    ASSERT(cudaSuccess == cudaMallocHost(&h_ptr, N * sizeof(int)), "Host allocation of "   << N << " ints failed", -1);

    cout << "Memory allocated successfully" << endl;

    dim3 cudaBlockSize(32,1,1);
    dim3 cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);
    sequence_gpu<<<cudaGridSize, cudaBlockSize>>>(h_d_ptr, N);
    ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
    ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);

    sequence_cpu(h_ptr, N);

    cout << "CUDA and CPU algorithm implementations finished" << endl;

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
