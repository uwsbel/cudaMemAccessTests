#include <string>
#include <iostream>
#include "oddsEnds.h"

extern __global__ void sequence_gpu(int *d_ptr, int length);

int test_uniformMem()
{
	using namespace std;

    cout << "Running Unified Memory test..." << endl;

    const int N = 100;

    int *m_ptr;
	int *h_ptr;

    ASSERT(cudaSuccess == cudaMallocManaged(&m_ptr, N * sizeof(int)), "Managed allocation of "   << N << " ints failed", -1);
    ASSERT(cudaSuccess == cudaMallocHost(&h_ptr, N * sizeof(int)), "Host allocation of "   << N << " ints failed", -1);

    cout << "Memory allocated successfully" << endl;

    dim3 cudaBlockSize(32,1,1);
    dim3 cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);
    sequence_gpu<<<cudaGridSize, cudaBlockSize>>>(m_ptr, N);
    ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
    ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);

    sequence_cpu(h_ptr, N);

    cout << "CUDA and CPU algorithm implementations finished" << endl;

    bool bValid = true;

    for (int i=0; i<N && bValid; i++)
        if (h_ptr[i] != m_ptr[i]) {
            bValid = false;
			break;
		}

    ASSERT(cudaSuccess == cudaFreeHost(h_ptr),   "Host deallocation failed",   -1);
    ASSERT(cudaSuccess == cudaFree(m_ptr), "Managed deallocation failed",   -1);

    cout << "Memory deallocated successfully" << endl;
    cout << "TEST Results: " << bValid << endl;

    return bValid;
}
