#include <cuda_runtime.h>
#include <cuda.h>
#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct Count1 : public thrust::unary_function<const char, int>
{
	__device__  __host__ int operator()(const char& c,const int& pos) const
	{
		if (c == '\n')
			return 0;
		else
			return 1;
	}
};



void CountPosition1(const char *text, int *pos, int text_size)
{
	thrust::device_ptr<const char> textPtr(text);
	thrust::device_ptr<int> resultPtr(pos);

	thrust::transform(thrust::device, textPtr, textPtr + text_size, resultPtr, resultPtr, Count1());
	thrust::inclusive_scan_by_key(resultPtr, resultPtr + text_size, resultPtr, resultPtr);
	
}

__global__ void count2(const char *input, int *output, int size) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size) {
		if (x == 0) {
			if (input[x] != '\n') {
				for (int i = 0; x + i < size; i++) {
					if (input[x + i] == '\n')
						break;
					else
						output[x + i] = i + 1;
				}
			}
		}
		else {
			if (input[x - 1] == '\n') {
				for (int i = 0; x + i < size; i++) {
					if (input[x + i] == '\n')
						break;
					else
						output[x + i] = i + 1;
				}
			}

		}
	}
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	count2 << <text_size/128, 128 >> >(text, pos, text_size);


}

