#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

const int W = 40;
const int H = 12;

__global__ void draw(char
	*odata)
{
	const char tmp_str[H][W] =
	  { ":::::::::::::::::::::::::::::::::::::::",
		":                                     :",
		":                                     :",
		":                                     :",
		":                                     :",
		":                 ####             <| :",
		":               ######              | :",
		":             ########              | :",
		":           ##########              | :",
		":         ############              | :",
		":       ##############              # :",
		":::::::::::::::::::::::::::::::::::::::" };
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < W && y < H) {
		char c;
		if (x == W - 1 && y != H - 1)
			c = '\n';
		else
			c = tmp_str[y][x];
		odata[y * W + x] = c;
	}
}
int main(void)
{
	char *h_data, *d_data;
	const int strlen = W*H;
	size_t strsize = strlen * sizeof(char);
	h_data = (char *)malloc(strsize);
	memset(h_data, 0, strlen);
	cudaMalloc((void **)&d_data, strsize);
	cudaMemcpy(d_data, h_data, strsize, cudaMemcpyHostToDevice);
	dim3 blocksize = dim3(16, 12, 1);
	dim3 nblock = dim3((W - 1) / 16 + 1, (H - 1) / 12 + 1, 1);
	draw <<<nblock, blocksize>>>(d_data);	
	cudaMemcpy(h_data, d_data, strlen, cudaMemcpyDeviceToHost);
	printf("%s", h_data);
	free(h_data);
	cudaFree(d_data);
}