#include "lab3.h"
#include "cuda_runtime.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a - 1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
		const float *background,
		const float *target,
		const float *mask,
		float *output,
		const int wb, const int hb, const int wt, const int ht,
		const int oy, const int ox
		)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (0 <= yt && 0 <= xt && yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void init(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int bWidth, const int bHeight,
	const int tWidth, const int tHeight,
	const int Oy, const int Ox
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int Ct = tWidth*yt + xt;
	if (yt >= 0 && xt >= 0 && yt < tHeight && xt < tWidth) {
		const int yb = Oy + yt, xb = Ox + xt;
		const int Cb = bWidth*yb + xb;
		if (0 <= yb && yb < bHeight && 0 <= xb && xb < bWidth) {
			if (mask[Ct] < 127.0f) {
				// set fixed to background
				for (int c = 0; c < 3; c++) {
					fixed[3 * Ct + c] = background[3 * Cb + c];
				}
			}
			else {
				int Nt = tWidth*(yt - 1) + xt;
				int St = tWidth*(yt + 1) + xt;
				int Wt = tWidth*yt + (xt - 1);
				int Et = tWidth*yt + (xt + 1);

				int Nb = bWidth*(yb - 1) + xb;
				int Sb = bWidth*(yb + 1) + xb;
				int Wb = bWidth*yb + (xb - 1);
				int Eb = bWidth*yb + (xb + 1);

				//Calculation
				for (int c = 0; c < 3; c++) {
					float CbPrime = 0.0f;
					if (yt > 0)
						CbPrime += target[3 * Ct + c] - target[3 * Nt + c];
					if (yt < tHeight - 1)
						CbPrime += target[3 * Ct + c] - target[3 * St + c];
					if (xt > 0)
						CbPrime += target[3 * Ct + c] - target[3 * Wt + c];
					if (xt < tWidth - 1)
						CbPrime += target[3 * Ct + c] - target[3 * Et + c];

					//solve boundary problems
					float boundary = 0.0f;
					if (yt == 0 || mask[Nt] < 127.0f)
						boundary += background[3 * Nb + c];
					if (yt == tHeight - 1 || mask[St] < 127.0f)
						boundary += background[3 * Sb + c];
					if (xt == 0 || mask[Wt] < 127.0f)
						boundary += background[3 * Wb + c];
					if (xt == tWidth - 1 || mask[Et] < 127.0f)
						boundary += background[3 * Eb + c];

					fixed[3 * Ct + c] = CbPrime + boundary;

				}
			}
		}

	}
}

__global__ void JacobiIteration(
	float *fixed,
	const float *mask,
	float *buf1, float *buf2,
	int tWidth, int ht
	)
{
	const int yt = (blockIdx.y * blockDim.y + threadIdx.y);
	const int xt = (blockIdx.x * blockDim.x + threadIdx.x);
	const int Ct = tWidth*yt + xt;
	if (0 <= yt && 0 <= xt && yt < ht && xt < tWidth) {
		if (mask[Ct] > 127.0f) {
			int Nt = tWidth*(yt - 1) + xt;
			int St = tWidth*(yt + 1) + xt;
			int Wt = tWidth*yt + (xt - 1);
			int Et = tWidth*yt + (xt + 1);

			//Calculation
			for (int c = 0; c < 3; c++) {
				float sum = 0.0f;
				if (yt > 0 && mask[Nt] > 127.0f)
					sum += buf1[3 * Nt + c];
				if (yt < ht - 1 && mask[St] > 127.0f)
					sum += buf1[3 * St + c];
				if (xt > 0 && mask[Wt] > 127.0f)
					sum += buf1[3 * Wt + c];
				if (xt < tWidth - 1 && mask[Et] > 127.0f)
					sum += buf1[3 * Et + c];

				float Cb_next = (sum + fixed[Ct * 3 + c]) / 4.0f;
				buf2[3 * Ct + c] = Cb_next;
			}
		}
	}
}

void PoissonImageCloning(
		const float *background,
		const float *target,
		const float *mask,
		float *output,
		const int wb, const int hb, const int wt, const int ht,
		const int oy, const int ox
		)
{
	float *fixed;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	float *buf1;
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	float *buf2;
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	cudaMemcpy(output, background, wb*hb*sizeof(float) * 3, cudaMemcpyDeviceToDevice);


	init<<<dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16)>>>(output, target, mask, fixed, wb, hb, wt, ht, oy, ox);

	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	int iterCount = 10000;
	for(int i = 0; i < iterCount; i++) {
		//Calculate Jacobi iteration and save target from buffer 1 to buffer 2
		JacobiIteration<<<dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16) >>>(fixed, mask, buf1, buf2, wt, ht);
		//Do second time in reverse direction (buffer 2 to buffer 1) in one iteration so we can save the time for swapping
		JacobiIteration<<<dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16) >>>(fixed, mask, buf2, buf1, wt, ht);
	}

	SimpleClone<<< dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16) >>>(background, buf1, mask, output,	wb, hb, wt, ht, oy, ox);

	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);

}