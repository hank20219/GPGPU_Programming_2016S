#include "lab1.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <wave.h>

static const __constant__ unsigned W = 640;
static const __constant__ unsigned H = 480;
static const __constant__ unsigned NFRAME = 240 * 6;

__global__
void draw(uint8_t *odata, float time)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < W * H) {
		float4 fragColor;
		float2 fragCoord = { x % W + 1, H - (x / W) };
		mainImage(fragColor, fragCoord, time);
		odata[x] = 0.299f * fragColor.x + 0.587f * fragColor.y + 0.114f * fragColor.z;
		if ((x % W) % 2 == 0 && (x / W) % 2 == 0) {
			int ustartx = W * H + ((x / 2) - ((x / W) / 2)*(W / 2));
			int vstartx = W * H * 1.25 + ((x / 2) - ((x / W) / 2)*(W / 2));
			odata[ ustartx ] = -0.169f * fragColor.x + -0.331f * fragColor.y + 0.500f * fragColor.z + 128;
			odata[ vstartx ] = 0.500f * fragColor.x + -0.419f * fragColor.y + -0.081f * fragColor.z + 128;
		}
	}

}

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab1VideoGenerator::Generate(uint8_t *yuv) {

	
	const int strlen = W*H;
	size_t videoSize = strlen * sizeof(char);
	dim3 blocksize = dim3(12, 1, 1);
	dim3 nblock = dim3((W * H - 1) / 12 + 1, 1, 1);
	draw <<<nblock, blocksize>>>(yuv, impl->t);
	
	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	//cudaMemset(yuv+W*H, 128, W*H/2);
	++(impl->t);
}
