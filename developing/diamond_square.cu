#include <cuda.h>

__global__ void diamond(curandState_t* rng, float *hm, int rectSize, float dh, int SIZE) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int squaresInRow = SIZE / rectSize;
  int offset = idx % squaresInRow;
  int i = ((idx - offset) * (rectSize * rectSize)) / SIZE;
  int j = offset * rectSize;
  int ni = (i + rectSize) % SIZE;
  int nj = (j + rectSize) % SIZE;
  int mi = (i + rectSize / 2.0f);
  int mj = (j + rectSize / 2.0f);
  float v1 = 0.0f - dh / 2.0f;
  float v2 = dh / 2.0f;
  curandState_t localState = rng[idx];
  int rand = v1 + (v2 - v1) * curand_uniform(&localState);
  float rand = v1 + (v2 - v1) * curand_uniform(&localState);
  hm[mi+mj*SIZE] = (hm[i+j*SIZE] + hm[ni+j*SIZE] + hm[i+nj*SIZE] + hm[ni+nj*SIZE]) / 4.0f + rand;
  rng[idx] = localState;
}

__global__ void square(curandState_t* rng, float* hm, int rectSize, float dh, int SIZE) {

}
