#include "possion_blending.cu"

#include <iostream>

template <typename T>
void DumpCudaMemInfo(T* src, int size, const char* tag) {
  T* dest;
  dest = (T*)malloc(sizeof(T) * size);
  checkCudaErrors(cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; ++i) {
    std::cout << tag << "[" << i << "]=" <<
      static_cast<int>(dest[i]) << ";\t";
  }
  printf("\n");
  free(dest);
}

void DumpUchar4(uchar4* src, int size, const char* tag) {
  uchar4* dest;
  dest = (uchar4*)calloc(sizeof(uchar4), size);
  checkCudaErrors(cudaMemcpy(dest, src, sizeof(uchar4) * size, cudaMemcpyDeviceToHost));
  for (int i  = 0; i < size; ++i) {
    printf("dest[%d]={%u,%u,%u,%u}; ", i, dest[i].x, dest[i].y, dest[i].z, dest[i].w);
  }
  free(dest);
  printf("\n");
}

template <typename T>
void DumpCpuMemInfo(T* src, int size, const char* tag) {
  for (int i = 0; i < size; ++i) {
    std::cout << tag << "[" << i << "]=" <<
      static_cast<int>(src[i]) << ";\t";
  }
  printf("\n");
}

struct PossionBlendingTest {
  void SetUp() {
    for (int i = 0; i < 100; ++i) {
      for (int j = 0; j < 100; ++j) {
        h_sourceImg[i][j].x = 255;
        h_sourceImg[i][j].y = 255;
        h_sourceImg[i][j].z = 255;
        h_sourceImg[i][j].w = 255;
        h_destImg[i][j].x = i + j * 100;
        h_destImg[i][j].y = i + j * 100;
        h_destImg[i][j].z = i + j * 100;
        h_destImg[i][j].w = i + j * 100;
      }
    }

    this->mgridDim = dim3(100 / 32 + 1, 100 / 32 + 1);
    this->mblockDim = dim3(32, 32);

    checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * 10000));
    checkCudaErrors(cudaMalloc(&d_destImg, sizeof(uchar4) * 10000));
    checkCudaErrors(cudaMalloc(&d_mask, sizeof(char) * 10000));
    checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * 10000, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4) * 10000, cudaMemcpyHostToDevice));
  }

  void TearDown() {
    checkCudaErrors(cudaFree(d_sourceImg));
    checkCudaErrors(cudaFree(d_destImg));
    checkCudaErrors(cudaFree(d_mask));
  }

  uchar4 h_sourceImg[100][100];
  uchar4 h_destImg[100][100];

  uchar4* d_sourceImg;
  uchar4* d_destImg;
  char* d_mask;
  dim3 mgridDim;
  dim3 mblockDim;
};

int main() {
  PossionBlendingTest tester;
  tester.SetUp();
  compute_masks_kernel<<<tester.mgridDim, tester.mblockDim>>>(tester.d_sourceImg, 100, 100, tester.d_mask);
  DumpCudaMemInfo<char>(tester.d_mask, 10000, "mask");
  DumpUchar4(tester.d_sourceImg, 10000, "sourceImg");
  tester.TearDown();
}
