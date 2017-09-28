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

template <typename T>
void DumpCudaMatrix(T* src, int height, int width) {
  T* dest;
  dest = (T*)malloc(sizeof(T) * height * width);
  checkCudaErrors(cudaMemcpy(dest, src, sizeof(T) * height * width, cudaMemcpyDeviceToHost));
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      printf("%d,", (int)dest[j + i * width]);
    }
    std::cout << std::endl;
  }
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
        h_sourceImg[i][j].x = j + i * 100;
        h_sourceImg[i][j].y = j + i * 100;
        h_sourceImg[i][j].z = j + i * 100;
        h_sourceImg[i][j].w = j + i * 100;
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

#define TEST_WARPER(func) \
  void gtest_##func() { \
    PossionBlendingTest tester; \
    tester.SetUp(); \
    func(tester); \
    tester.TearDown(); \
  } \

void compute_mask_kernel_test(PossionBlendingTest& tester) {
  uchar4 n1_ = make_uchar4(255, 255, 255, 255);
  uchar4 n0_ = make_uchar4(250, 250, 250, 250);
  uchar4 h_sourceImg[10][10] = {
    n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_,
    n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_,
    n1_,n1_,n0_,n0_,n0_,n0_,n0_,n1_,n1_,n1_,
    n1_,n0_,n0_,n0_,n0_,n0_,n0_,n0_,n0_,n1_,
    n1_,n0_,n0_,n0_,n0_,n0_,n0_,n0_,n0_,n1_,
    n1_,n0_,n0_,n0_,n0_,n0_,n0_,n0_,n0_,n1_,
    n1_,n1_,n0_,n0_,n0_,n0_,n0_,n0_,n0_,n1_,
    n1_,n1_,n1_,n0_,n0_,n0_,n0_,n1_,n1_,n1_,
    n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_,
    n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_,n1_
  };

  uchar4* d_sourceImg;
  checkCudaErrors(cudaMalloc(&d_sourceImg, 100 * sizeof(uchar4)));
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, 100 * sizeof(uchar4), cudaMemcpyHostToDevice));

  compute_masks_kernel<<<dim3(3, 3, 1), dim3(32, 32, 1)>>>(d_sourceImg, 10, 10, tester.d_mask);
  DumpCudaMatrix<char>(tester.d_mask, 10, 10);
  checkCudaErrors(cudaFree(d_sourceImg));

}

void ComputeSum2Kernel_test(PossionBlendingTest& tester) {
  uchar4* sum2;
  checkCudaErrors(cudaMalloc(&sum2, sizeof(uchar4) * 10000));
  ComputeSum2Kernel<<<tester.mgridDim, tester.mblockDim>>>(tester.d_sourceImg, sum2, 100, 100);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  DumpUchar4(sum2, 10000, "sum2");
  checkCudaErrors(cudaFree(sum2));
}

void JacobiKernel_test(PossionBlendingTest& tester) {
}

void InitBuffer_test(PossionBlendingTest& tester) {
}

void CopyResult_test(PossionBlendingTest& tester) {
}

void TestBlend(int argc, char** argv) {
  char* sourceImg = argv[1];
  char* destImg = argv[2];
}

TEST_WARPER(compute_mask_kernel_test);
TEST_WARPER(ComputeSum2Kernel_test);
TEST_WARPER(JacobiKernel_test);
TEST_WARPER(InitBuffer_test);
TEST_WARPER(CopyResult_test);

int main() {
  gtest_compute_mask_kernel_test();
  return 0;
}
