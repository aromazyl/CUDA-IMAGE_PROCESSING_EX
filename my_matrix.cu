#include <cuda.h>
#include <cstdio>
#include "matrix.h"
#include "utils.h"

#define BLOCK_SIZE 32
#define PATCH(n) ((n) + (BLOCK_SIZE - (n) % BLOCK_SIZE))
#define DECLARE(Type) template void ComputeMatrixMul<Type>(const Matrix<Type>&, const Matrix<Type>&, Matrix<Type>&);

template <typename T>
__global__ void ComputeMatrixKernel(
    Matrix<T> matrixA,
    Matrix<T> matrixB,
    Matrix<T> matrixC) {
  __shared__ T shared_matrixA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ T shared_matrixB[BLOCK_SIZE][BLOCK_SIZE];

  float Csum = 0.0f;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int midy = threadIdx.y + blockIdx.y * blockDim.y;

  // if (midx >= matrixB.col_nums || midy >= matrixA.row_nums) return;

  int N = matrixA.col_nums / BLOCK_SIZE;
  if (N * BLOCK_SIZE < matrixA.col_nums) N += 1;
  for (int i = 0; i < N; ++i) {
    shared_matrixA[tidy][tidx] =
      matrixA.elements[BLOCK_SIZE * i + tidx + midy * matrixA.col_nums];
    shared_matrixB[tidy][tidx] =
      matrixB.elements[(BLOCK_SIZE * i + tidy) * matrixB.col_nums + midx];
    __syncthreads();
    for (int j = 0; j < BLOCK_SIZE; ++j) {
      Csum += shared_matrixA[tidy][j] * shared_matrixB[j][tidx];
    }

    __syncthreads();
  }
  if (midx >= matrixC.col_nums || midy >= matrixC.row_nums) return;
  matrixC.elements[midy * matrixC.col_nums + midx] = Csum;
}

template <typename T>
void ComputeMatrixMul(
    const Matrix<T>& matrixA,
    const Matrix<T>& matrixB,
    Matrix<T>& matrixC
    ) {
  Matrix<T> d_matrixC, d_matrixA, d_matrixB;

  int A_Row = PATCH(matrixA.row_nums);
  int A_Col = PATCH(matrixA.col_nums);
  int B_Row = PATCH(matrixB.row_nums);
  int B_Col = PATCH(matrixB.col_nums);
  checkCudaErrors(cudaMalloc(&d_matrixC.elements, sizeof(T) * A_Row * B_Col));
  checkCudaErrors(cudaMalloc(&d_matrixA.elements, sizeof(T) * A_Row * A_Col));
  checkCudaErrors(cudaMalloc(&d_matrixB.elements, sizeof(T) * B_Row * B_Col));
  checkCudaErrors(cudaMemset(d_matrixA.elements, T(0), sizeof(T) * A_Row * A_Col));
  checkCudaErrors(cudaMemset(d_matrixB.elements, T(0), sizeof(T) * B_Row * B_Col));
  checkCudaErrors(cudaMemcpy(d_matrixA.elements, matrixA.elements,
        sizeof(T) * matrixA.col_nums * matrixA.row_nums, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_matrixB.elements, matrixB.elements,
        sizeof(T) * matrixB.col_nums * matrixB.row_nums, cudaMemcpyHostToDevice));
  d_matrixA.col_nums = matrixA.col_nums;
  d_matrixA.row_nums = matrixA.row_nums;
  d_matrixB.col_nums = matrixB.col_nums;
  d_matrixB.row_nums = matrixB.row_nums;
  d_matrixC.row_nums = matrixA.row_nums;
  d_matrixC.col_nums = matrixB.col_nums;

  int gx = (matrixC.col_nums % BLOCK_SIZE) ? 1 : 0;
  int gy = (matrixC.row_nums % BLOCK_SIZE) ? 1 : 0;
  dim3 gridDim(matrixC.col_nums / BLOCK_SIZE + gx, matrixC.row_nums / BLOCK_SIZE + gy, 1);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
  ComputeMatrixKernel<T><<<gridDim, blockDim>>>(d_matrixA, d_matrixB, d_matrixC);
  checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy(matrixC.elements, d_matrixC.elements,
        sizeof(T) * matrixC.row_nums * matrixC.col_nums, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_matrixA.elements));
  checkCudaErrors(cudaFree(d_matrixB.elements));
  checkCudaErrors(cudaFree(d_matrixC.elements));
}


DECLARE(int)
DECLARE(float)
