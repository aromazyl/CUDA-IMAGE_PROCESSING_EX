/*
 * my_matrix_test.cc
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "utils.h"
#include "matrix.h"
#include "my_matrix.hpp"

// #define DECLARE(Type) extern template void ComputeMatrixMul<Type>(const Matrix<Type>&, const Matrix<Type>&, Matrix<Type>&);

// DECLARE(float)

template <typename T>
void ComputeMatrixMul(const Matrix<T>&, const Matrix<T>&, Matrix<T>&);

template <typename T>
void DumpMatrix(const Matrix<T>& matrix) {
  for (int i = 0; i < matrix.row_nums; ++i) {
    for (int j = 0; j < matrix.col_nums; ++j) {
      printf("[%d][%d]{%f};", i, j, matrix.elements[i * matrix.col_nums + j]);
    }
    printf("\n");
  }
}

int main() {
  Matrix<float> A, B, C1, C2;
  A.col_nums = 3;
  A.row_nums = 2;
  B.col_nums = 2;
  B.row_nums = 3;
  C1.col_nums = B.col_nums;
  C1.row_nums = A.row_nums;
  C2.col_nums = B.col_nums;
  C2.row_nums = A.row_nums;

  A.elements = (float*)calloc(sizeof(float), A.col_nums * A.row_nums);
  B.elements = (float*)calloc(sizeof(float), B.col_nums * B.row_nums);
  C1.elements = (float*)calloc(sizeof(float), B.col_nums * A.row_nums);
  C2.elements = (float*)calloc(sizeof(float), B.col_nums * A.row_nums);

  for (int i = 0; i < A.col_nums; ++i) {
    for (int j = 0; j < A.row_nums; ++j) {
      A.elements[j * A.col_nums + i] = j * A.col_nums + i + 1;
    }
  }

  for (int i = 0; i < B.col_nums; ++i) {
    for (int j = 0; j < B.row_nums; ++j) {
      B.elements[j * B.col_nums + i] = j * B.col_nums + i + 7;
    }
  }

  DumpMatrix<float>(A);
  DumpMatrix<float>(B);
  // MatrixMul<float>(A, B, &C1);
  // DumpMatrix<float>(C1);
  ComputeMatrixMul<float>(A, B, C2);
  DumpMatrix<float>(C2);
  return 0;
  assert(C1.row_nums == C2.row_nums);
  assert(C1.col_nums == C2.col_nums);
  for (int i = 0; i < C1.row_nums; i ++) {
    for (int j = 0; j < C1.col_nums; ++j) {
      if (fabs(C1.elements[i * C1.col_nums + j] - C2.elements[i * C2.col_nums + j]) > 0.0001) {
        printf("C1!=C2 at[%d][%d], C1=%f, C2=%f\n",
            i, j, C1.elements[i * C1.col_nums + j], C2.elements[i * C2.col_nums + j]);
        exit(1);
      }
    }
  }
  free(A.elements);
  free(B.elements);
  free(C1.elements);
  free(C2.elements);
  return 0;
}
