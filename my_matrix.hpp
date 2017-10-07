/*
 * my_matrix.cpp
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#pragma once
#include "matrix.h"
#include <cstdio>
#include <cassert>

template <typename T>
void MatrixMul(
    const Matrix<T>& A,
    const Matrix<T>& B,
    Matrix<T>* matC) {
  assert(A.col_nums == B.row_nums);
  matC->col_nums = B.col_nums;
  matC->row_nums = A.row_nums;
  for (int i = 0; i < A.row_nums; ++i) {
    for (int j = 0; j < B.col_nums; ++j) {
      matC->elements[i * matC->col_nums + j] = T(0);
      for (int p = 0; p < A.col_nums; ++p) {
        matC->elements[i * matC->col_nums + j] += A.elements[i * A.col_nums + p] * B.elements[p * B.col_nums + j];
      }
    }
  }
}
