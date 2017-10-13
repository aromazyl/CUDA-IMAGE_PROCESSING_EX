/*
 * storage.c
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "storage.h"


Storage* Storage_set(State* state, Storage* self, ptrdiff_t index, float value) {
  int device;
  checkCudaErrors(cudaGetDevice(&device));
}
