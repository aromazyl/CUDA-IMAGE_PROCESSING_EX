#pragma once


#include <stdlib.h>

typedef struct mAllocator {
  cudaError_t (*malloc)(void*, void**, size_t, cudaStream_t);
  cudaError_t (*realloc)(void*, void**, size_t, size_t, cudaStream_t);
  cudaError_t (*free)(void*, void*);
  cudaError_t (*emptyCache)(void*);
  cudaError_t (*cacheInfo)(void*, int, size_t*, size_t*);
  void* state;
} DeviceAllocator;

typedef struct State {
  struct cudaDeviceProp* deviceProperties;
} State;
