/*
 * state.h
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef STATE_H
#define STATE_H

#define MAX_DEVICES 256

typedef struct Generator {
  struct curandStateMtgp32* gen_states;
  struct mtgp32_kernel_params* kernel_params;
  int initf;
  unsigned long long inital_seed;
} Generator;

typedef struct RNGState {
  Generator* gen;
  int num_devices;
} RNGState;

typedef struct Stream {
  cudaStream_t stream;
  int device;
  int refcount;
} Stream;

// flag cudaStreamDefault cudaStreamNonBlocking
inline Stream* Stream_new(unsigned int flags) {
  Stream* self = (Stream*)malloc(sizeof(Stream));
  self->refcount = 1;
  checkCudaErrors(cudaGetDevice(&self->device));
  checkCudaErrors(cudaStreamCreateWithFlags(&self->stream, flags));
  return self;
}



typedef struct CudaResourcePerDevice {
  Stream** streams;
  size_t scratchSpacePerStream;
  void** devScratchSpacePerStream;
} CudaResourcePerDevice;

typedef struct State {
  struct RNGState* rngState;
  struct cudaDeviceProp* deviceProperties;
  CudaResourcePerDevice* resourcesPerDevice;
  int numDevices;
  int numUserStreams;
  DeviceAllocator* cudaDeviceAllocator;
  void (*cuGCFunction)(void* data);
  void* cuGCData;
  ptrdiff_t heapSoftmax;
  ptrdiff_t heapDelta;
} State;

void CudaInit(State* state);

#endif /* !STATE_H */
