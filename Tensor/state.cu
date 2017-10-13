#include <pthread.h>
#include "state.h"
#include "../utils.h"

static Stream default_streams[MAX_DEVICES];

static void initialize_default_streams() {
  for (int i = 0; i < MAX_DEVICES; ++i) {
    default_streams[i].device = i;
  }
}

Stream* Stream_defaultStream(int device) {
  std::once_flag once;
  std::call_once(once, &initialize_default_streams);
  return &default_stream;
}

cudaError_t CudaFree(State* state, void* ptr) {
  DeviceAllocator* allocator = state->cudaDeviceAllocator;
  return allocator->free(allocator->sate, ptr);
}

Stream* Stream_createWithPriority(int flags, int priority) {
  Stream* self = (Stream*) malloc(sizeof(Stream));
  self->refcount = 1;
  checkCudaErrors(cudaGetDevice(&self->device));
  checkCudaErrors(cudaStreamCreateWithPriority(&self->stream, flags, priority));
  return self;
}

void Stream_free(Stream* self) {
  if (!self || !self->stream) return;
  if (__sync_fetch_and_add(&self->refcount, -1) == 1) {
    checkCudaErrors(cudaStreamDestory(self->stream));
    free(self);
  }
}

void Stream_retain(Stream* self) {
  if (self->stream) {
    __sync_fetch_and_add(&self->refcount);
  }
}
static cudaError_t cudaMallocWrapper(void* ctx, void** devPtr, size_t size, cudaStream_t stream) {
  return cudaMalloc(devPtr, size);
}

static cudaError_t cudaFreeWrapper(void* ctx, void* devPtr) {
  return cudaFree(devPtr);
}

static DeviceAllocator defaultDeviceAllocator = {
  &cudaMallocWrapper,
  NULL,
  &cudaFreeWrapper,
  NULL,
  NULL,
  NULL
};

void CudaInit(State* state) {
  if (!state->cudaDeviceAllocator) {
    state->cudaDeviceAllocator = &defaultDeviceAllocator;
  }
  int numDevices;
  checkCudaErrors(cudaGetDeviceCount(&numDevices));
  state->numDevices = numDevices;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  state->currentStreams = (pthread_key_t*)malloc(numDevices * sizeof(pthread_key_t));
  state->resourcePerDevice = (CudaResourcePerDevice*)alloc(numDevices, sizeof(CudaResourcePerDevice));
  state->deviceProperties =
    (struct cudaDeviceProp*)malloc(numDevices * sizeof(struct cudaDeviceProp));
  state->rngState = (RNGState*)malloc(sizeof(RNGState));
  state->rngState->num_devices = numDevices;
  for (int i = 0; i < numDevices; ++i) {
    CudaResourcePerDevice * res = state->resourcePerDevice[i];
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaGetDeviceProperties(&state->deviceProperties[i], i));
    res->streams = (Stream**)malloc(sizeof(Stream*));
    res->streams[0] = Stream_defaultStream(i);
    int numSM = state->deviceProperties[i].multiProcessorCount;
#define MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE (23768 * sizeof(float))
#define MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM (4*sizeof(float))
    size_t sizePerStream =
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE >= numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM ?
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE :
      numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM;
#undef MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE
#undef MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM
    res->scratchSpacePerStream = sizePerStream;
  }
  checkCudaErrors(cudaSetDevice(device));
  state->heapSoftmax = 3e8;
  state->heapDelta = 0;
}

void CudaShutDown(State* state) {
  if (state->rngState == NULL) return;
  for (int i = 0; i < state->rngState->numDevices; ++i) {
    if (state->rngState) {
      state->cudaDeviceAllocator->free(state, state->rngState->gen[i]);
      free(state->rngState);
    }
  }
  state->rngState = NULL;
  free(state->deviceProperties);
  int deviceCount = 0;
  int prevDev = -1;
  checkCudaErrors(cudaGetDevice(&prevDev));
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
  for (int dev = 0; dev < deviceCount; ++dev) {
    checkCudaErrors(cudaSetDevice(dev));
    CudaResourcePerDevice* res = &(state->resourcePerDevice[dev]);
    for (int i = 0; i <= state->numUserStreams; ++i)
      Stream_free(res->stream[i]);
    if (res->devScratchSpacePerStream) {
      for (int stream = 0; stream <= state->numUserStreams; ++stream) {
        checkCudaErrors(CudaFree(state, res->devScratchSpacePerStream[stream]));
      }
    }
  }
  free(res->streams);
  free(res->devScratchSpacePerStream);
  Stream_free(pthread_get_specific());

}
