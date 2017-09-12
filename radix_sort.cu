//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <time.h>
#define BLOCK_SIZE 1024

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

Note: ascending order == smallest to largest

Each score is associated with a position, when you sort the scores, you must
also move the positions accordingly.

Implementing Parallel Radix Sort with CUDA
==========================================

The basic idea is to construct a histogram on each pass of how many of each
"digit" there are.   Then we scan this histogram so that we know where to put
the output of each digit.  For example, the first 1 must come after all the
0s so we have to know how many 0s there are to be able to start moving 1s
into the correct position.

1) Histogram of the number of occurrences of each digit
2) Exclusive Prefix Sum of Histogram
3) Determine relative offset of each digit
For example [0 0 1 1 0 0 1]
->  [0 1 0 1 2 3 2]
4) Combine the results of steps 2 & 3 to determine the final
output location for each element and move it there

LSB Radix sort is an out-of-place sort and you will need to ping-pong values
between the input and output buffers we have provided.  Make sure the final
sorted results end up in the output buffer!  Hint: You may need to do a copy
at the end.

 */

__global__ void HistogramKernel(unsigned int * input, unsigned int size, unsigned int* histogram, unsigned int pass) {
  int mid = threadIdx.x + blockIdx.x * blockDim.x;
  if (mid < size) {
    atomicAdd(&histogram[(input[mid]>>pass) & 0x01], 1);
  }
}

__global__ void scan_sum_kernel(unsigned int* input_vals, unsigned int pass, unsigned int * output, unsigned int* output_block, unsigned int size, unsigned int block_num) {
  unsigned int tid = threadIdx.x;
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ unsigned int shared_input_vals[BLOCK_SIZE];
  __shared__ unsigned int shared_output[BLOCK_SIZE];
  if (mid >= size) {
    shared_input_vals[tid] = 0xFFFFFFFF;
  } else {
    shared_input_vals[tid] = input_vals[mid];
  }

  __syncthreads();

  if (tid == 0 || ((shared_input_vals[tid - 1] >> pass) & 0x01)) {
    shared_output[tid] = 0;
  } else {
    shared_output[tid] = 1;
  }

  __syncthreads();

  for (unsigned int i = 1; i < BLOCK_SIZE; i <<= 1) {
    unsigned int val = 0;
    if (tid >= i) {
      val = shared_output[tid - i];
    }
    __syncthreads();
    shared_output[tid] += val;
    __syncthreads();
  }

  if (mid < size) {
    output[mid] = shared_output[tid];
    if ((mid == size - 1) || (tid == BLOCK_SIZE-1)) {
      output_block[blockIdx.x] = shared_output[tid];
      if (!((shared_input_vals[tid] >> pass) & 0x01)) {
        //output_block[mid/BLOCK_SIZE] += 1;
        output_block[blockIdx.x] += 1;
      }
    }
  }
  __syncthreads();
}

__global__ void scan_kernel(unsigned int* output_block, unsigned int block_num) {
  __shared__ unsigned int shared_output[BLOCK_SIZE];

  if (threadIdx.x >= block_num || threadIdx.x == 0) {
    shared_output[threadIdx.x] = 0x0;
  }  else {
    shared_output[threadIdx.x] = output_block[threadIdx.x - 1];
  }
  __syncthreads();

  for (unsigned int i = 1; i < block_num; i <<= 1) {
    unsigned int val = 0;
    if (threadIdx.x >= i) {
      val = shared_output[threadIdx.x - i];
    }
    __syncthreads();
    shared_output[threadIdx.x] += val;
    __syncthreads();
  }

  if (threadIdx.x < block_num) {
    output_block[threadIdx.x] = shared_output[threadIdx.x];
  }
  __syncthreads();
}

void show_data(unsigned int* d_data, unsigned int len, char* tag) {
  unsigned int* h_data = (unsigned int*) malloc(len * sizeof(unsigned int));
  cudaMemcpy(h_data, d_data, sizeof(unsigned int) * len, cudaMemcpyDeviceToHost);
  unsigned int last = h_data[0];
  for (unsigned int i = 0; i < len; ++i) {
    printf("%s[%u]=%u; ", tag, i, h_data[i]);
    last = h_data[i];
  }
  free(h_data);
}

__global__ void scan_large_sum_kernel(unsigned int* output_block,
    unsigned int* output_val,
    unsigned int* output_pos,
    unsigned int* input_val,
    unsigned int* input_pos,
    unsigned int* histogram,
    unsigned int pass,
    unsigned int block_num,
    unsigned int size) {

  __shared__ unsigned int shared_prefix_sum[BLOCK_SIZE];
  unsigned int tid = threadIdx.x;
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;

  if (mid >= size) {
    shared_prefix_sum[tid] = 1;
  } else {
    shared_prefix_sum[tid] = output_block[blockIdx.x] + output_val[mid];
  }
  //if (shared_prefix_sum[tid] >= size) printf("mid/BLOCK_SIZE=%d\n", mid/BLOCK_SIZE);
  __syncthreads();


  if (mid < size) {
    unsigned int location = shared_prefix_sum[tid];
    if ((input_val[mid] >> pass) & 0x01) {
      location = mid + histogram[0] - shared_prefix_sum[tid];
    }
    if (location >= size) printf("pass=%d,input[mid]=%d,mid=%d, blockIdx.x=%d, histogram[0]=%d, shared_prefix_sum[tid]=%d\n", 
        pass, input_val[mid], mid, blockIdx.x, histogram[0], shared_prefix_sum[tid]);
    output_val[mid] = location;
  }
  __syncthreads();
}

__global__ void scatter_kernel(unsigned int* d_inputVals,
    unsigned int* d_inputPos,
    unsigned int* d_outputVals,
    unsigned int* d_outputPos,
    unsigned int* cu_outputVals,
    size_t numElems) {
  //unsigned int tid = threadIdx.x;
  unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int val;
  if (mid < numElems) {
    val = cu_outputVals[mid];
  }

  if (mid < numElems) {
    d_outputVals[val] = d_inputVals[mid];
    d_outputPos[val] = d_inputPos[mid];
  }
  __syncthreads();
}
void your_sort(unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* const d_outputVals,
    unsigned int* const d_outputPos,
    const size_t numElems)
{
  //printf("numElems:%d\n", numElems);

  unsigned int block_nums;
  if (numElems % BLOCK_SIZE == 0) {
    block_nums = numElems / BLOCK_SIZE;
  } else {
    block_nums = numElems / BLOCK_SIZE + 1;
  }
  dim3 scan_sum_blockDim(BLOCK_SIZE);
  dim3 scan_sum_gridDim(block_nums);
  dim3 scan_gridDim(1);
  dim3 scan_blockDim(BLOCK_SIZE);
  dim3 scan_large_sum_blockDim(BLOCK_SIZE);
  dim3 scan_large_sum_gridDim(block_nums);
  unsigned int* cu_inputVals;
  unsigned int* cu_inputPos;
  unsigned int* cu_outputVals;
  unsigned int* cu_outputPos;
  unsigned int* cu_block;
  cu_inputVals = d_inputVals;
  cu_inputPos = d_inputPos;
  cu_outputVals = d_outputVals;
  cu_outputPos = d_outputPos;
  unsigned int* d_histogram;
  checkCudaErrors(cudaMalloc(&d_histogram, 2 * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&cu_inputVals, numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&cu_inputPos, numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(cu_inputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(cu_inputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMalloc(&cu_outputVals, numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&cu_outputPos, numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&cu_block, block_nums * sizeof(unsigned int)));

  for (unsigned int pass = 0; pass < 32; ++pass) {
    checkCudaErrors(cudaMemset(d_outputVals, 0, block_nums * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(cu_outputVals, 0, block_nums * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(cu_outputPos, 0, block_nums * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_outputPos, 0, block_nums * sizeof(unsigned int)));
    // checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemset(d_histogram, 0, 2 * sizeof(unsigned int)));
    checkCudaErrors(cudaDeviceSynchronize());
    HistogramKernel<<<scan_sum_gridDim, scan_sum_blockDim>>>(d_inputVals, numElems, d_histogram, pass);
    checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemset(cu_block, 0, block_nums * sizeof(unsigned int)));
    checkCudaErrors(cudaDeviceSynchronize());
    scan_sum_kernel<<<scan_sum_gridDim, scan_sum_blockDim>>>(d_inputVals, pass, cu_outputVals, cu_block, numElems, block_nums);
    checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());


    scan_kernel<<<1, BLOCK_SIZE>>>(cu_block, block_nums);
    checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());
    scan_large_sum_kernel<<<scan_large_sum_gridDim, scan_large_sum_blockDim>>>(
        cu_block, cu_outputVals, 
        cu_outputPos, d_inputVals, 
        d_inputPos, d_histogram, 
        pass, block_nums, numElems);

    checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());

    scatter_kernel<<<scan_sum_gridDim, scan_sum_blockDim>>>(
        d_inputVals, d_inputPos,
        d_outputVals, d_outputPos,
        cu_outputVals, numElems);



    checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(cu_inputVals, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(cu_inputPos, d_outputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

  }

  checkCudaErrors(cudaFree(cu_block));
  checkCudaErrors(cudaFree(cu_inputVals));
  checkCudaErrors(cudaFree(cu_inputPos));
  checkCudaErrors(cudaFree(cu_outputVals));
  checkCudaErrors(cudaFree(cu_outputPos));

}

int main() {
  static const unsigned int numElems = 20000;
  unsigned int h_inputVal[numElems];
  unsigned int h_inputPos[numElems];
  srand(time(NULL));
  unsigned int* d_inputVal;
  unsigned int* d_inputPos;
  unsigned int* d_outputVal;
  unsigned int* d_outputPos;

  cudaMalloc(&d_inputVal, numElems * sizeof(unsigned int));
  cudaMalloc(&d_inputPos, numElems * sizeof(unsigned int));
  cudaMalloc(&d_outputPos, numElems * sizeof(unsigned int));
  cudaMalloc(&d_outputVal, numElems * sizeof(unsigned int));
  for (int i = 0; i < numElems; ++i) {
    h_inputVal[i] = h_inputPos[i] = rand();
  }

  cudaMemcpy(d_inputVal, h_inputPos, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_inputPos, h_inputVal, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice);

  your_sort(d_inputVal, d_inputPos, d_outputPos, d_outputVal, numElems);
  show_data(d_inputVal, numElems, "d_inputVals");
  printf("**********");
  show_data(d_outputVal, numElems, "d_outputVals");
}
