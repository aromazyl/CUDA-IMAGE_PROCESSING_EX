//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
   as boundary conditions for solving a Poisson equation that tells
   us how to blend the images.

   No pixels from the destination except pixels on the border
   are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
else if the neighbor in on the border then += DestinationImg[neighbor]

Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

2) Calculate the new pixel value:
float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


In this assignment we will do 800 iterations.
 */



#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"
__global__ void compute_masks_kernel(const uchar4* d_sourceImg,
    int numRows, int numCols, char* d_mask) {
  int xmid = threadIdx.x + blockIdx.x * blockDim.x;
  int ymid = threadIdx.y + blockIdx.y * blockDim.y;
  if (xmid >= numCols || ymid >= numRows) return;
  uchar4 pixel_val = d_sourceImg[xmid + ymid * numCols];
  int k = 0;
  if (pixel_val.x == 255 && pixel_val.y == 255 && pixel_val.z == 255) { k = 0; }
  else k = 1;
  d_mask[xmid + ymid * numCols] = k;
  __syncthreads();
  uchar4 pixel_up = make_uchar4(255, 255, 255, 255);
  uchar4 pixel_dw = make_uchar4(255, 255, 255, 255);
  uchar4 pixel_lf = make_uchar4(255, 255, 255, 255);
  uchar4 pixel_rt = make_uchar4(255, 255, 255, 255);

  if (ymid > 0) {
    pixel_up = d_sourceImg[xmid + (ymid-1)*numCols];
  }
  if (xmid > 0) {
    pixel_lf = d_sourceImg[xmid - 1 + ymid*numCols];
  }
  if (ymid < numRows - 1) {
    pixel_dw = d_sourceImg[xmid + (ymid+1)*numCols];
  }
  if (xmid < numCols - 1) {
    pixel_rt = d_sourceImg[xmid + 1 + ymid*numCols];
  }

  if ((((pixel_up.x & pixel_dw.x & pixel_lf.x & pixel_rt.x)
          != 0xFF)) || ((pixel_up.y & pixel_dw.y & pixel_lf.y & pixel_rt.y)
          != 0xFF) || ((pixel_up.z & pixel_dw.z & pixel_lf.z & pixel_rt.z)
            != 0xFF)) {
    if (d_mask[xmid + ymid * numCols] == 0) {
      d_mask[xmid + ymid * numCols] = 2;
    }
  }
}

__global__ void ComputeSum2Kernel(
    const uchar4* d_sourceImg, float4* t_sum2,
    size_t numCols, size_t numRows) {
  int xmid = threadIdx.x + blockIdx.x * blockDim.x;
  int ymid = threadIdx.y + blockIdx.y * blockDim.y;
  if (xmid >= numCols || ymid >= numRows) return;
  float4 sum2;
  uchar4 val = d_sourceImg[xmid + ymid * numCols];
  if (ymid > 0) {
    sum2.x = val.x - d_sourceImg[xmid + (ymid - 1) * numCols].x;
    sum2.y = val.y - d_sourceImg[xmid + (ymid - 1) * numCols].y;
    sum2.z = val.z - d_sourceImg[xmid + (ymid - 1) * numCols].z;
  }
  if (ymid < numRows - 1) {
    sum2.x += val.x - d_sourceImg[xmid + (ymid + 1) * numCols].x;
    sum2.y += val.y - d_sourceImg[xmid + (ymid + 1) * numCols].y;
    sum2.z += val.z - d_sourceImg[xmid + (ymid + 1) * numCols].z;
  }
  if (xmid > 0) {
    sum2.x += val.x - d_sourceImg[xmid - 1 + ymid * numCols].x;
    sum2.y += val.y - d_sourceImg[xmid - 1 + ymid * numCols].y;
    sum2.z += val.z - d_sourceImg[xmid - 1 + ymid * numCols].z;
  }
  if (xmid < numCols - 1) {
    sum2.x += val.x - d_sourceImg[xmid + 1 + ymid * numCols].x;
    sum2.y += val.y - d_sourceImg[xmid + 1 + ymid * numCols].y;
    sum2.z += val.z - d_sourceImg[xmid + 1 + ymid * numCols].z;
  }
  t_sum2[xmid + ymid * numCols] = sum2;
}

__global__ void JacobiKernel(
    const char* d_mask,
    const float4* sum2,
    const float4* bufferA, float4* bufferB,
    int numCols, int numRows) {
  int xmid = threadIdx.x + blockIdx.x * blockDim.x;
  int ymid = threadIdx.y + blockIdx.y * blockDim.y;
  if (xmid >= numCols || ymid >= numRows) return;
  float4 sum1 = make_float4(0.0f,0.0f,0.0f,0.0f);
  float4 val;
  if (d_mask[xmid + ymid * numCols] == 1)  {

    if (ymid > 0.0f) {
      val = bufferA[xmid + (ymid - 1) * numCols];
      sum1.x += val.x;
      sum1.y += val.y;
      sum1.z += val.z;
    }
    if (ymid < numRows - 1) {
      val = bufferA[xmid + (ymid + 1) * numCols];
      sum1.x += val.x;
      sum1.y += val.y;
      sum1.z += val.z;
    }
    if (xmid > 0.0f) {
      val = bufferA[xmid - 1 + ymid * numCols];
      sum1.x += val.x;
      sum1.y += val.y;
      sum1.z += val.z;
    }
    if (xmid < numCols - 1) {
      val = bufferA[xmid + 1 + ymid * numCols];
      sum1.x += val.x;
      sum1.y += val.y;
      sum1.z += val.z;
    }
    float4 newVal;
    newVal.x = (sum1.x + sum2[xmid+ymid*numCols].x) / 4;
    newVal.y = (sum1.y + sum2[xmid+ymid*numCols].y) / 4;
    newVal.z = (sum1.z + sum2[xmid+ymid*numCols].z) / 4;

    bufferB[xmid + ymid * numCols].x = min(255.0f, max(0.0f, newVal.x));
    bufferB[xmid + ymid * numCols].y = min(255.0f, max(0.0f, newVal.y));
    bufferB[xmid + ymid * numCols].z = min(255.0f, max(0.0f, newVal.z));
  } else {
    bufferB[xmid + ymid * numCols] = bufferA[xmid + ymid * numCols];
  }
}

__global__ void InitBuffer(
    const uchar4* const d_sourceImg, float4* buffer,
    const char* d_mask, const uchar4* const d_destImg,
    size_t numRowsSource, size_t numColsSource) {
  int xmid = threadIdx.x + blockDim.x * blockIdx.x;
  int ymid = threadIdx.y + blockDim.y * blockIdx.y;
  if (xmid >= numColsSource || ymid >= numRowsSource) return;
  int pos = xmid + ymid * numColsSource;
  if (d_mask[pos] == 1) {
    buffer[pos].x = d_sourceImg[pos].x;
    buffer[pos].y = d_sourceImg[pos].y;
    buffer[pos].z = d_sourceImg[pos].z;
  } else if (d_mask[pos] == 2) {
    buffer[pos].x = d_destImg[pos].x;
    buffer[pos].y = d_destImg[pos].y;
    buffer[pos].z = d_destImg[pos].z;
  } else {
    buffer[pos] = make_float4(0.0f,0.0f,0.0f,0.0f);
  }
}

__global__ void CopyResult(const float4* buffer, const char* d_mask,
    uchar4* d_blendedImg, const uchar4* d_destImg,
    const size_t numRowsSource, const size_t numColsSource) {
  int xmid = threadIdx.x + blockDim.x * blockIdx.x;
  int ymid = threadIdx.y + blockDim.y * blockIdx.y;
  if (xmid >= numColsSource || ymid >= numRowsSource) return;
  int pos = xmid + ymid * numColsSource;
  if (d_mask[pos] == 1) {
    d_blendedImg[pos].x = buffer[pos].x;
    d_blendedImg[pos].y = buffer[pos].y;
    d_blendedImg[pos].z = buffer[pos].z;
    d_blendedImg[pos].w = d_destImg[pos].w;
  } else {
    d_blendedImg[pos] = d_destImg[pos];
  }
}
void your_blend(const uchar4* const h_sourceImg,  //IN
    const size_t numRowsSource, const size_t numColsSource,
    const uchar4* const h_destImg, //IN
    uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement

     1) Compute a mask of the pixels from the source image to be copied
     The pixels that shouldn't be copied are completely white, they
     have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
     pixel has all 4 neighbors also inside the mask.  A border pixel is
     in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
     act as our guesses.  Initialize them to the respective color
     channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
     above 800 times.

     6) Create the output image by replacing all the interior pixels
     in the destination image with the result of the Jacobi iterations.
     Just cast the floating point values to unsigned chars since we have
     already made sure to clamp them to the correct range.

     Since this is final assignment we provide little boilerplate code to
     help you.  Notice that all the input/output pointers are HOST pointers.

     You will have to allocate all of your own GPU memory and perform your own
     memcopies to get data in and out of the GPU memory.

     Remember to wrap all of your calls with checkCudaErrors() to catch any
     thing that might go wrong.  After each kernel call do:

     cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

     to catch any errors that happened while executing the kernel.
   */



  /* The reference calculation is provided below, feel free to use it
     for debugging purposes. 
   */
  uchar4* d_sourceImg;
  uchar4* d_destImg;
  char* d_mask;
  float4* bufferA;
  float4* bufferB;
  float4* sum2;
  uchar4* d_blendedImg;

  unsigned int image_size = numColsSource * numRowsSource;
  checkCudaErrors(cudaMalloc(&d_blendedImg, image_size * sizeof(uchar4)));
  checkCudaErrors(cudaMalloc(&d_sourceImg, image_size * sizeof(uchar4)));
  checkCudaErrors(cudaMalloc(&d_destImg, image_size * sizeof(uchar4)));
  checkCudaErrors(cudaMalloc(&d_mask, image_size * sizeof(char)));
  checkCudaErrors(cudaMalloc(&bufferA, image_size * sizeof(float4)));
  checkCudaErrors(cudaMalloc(&bufferB, image_size * sizeof(float4)));
  checkCudaErrors(cudaMalloc(&sum2, image_size * sizeof(float4)));

  dim3 gridDim(numColsSource / 32 + 1, numRowsSource / 32 + 1);
  dim3 blockDim(32, 32);
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, image_size * sizeof(uchar4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, image_size * sizeof(uchar4), cudaMemcpyHostToDevice));

  compute_masks_kernel<<<gridDim, blockDim>>>(d_sourceImg, numRowsSource, numColsSource, d_mask);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  ComputeSum2Kernel<<<gridDim, blockDim>>>(d_sourceImg, sum2, numColsSource, numRowsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  InitBuffer<<<gridDim, blockDim>>>(d_sourceImg, bufferA, d_mask, d_destImg, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  for (int turn = 0; turn < 800; ++turn) {
    if (turn & 0x01) {
      JacobiKernel<<<gridDim, blockDim>>>(d_mask, sum2, bufferB, bufferA, numColsSource, numRowsSource);
    } else {
      JacobiKernel<<<gridDim, blockDim>>>(d_mask, sum2, bufferA, bufferB, numColsSource, numRowsSource);
    }
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }

  checkCudaErrors(cudaFree(bufferB));
  checkCudaErrors(cudaFree(sum2));

  CopyResult<<<gridDim, blockDim>>>(bufferA, d_mask, d_blendedImg, d_destImg, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); cudaGetLastError();

  checkCudaErrors(cudaFree(bufferA));
  checkCudaErrors(cudaFree(d_mask));
  checkCudaErrors(cudaFree(d_destImg));

  checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * image_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_blendedImg));
  /*
     uchar4* h_reference = new uchar4[srcSize];
     reference_calc(h_sourceImg, numRowsSource, numColsSource,
     h_destImg, h_reference);

     checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
     delete[] h_reference; */
}

