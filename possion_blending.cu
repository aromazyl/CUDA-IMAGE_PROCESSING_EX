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

__global__ void compute_masks_kernel(const uchar4* d_sourceImg,
    int numRows, int numCols, const int* d_destImg, char* d_mask) {
  int xtid = threadIdx.x;
  int xmid = threadIdx.x + blockIdx.x * blockDim.x;
  int ytid = threadIdx.y;
  int ymid = threadIdx.y + blockIdx.y * blockDim.y;
  if (xmid >= numRows || ymid >= numCols) return;
  uchar4 pixel_val = d_sourceImg[xmid + ymid * numCols];
  int k = 0;
  if (pixel_val.x == 255 && pixel_val.y == 255 && pixel_val.z == 255) { k = 0; }
  else k = 1;
  d_mask[xmid + ymid * numCols] = k;
  __syncthreads();
  unsigned int pixel_up = 255;
  unsigned int pixel_dw = 255;
  unsigned int pixel_lf = 255;
  unsigned int pixel_rt = 255;

  if (ymid > 0) {
    pixel_up = d_sourceImg[xmid + (ymid-1)*numCols];
  }
  if (xmid > 0) {
    pixel_lf = d_sourceImg[xmid + 1 + xmid*numCols];
  }
  if (ymid < numRows - 1) {
    pixel_dw = d_sourceImg[xmid + (ymid+1)*numCols];
  }
  if (xmid < numCols - 1) {
    pixel_rt = d_sourceImg[xmid + 1 + ymid*numCols];
  }

  if (((pixel_up & pixel_dw & pixel_l & pixel_rt) & 0xFFFFFF)
      != 0xFFFFFF) {
    if (d_mask[xmid + ymid * numCols] == 0) {
      d_mask[xmid + ymid * numCols] = 2;
    }
  }
}
/*

__global__ void separate_kernel(
    const uchar4* d_sourceImg,
    unsigned char* r_channel,
    unsigned char* g_channel,
    unsigned char* b_channel, int numCols, int numRows) {

  int xtid = threadIdx.x;
  int xmid = threadIdx.x + blockIdx.x * blockDim.x;
  int ytid = threadIdx.y;
  int ymid = threadIdx.y + blockIdx.y * blockDim.y;

  if (xmid >= numCols || ymid >= numRows) return;
  int index = xmid + ymid * numCols;
  uchar4 rgba = d_sourceImg[index];
  r_channel[index] = rgba.x;
  g_channel[index] = rgba.y;
  b_channel[index] = rgba.z;
}
*/

__global__ void ComputeSum2Kernel(const char* d_mask,
    const uchar4* d_sourceImg, uchar4* t_sum2) {
  int xtid = threadIdx.x;
  int ytid = threadIdx.y;
  int xmid = threadIdx.x + blockIdx.x * blockDim.x;
  int ymid = threadIdx.y + blockIdx.y * blockDim.y;
  if (xmid >= numCols || ymid >= numRows) return;
  uchar4 sum2;
  uchar4 val = d_sourceImg[xmid + ymid * numCols];
  if (ymid > 0) {
    sum2 = val - d_sourceImg[xmid + (ymid - 1) * numCols];
  }
  if (ymid < numRows - 1) {
    sum2 += val - d_sourceImg[xmid + (ymid + 1) * numCols];
  }
  if (xmid > 0) {
    sum2 += val - d_sourceImg[xmid - 1 + ymid * numCols];
  }
  if (xmid < numCols - 1) {
    sum2 += val - d_sourceImg[xmid + 1 + ymid * numCols];
  }
  t_sum2[xmid + ymid * numCols] = sum2;
}

__global__ void JacobiKernel(
    const char* d_mask,
    const uchar4* sum2,
    const uchar4* bufferA, uchar4* bufferB,
    int numCols, int numRows) {
  int xtid = threadIdx.x;
  int ytid = threadIdx.y;
  int xmid = threadIdx.x + blockIdx.x * blockDim.x;
  int ymid = threadIdx.y + blockIdx.y * blockDim.y;
  if (xmid >= numCols || ymid >= numRows) return;
  int4 sum1 = 0;
  if (ymid > 0) {
    uchar4 val = d_mask[xmid + (ymid - 1) * numCols];
    if (val != 0) {
      val = bufferA[xmid + (ymid - 1) * numCols];
      sum1.x += val.x;
      sum1.y += val.y;
      sum1.z += val.z;
    }
  }
  if (ymid < numRows - 1) {
    uchar4 val = d_mask[xmid + (ymid + 1) * numCols];
    if (val != 0) {
      val = bufferA[xmid + (ymid + 1)] * numCols;
      sum1.x += val.x;
      sum1.y += val.y;
      sum1.z += val.z;
    }
  }
  if (xmid > 0) {
    uchar4 val = d_mask[xmid - 1 + ymid * numCols];
    if (val != 0) {
      val = bufferA[xmid - 1 + ymid * numCols];
      sum1.x += val.x;
      sum1.y += val.y;
      sum1.z += val.z;
    }
  }
  if (xmid < numCols - 1) {
    uchar4 val = d_mask[xmid + 1 + ymid * numCols];
    if (val != 0) {
      val = bufferA[xmid + 1 + ymid * numCols];
      sum1.x += val.x;
      sum1.y += val.y;
      sum1.z += val.z;
    }
  }
  float4 newVal;
  newVal.x = (sum1.x + sum2[xmid+ymid*numCols].x) / 4;
  newVal.y = (sum1.y + sum2[xmid+ymid*numCols].y) / 4;
  newVal.z = (sum1.z + sum2[xmid+ymid*numCols].z) / 4;

  bufferB[xmid + ymid * numCols].x = min(255, max(0, int(newVal.x)));
  bufferB[xmid + ymid * numCols].y = min(255, max(0, int(newVal.y)));
  bufferB[xmid + ymid * numCols].z = min(255, max(0, int(newVal.z)));
}

__global__ void InitBuffer(
    const uchar4* const d_sourceImg, uchar4* buffer,
    const char* d_mask, const uchar4* const d_destImg,
    size_t numRowsSource, size_t numColsSource) {
  int xtid = threadIdx.x;
  int xmid = threadIdx.x + blockDim.x * blockIdx.x;
  int ytid = threadIdx.y;
  int ymid = threadIdx.y + blockDim.y * blockIdx.y;
  if (xmid >= numColsSource || ymid >= numRowsSource) return;
  int pos = xmid + ymid * numColsSource;
  if (d_mask[pos] == 1) {
    buffer[pos] = d_sourceImg[pos];
  } else if (d_mask[pos] == 2) {
    buffer[pos] = d_destImg[pos];
  } else {
    buffer[pos] = 0;
  }
}

__global__ void CopyResult(const uchar4* buffer, const char* d_mask,
    uchar4* d_blendedImg, const uchar4* d_destImg,
    const size_t numRowsSource, const size_t numColsSource) {
  int xtid = threadIdx.x;
  int xmid = threadIdx.x + blockDim.x * blockIdx.x;
  int ytid = threadIdx.y;
  int ymid = threadIdx.y + blockDim.y * blockIdx.y;
  if (xmid >= numColsSource || ymid >= numRowsSource) return;
  int pos = xmid + ymid * numColsSource;
  if (d_mask[pos] == 1) {
    d_blendedImg[pos] = buffer[pos];
  } else {
    d_blenedImg[pos] = d_destImg[pos];
  }
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
    uchar4* d_sourceImg;
    uchar4* d_destImg;
    uchar4* d_mask;
    uchar4* bufferA;
    uchar4* bufferB;
    uchar4* sum2;

    unsigned int image_size = numColsSource * numRowsSource;
    checkCudaErrors(cudaMalloc(&d_sourceImg, image_size * sizeof(uchar4)));
    checkCudaErrors(cudaMalloc(&d_destImg, image_size * sizeof(uchar4)));
    checkCudaErrors(cudaMalloc(&d_mask, image_size * sizeof(char)));
    checkCudaErrors(cudaMalloc(&bufferA, image_size * sizeof(uchar4)));
    checkCudaErrors(cudaMalloc(&bufferB, image_size * sizeof(uchar4)));
    checkCudaErrors(cudaMalloc(&sum2, image_size * sizeof(uchar4)));

    dim3 gridDim(numColsSource / 32 + 1, numRowsSource / 32 + 1);
    dim3 blockDim(32, 32);
    checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, image_size * sizeof(uchar4)));
    checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, image_size * sizeof(uchar4)));

    compute_masks_kernel<<<gridDim, blockDim>>>(d_sourceImg, numRowsSource, numColsSource, d_destImg, d_mask);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    ComputeSum2Kernel<<<gridDim, blockDim>>>(d_mask, d_sourceImg, sum2);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    InitBuffer<<<gridDim, blockDim>>>(d_sourceImg, bufferA, d_mask, d_destImg, numRowsSource, numColsSource);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    for (int turn = 0; turn < 800; ++turn) {
      if (turn & 0x01) {
        JacobiKernel<<<gridDim, blockDim>>>(d_mask, sum2, bufferB, bufferA, numColsSource, numRowsSource);
      } else {
        JacobiKernel<<<gridDim, blockDim>>>(d_mask, sum2, bufferA, bufferB, numColsSource, numRowsSource);
      }
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastErrors());
    }

    checkCudaErrors(cudaFree(bufferB));
    checkCudaErrors(cudaFree(sum2));

    CopyResult<<<gridDim, blockDim>>>(bufferA, d_mask, d_blendedImg, d_destImg, numRowsSource, numColsSource);
    cudaDeviceSynchronize(); cudaGetLastError();

    checkCudaErrors(cudaFree(bufferA));
    checkCudaErrors(cudaFree(d_mask));
    checkCudaErrors(cudaFree(d_destImg));

    checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * image_size));
    checkCudaErrors(cudaFree(d_blendedImg));

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
}
