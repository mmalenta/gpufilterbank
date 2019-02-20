#ifndef _H_GPUFIL_KERNELS
#define _H_GPUFIL_KERNELS

#include <cufft.h>

__global__ void UnpackDadaKernel(int ntimes, uchar4* __restrict__ indata, cufftComplex* __restrict__ outdata);

__global__ void DetectDadaKernel(int ntimes, cufftComplex* __restrict__ fftdata, float* __restrict__ powerdata);

__global__ void BandpassKernel(int ntimes, float* __restrict__ powerdata, float* __restrict__ bandpass);

#endif