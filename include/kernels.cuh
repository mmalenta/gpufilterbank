#ifndef _H_GPUFIL_KERNELS
#define _H_GPUFIL_KERNELS

#include <cufft.h>

__global__ void UnpackDadaKernel(int ntimes, uchar4* __restrict__ indata, cufftComplex* __restrict__ outdata);

__global__ void DetectDadaKernel(int ntimes, cufftComplex* __restrict__ fftdata, float* __restrict__ powerdata, int nbands);

__global__ void BandpassKernel(int ntimes, float* __restrict__ powerdata, float* __restrict__ bandpass);

__global__ void MaskKernel(float* __restrict__ powerdata, int* __restrict__ mask, int ntimes, int nchans);

#endif