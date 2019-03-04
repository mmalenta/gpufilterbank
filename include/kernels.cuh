#ifndef _H_GPUFIL_KERNELS
#define _H_GPUFIL_KERNELS

#include <cufft.h>

__global__ void UnpackDadaKernel(int ntimes, uchar4* __restrict__ indata, cufftComplex* __restrict__ outdata);

__global__ void DetectDadaKernel(int ntimes, cufftComplex* __restrict__ fftdata, float* __restrict__ powerdata, int nbands);

__global__ void BandpassKernel(int ntimes, float* __restrict__ powerdata, float* __restrict__ bandpass);

__global__ void MaskKernel(float* __restrict__ powerdata, int* __restrict__ mask, int ntimes, int nchans);

__global__ void AdjustKernel(float* __restrict__ powerdata, unsigned char* __restrict__ scaledpower, float* __restrict__ diffs, int nbands, int ntimes, float datamin, float scale, int runscale);

#endif