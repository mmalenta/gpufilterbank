#include "constants.hpp"
#include "kernels.cuh"

#include <cuda.h>
#include <cufft.h>

// NOTE: Not really optimised yet
__global__ void UnpackDadaKernel(int ntimes, uchar4* __restrict__ indata, cufftComplex* __restrict__ outdata) {

    uchar4 tmpread;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ntimes; idx += gridDim.x * blockDim.x) {
        
        // tmpread = indata[idx];
        // outdata[idx].x = (float)(tmpread.x);
        // outdata[idx].y = (float)(tmpread.y);
        // outdata[idx + ntimes].x = (float)(tmpread.z);
        // outdata[idx + ntimes].y = (float)(tmpread.w);

        tmpread = indata[idx];
        outdata[idx].x = static_cast<float>(static_cast<unsigned short>(tmpread.x));
        outdata[idx].y = static_cast<float>(static_cast<unsigned short>(tmpread.y));
        outdata[idx + ntimes].x = static_cast<float>(static_cast<unsigned short>(tmpread.z));
        outdata[idx + ntimes].y = static_cast<float>(static_cast<unsigned short>(tmpread.w));

    }

}

__global__ void DetectDadaKernel(int ntimes, cufftComplex* __restrict__ fftdata, float* __restrict__ powerdata) {

    //int inidx;
    //int outidx;
    int timeoffset;
    int poloffset = ntimes * OUTCHANS;

    float power = 0.0;;
    cufftComplex tmpvalue;
    for (int timeidx = blockIdx.x * TIMEAVG; timeidx < ntimes; timeidx += gridDim.x * TIMEAVG) {

        timeoffset = timeidx * OUTCHANS;

        for (int iavg = 0; iavg < TIMEAVG; ++iavg) {
            tmpvalue = fftdata[timeoffset + iavg * OUTCHANS + threadIdx.x];
            power += tmpvalue.x * tmpvalue.x + tmpvalue.y * tmpvalue.y;
            tmpvalue = fftdata[poloffset + timeoffset + iavg * OUTCHANS + threadIdx.x];
            power += tmpvalue.x * tmpvalue.x + tmpvalue.y * tmpvalue.y;
        }

        powerdata[timeoffset / TIMEAVG + threadIdx.x] = power;
        power = 0.0f;
    }

}

// NOTE: This is a very naive approach, but it works fast enough for now
__global__ void BandpassKernel(int ntimes, float* __restrict__ powerdata, float* __restrict__ bandpass) {

    float sum;

    sum = 0.0f;

    for (int isamp = 0; isamp < ntimes; ++isamp) {
        sum += powerdata[isamp * OUTCHANS + threadIdx.x];
    }

    bandpass[threadIdx.x] += sum;

}