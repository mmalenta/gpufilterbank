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
// NOTE: For this kernel, ntime is the number of non-averaged time samples for a single band
__global__ void DetectDadaKernel(int ntimes, cufftComplex* __restrict__ fftdata, float* __restrict__ powerdata, int nbands) {

    int timeoffset;
    int bandoffset;
    int poloffset = ntimes * OUTCHANS * nbands;

    float power = 0.0;;
    cufftComplex tmpvalue;
    //NOTE: We need to make sure we can process more than one band
    for (int iband = 0; iband < nbands; ++iband) {
        bandoffset = iband * ntimes * OUTCHANS;
        // NOTE: Time samples and channels within the bands are continguous - same as in the single band case, so no problem here
        for (int timeidx = blockIdx.x * TIMEAVG; timeidx < ntimes; timeidx += gridDim.x * TIMEAVG) {

            timeoffset = timeidx * OUTCHANS;

            for (int iavg = 0; iavg < TIMEAVG; ++iavg) {
                tmpvalue = fftdata[bandoffset + timeoffset + iavg * OUTCHANS + threadIdx.x];
                power += tmpvalue.x * tmpvalue.x + tmpvalue.y * tmpvalue.y;
                tmpvalue = fftdata[poloffset + bandoffset + timeoffset + iavg * OUTCHANS + threadIdx.x];
                power += tmpvalue.x * tmpvalue.x + tmpvalue.y * tmpvalue.y;
            }

            // NOTE: Need to swap the negative frequencies
            int outthreadidx = (threadIdx.x + 512) % OUTCHANS;

            powerdata[timeidx / TIMEAVG * OUTCHANS * nbands + iband * OUTCHANS + outthreadidx] = power;
            power = 0.0f;
        }
    }
}

// NOTE: This is a very naive approach, but it works fast enough for now
__global__ void BandpassKernel(int ntimes, float* __restrict__ powerdata, float* __restrict__ bandpass) {

    float sum;

    int chanidx = blockIdx.x * OUTCHANS + threadIdx.x;
    int fullchans = gridDim.x * OUTCHANS;


    sum = 0.0f;

    for (int isamp = 0; isamp < ntimes; ++isamp) {
        sum += powerdata[isamp * fullchans + chanidx];
    }

    bandpass[chanidx] += sum;

}

__global__ void AdjustKernel(float *powerdata, float *diffs, int nbands, int ntimes, float datamin, float scale, int runscale) {

    float diff = 0.0f;
    float tmp = 0.0f;
    size_t idx = 0;
    for (int timeidx = blockIdx.x; timeidx < ntimes; timeidx += gridDim.x) {
        for (int iband = 0; iband < nbands; ++iband) {
            diff = diffs[iband];
            idx = timeidx * OUTCHANS * nbands + iband * OUTCHANS + threadIdx.x;
            tmp = powerdata[idx];
            if (runscale) {
                tmp = (tmp + diff - datamin) * scale;
                // NOTE: Divergent warp because why not!
                if (tmp < 0.0f) {
                    tmp = 0.0f;
                } else if (tmp > 255.0f) {
                    tmp = 255.0f;
                }
                powerdata[idx] = tmp;
            } else {
                powerdata[idx] = tmp + diff;
            }
        }
    }

}

 /*
struct FactorFunctor {
    __host__ __device__ float operator()(float val) {
        return val != 0 ? 1.0f/val : val;
    }
};

__constant__ unsigned char kMask[] = {0x03, 0x0C, 0x30, 0xC0};

__global__ void UnpackKernel(unsigned char **in, float **out, size_t samples) {

    // NOTE: Each thread in the block processes 625 incoming bytes
    int idx = blockIdx.x * blockDim.x * PERBLOCK + threadIdx.x;
    int tmod = threadIdx.x % 4;

    // NOTE: Each thread can store one value
    __shared__ unsigned char incoming[1024];

    int outidx = blockIdx.x * blockDim.x * PERBLOCK * 4;

    for (int isamp = 0; isamp < PERBLOCK; ++isamp) {
        if (idx < samples) {
            for (int ipol = 0; ipol < 2; ++ipol) {
                incoming[threadIdx.x] = in[ipol][idx];
                __syncthreads();
                int outidx2 = outidx + threadIdx.x;
		for (int ichunk = 0; ichunk < 4; ++ichunk) {
                    int inidx = threadIdx.x / 4 + ichunk * 256;
                    unsigned char inval = incoming[inidx];
                    out[ipol][outidx2] = static_cast<float>(static_cast<short>(((inval & kMask[tmod]) >> (2 * tmod))));
                    outidx2 += 1024;
                }
            }
        }
        idx += blockDim.x;
        outidx += blockDim.x * 4;
    }
}

// NOTE: Does not do any frequency averaging
// NOTE: Outputs only the total intensity and no other Stokes parameters
// NOTE: PERBLOCK is the number of output samples per block
__global__ void DetectKernel(cufftComplex** __restrict__ in, float* __restrict__ out) {
    int outidx = blockIdx.x * PERBLOCK * FFTUSE + FFTUSE - threadIdx.x - 1;
    int inidx = blockIdx.x * PERBLOCK * TIMEAVG * FFTOUT + threadIdx.x + 1;

    float outvalue = 0.0f;
    cufftComplex polval;

    for (int isamp = 0; isamp < PERBLOCK; ++isamp) {

        // NOTE: Read the data from the incoming array
        for (int ipol = 0; ipol < 2; ++ipol) {
            for (int iavg = 0; iavg < TIMEAVG; ++iavg) {
                polval = in[ipol][inidx + iavg * FFTOUT];
                outvalue += polval.x * polval.x + polval.y * polval.y;
            }

        }
        outvalue *= TIMESCALE;
        out[outidx] = outvalue;
        inidx += FFTOUT * TIMEAVG;
        outidx += FFTUSE;
        outvalue = 0.0;
    }
}

__global__ void DetectScaleKernel(cufftComplex** __restrict__ in, unsigned char* __restrict__ out, float* __restrict__ means, float* __restrict__ stdevs) {
    int outidx = blockIdx.x * PERBLOCK * FFTUSE + FFTUSE - threadIdx.x - 1;
    int inidx = blockIdx.x * PERBLOCK * TIMEAVG * FFTOUT + threadIdx.x + 1;

    float outvalue = 0.0f;
    cufftComplex polval;

    int scaled = 0;

    for (int isamp = 0; isamp < PERBLOCK; ++isamp) {

        // NOTE: Read the data from the incoming array
        for (int ipol = 0; ipol < 2; ++ipol) {
            for (int iavg = 0; iavg < TIMEAVG; ++iavg) {
                polval = in[ipol][inidx + iavg * FFTOUT];
                outvalue += polval.x * polval.x + polval.y * polval.y;
            }

        }
        outvalue *= TIMESCALE;
        scaled = __float2int_ru((outvalue - means[FFTUSE - threadIdx.x - 1]) / stdevs[FFTUSE - threadIdx.x - 1] * 32.0f + 128.0f);
        if (scaled > 255) {
            scaled = 255;
        } else if (scaled < 0) {
            scaled = 0;
        }
        out[outidx] = (unsigned char)scaled;
        inidx += FFTOUT * TIMEAVG;
        outidx += FFTUSE;
        outvalue = 0.0;
    }
}

__global__ void InitDivFactors(float *factors, size_t togenerate) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // NOTE: I don't want to be dividing by 0
    // NOTE: idx of 0 will not be used anyway
    if (idx < togenerate) {
        if (idx != 0) {
            factors[idx] = 1.0f / idx;
        } else {
            factors[idx] = idx;
        }
    }
}

__global__ void GetScalingFactorsKernel(float* __restrict__ indata, float *base, float *stdev, float *factors, int processed) {

    // NOTE: Filterbank file format coming in
    //float mean = indata[threadIdx.x];
    float mean = 0.0f;
    // NOTE: Depending whether I save STD or VAR at the end of every run
    // float estd = stdev[threadIdx.x];
    float estd = stdev[threadIdx.x] * stdev[threadIdx.x] * (processed - 1.0f);
    float oldmean = base[threadIdx.x];

    //float estd = 0.0f;
    //float oldmean = 0.0;

    float val = 0.0f;
    float diff = 0.0;
    // NOTE: There are 15625 output time samples per NACCUMULATE frames
    for (int isamp = 0; isamp < 15625; ++isamp) {
        val = indata[isamp * FFTUSE + threadIdx.x];
        diff = val - oldmean;
        mean = oldmean + diff * factors[processed + isamp + 1];
        estd += diff * (val - mean);
        oldmean = mean;
    }
    base[threadIdx.x] = mean;
    stdev[threadIdx.x] = sqrtf(estd / (float)(processed + 15625 - 1.0f));
    // stdev[threadIdx.x] = estd;
}
*/
