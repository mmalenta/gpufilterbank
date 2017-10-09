#include <iostream>
#include <fstream>
#include <string>

#include <cuda.h>
#include <cufft.h>

#include "errors.hpp"

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;

#define NACCUMULATE 4000

__global__ void UnpackKernel(unsigned char **in, float **out, size_t samples) {

    // NOTE: Each thread in the block processes 625 samples
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
__global__ void PowerKernel(cufftComplex **in, unsigned char **out, int nogulps, int gulpsize, int extra, unsigned int framet) {
    // NOTE: framet should start at 0 and increase by accumulate every time this kernel is called
    // NOTE: REALLY make sure it starts at 0
    // NOTE: I'M SERIOUS - FRAME TIME CALCULATIONS ARE BASED ON THIS ASSUMPTION
    unsigned int filtime = framet / ACC * gridDim.x * PERBLOCK + blockIdx.x * PERBLOCK;
    unsigned int filidx;
    unsigned int outidx;
    int inidx = blockIdx.x * PERBLOCK * TIMEAVG * FFTOUT + threadIdx.x + 1;

    float outvalue = 0.0f;
    cufftComplex polval;

    for (int isamp = 0; isamp < PERBLOCK; ++isamp) {

        // NOTE: Read the data from the incoming array
        for (int ipol = 0; ipol < 2; ++ipol) {
            for (int iavg = 0; iavg < TIMEAVG; iavg++) {
                polval = in[ipol][inidx + iavg * FFTOUT];
                outvalue += polval.x * polval.x + polval.y * polval.y;
            }

        }

        filidx = filtime % (nogulps * gulpsize);
        outidx = filidx * FFTUSE + threadIdx.x;

        outvalue *= TIMESCALE;

        out[0][outidx] = outvalue;
        // NOTE: Save to the extra part of the buffer
        if (filidx < extra) {
            out[0][outidx + nogulps * gulpsize * FFTUSE] = outvalue;
        }
        inidx += FFTOUT * TIMEAVG;
        filtime++;
        outvalue = 0.0;
    }
}

int main(int argc, char *argv[]) {

    string inpola, inpolb, outfil;

    inpola = std::string(argv[1]);
    inpolb = std::string(argv[2]);
    outfil = std::string(argv[3]);

    ifstream filepola(inpola.c_str(), ifstream::in | ifstream::binary);
    ifstream filepolb(inpolb.c_str(), ifstream::in | ifstream::binary);
    ofstream filfile(outfil.c_str(), ofstream::out | ofstream::binary);

    // TODO: Can save the filterbank header straight away, after the first header is read
    unsigned char vdifheadpola[32];
    unsigned char vdifheadpolb[32];
    filepola.read(reinterpret_cast<char*>(vdifheadpola), 32);
    filepolb.read(reinterpret_cast<char*>(vdifheadpolb), 32);

    unsigned int toread = NACCUMULATE * 8032;

    unsigned char *datapola = new unsigned char[toread];
    unsigned char *datapolb = new unsigned char[toread];

    filepola.seekg(0, filepola.end);
    unsigned int filelength = filepola.tellg();
    filepola.seekg(0, filepola.beg);

    unsigned int noblocks = (unsigned int)(filelength / toread);

    unsigned char *devpola;
    cudaCheckError(cudaMalloc((void**)&devpola, toread * sizeof(unsigned char)));
    unsigned char *devpolb;
    cudaCheckError(cudaMalloc((void**)&devpolb, toread * sizeof(unsigned char)));

    for (unsigned int iblock = 0; iblock < noblocks; ++iblock) {
        filepola.read(reinterpret_cast<char*>(datapola), toread);
        filepolb.read(reinterpret_cast<char*>(datapolb), toread);

        cudaCheckError(cudaMemcpy(devpola, datapola, toread * sizeof(unsigned char), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(devpolb, datapolb, toread * sizeof(unsigned char), cudaMemcpyHostToDevice));

        // UnpackKernel<<<>>>
        // cufftCheckError(cufftExecR2C());
        // PowerKernel<<<>>>

    }

    return 0;
}
