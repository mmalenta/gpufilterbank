#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cufft.h>

#include "filterbank.hpp"
#include "errors.hpp"

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;

#define DEBUG 1
#define GPURUN 0
#define NACCUMULATE 4000
#define NPOL 2
#define PERBLOCK 625
#define TIMEAVG 8
#define TIMESCALE 0.125
#define UNPACKFACTOR 4
#define VDIFSIZE 8000
#define FFTOUT 513
#define FFTUSE 512

struct FrameInfo {
    unsigned int frameno;
    unsigned int refsecond;
    unsigned int refepoch;
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
__global__ void PowerKernel(cufftComplex **in, float *out) {
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

        outidx = filidx * FFTUSE + threadIdx.x;

        outvalue *= TIMESCALE;

        out[outidx] = outvalue;
        inidx += FFTOUT * TIMEAVG;;
        outvalue = 0.0;
    }
}

int main(int argc, char *argv[]) {

    string inpola, inpolb, outfil, config;

    inpola = std::string(argv[1]);
    inpolb = std::string(argv[2]);
    outfil = std::string(argv[3]);
    config = std::string(argv[4]);

    FilHead filhead;
    ReadFilterbankHeader(config, filhead);

    // TODO: Make sure there are correct values for bandwidth and sampling time in the header after taking averaging into account

    ifstream filepola(inpola.c_str(), ifstream::in | ifstream::binary);
    ifstream filepolb(inpolb.c_str(), ifstream::in | ifstream::binary);
    ofstream filfile(outfil.c_str(), ofstream::out | ofstream::binary);

    // TODO: Can save the filterbank header straight away, after the first header is read
    unsigned char vdifheadpola[32];
    unsigned char vdifheadpolb[32];
    filepola.read(reinterpret_cast<char*>(vdifheadpola), 32);
    filepolb.read(reinterpret_cast<char*>(vdifheadpolb), 32);

    filepola.seekg(0, filepola.end);
    unsigned int filelengtha = filepola.tellg();
    filepola.seekg(0, filepola.beg);

    filepolb.seekg(0, filepolb.end);
    unsigned int filelengthb = filepolb.tellg();
    filepolb.seekg(0, filepolb.beg);

    unsigned int startframe;
    unsigned int startsecond;

    startframe = (unsigned int)(vdifheadpola[4] | (vdifheadpola[5] << 8) | (vdifheadpola[6] << 16));	// frame number in this second
    startsecond = (unsigned int)(vdifheadpola[0] | (vdifheadpola[1] << 8) | (vdifheadpola[2] << 16) | ((vdifheadpola[3] & 0x3f) << 24));

    if (DEBUG) {
        cout << "Starting time: " << startsecond << ":" << startframe << endl;
    }

    // NOTE: Need to read headers in
    unsigned int toread = NACCUMULATE * 8000;
    // NOTE: No more headers after unpacking
    unsigned int unpackedsize = NACCUMULATE * VDIFSIZE * UNPACKFACTOR;
    unsigned int fftedsize = unpackedsize / 1024 * FFTOUT;
    unsigned int powersize = fftedsize / TIMEAVG;

    cufftHandle fftplan;
    int fftsizes[1];
    fftsizes[0] = 2 * FFTUSE;
    int fftbatchsize = unpackedsize / fftsizes[0];
    cout << fftbatchsize << endl;
    cufftCheckError(cufftPlanMany(&fftplan, 1, fftsizes, NULL, 1, 512, NULL, 1, 512, CUFFT_R2C, fftbatchsize));

    unsigned char *tmppola = new unsigned char[toread];
    unsigned char *tmppolb = new unsigned char[toread];

    unsigned char *devpola;
    unsigned char *devpolb;
    unsigned char **datapol = new unsigned char*[NPOL];
    unsigned char **devpol;
    float **unpacked = new float*[NPOL];
    float **devunpacked;
    cufftComplex **ffted = new cufftComplex*[NPOL];
    cufftComplex **devffted;
    float *devpower;
    float *tmppower = new float[powersize];

    if (GPURUN) {
        cudaCheckError(cudaMalloc((void**)&devpola, toread * sizeof(unsigned char)));
        cudaCheckError(cudaMalloc((void**)&devpolb, toread * sizeof(unsigned char)));

        cudaCheckError(cudaMalloc((void**)&devpol, NPOL * sizeof(unsigned char*)));
        cudaCheckError(cudaMalloc((void**)&datapol[0], toread * sizeof(unsigned char)));
        cudaCheckError(cudaMalloc((void**)&datapol[1], toread * sizeof(unsigned char)));
        cudaCheckError(cudaMemcpy(devpol, datapol, NPOL * sizeof(unsigned char*), cudaMemcpyHostToDevice));

        cudaCheckError(cudaMalloc((void**)&devunpacked, NPOL * sizeof(float*)));
        cudaCheckError(cudaMalloc((void**)&unpacked[0], unpackedsize * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&unpacked[1], unpackedsize * sizeof(float)));
        cudaCheckError(cudaMemcpy(devunpacked, unpacked, NPOL * sizeof(float*), cudaMemcpyHostToDevice));

        cudaCheckError(cudaMalloc((void**)&devffted, NPOL * sizeof(cufftComplex*)));
        cudaCheckError(cudaMalloc((void**)&ffted[0], fftedsize * sizeof(cufftComplex)));
        cudaCheckError(cudaMalloc((void**)&ffted[1], fftedsize * sizeof(cufftComplex)));
        cudaCheckError(cudaMemcpy(devffted, ffted, NPOL * sizeof(cufftComplex*), cudaMemcpyHostToDevice));

        cudaCheckError(cudaMalloc((void**)&devpower, powersize * sizeof(float)));
    }

    vector<std::pair<FrameInfo, FrameInfo>> vdifframes;

    FrameInfo tmpframea, tmpframeb;
    int refsecond;
    int frameno;
    int epoch;

    WriteFilterbankHeader(filfile, filhead);

    while((filepola.tellg() < (filelengtha - NACCUMULATE * 8000)) && (filepolb.tellg() < (filelengthb - NACCUMULATE * 8000))) {
        //cout << filepola.tellg() << endl;
        // NOTE: This implementation
        for (int iacc = 0; iacc < NACCUMULATE; ++iacc) {
            filepola.read(reinterpret_cast<char*>(vdifheadpola), 32);
            filepolb.read(reinterpret_cast<char*>(vdifheadpolb), 32);
            filepola.read(reinterpret_cast<char*>(tmppola) + iacc * 8000, 8000);
            filepolb.read(reinterpret_cast<char*>(tmppolb) + iacc * 8000, 8000);

            refsecond = (unsigned int)(vdifheadpola[0] | (vdifheadpola[1] << 8) | (vdifheadpola[2] << 16) | ((vdifheadpola[3] & 0x3f) << 24));
            frameno = (unsigned int)(vdifheadpola[4] | (vdifheadpola[5] << 8) | (vdifheadpola[6] << 16));
            epoch = (unsigned int)(vdifheadpola[7] & 0x3f);
//            frameno += (refsecond - startsecond) * 4000;

            tmpframea.frameno = frameno;
            tmpframea.refsecond = refsecond;
            tmpframea.refepoch = epoch;

            refsecond = (unsigned int)(vdifheadpolb[0] | (vdifheadpolb[1] << 8) | (vdifheadpolb[2] << 16) | ((vdifheadpolb[3] & 0x3f) << 24));
            frameno = (unsigned int)(vdifheadpolb[4] | (vdifheadpolb[5] << 8) | (vdifheadpolb[6] << 16));
            epoch = (unsigned int)(vdifheadpolb[7] & 0x3f);
//            frameno += (refsecond - startsecond) * 4000;

            tmpframeb.frameno = frameno;
            tmpframeb.refsecond = refsecond;
            tmpframeb.refepoch = epoch;

            vdifframes.push_back(std::make_pair(tmpframea, tmpframeb));

            // NOTE: Can use subtract startframe to put frame count at 0 and use that to save into the buffer

        }

        if (GPURUN) {
            cudaCheckError(cudaMemcpy(datapol[0], tmppola, toread * sizeof(unsigned char), cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(datapol[1], tmppolb, toread * sizeof(unsigned char), cudaMemcpyHostToDevice));

            UnpackKernel<<<50, 1024, 0, 0>>>(devpol, devunpacked, toread);
            for (int ipol = 0; ipol < NPOL; ++ipol) {
                cufftCheckError(cufftExecR2C(fftplan, unpacked[ipol], ffted[ipol]));
            }
            PowerKernel<<<25, 512, 0, 0>>>(devffted, devpower);
            cudaCheckError(cudaMemcpy(devpower, tmppower, powersize * sizeof(float), cudaMemcpyDeviceToHost));

            filfile.write(reinterpret_cast<char*>(tmppower), powersize * sizeof(float));
        }
        cout << "Completed " << (float)filepola.tellg() / (float)filelengtha * 100.0f << "%\r";
        cout.flush();
    }

    cout << endl;
    filfile.close();

    if (DEBUG) {
        std::ofstream outframes("dataframes.dat");

        outframes << "Ref Epoch A\tRef second A\tRef frame A\tRef Epoch B\tRef second B\tRef frame b\n";
        for (auto iframe = vdifframes.begin(); iframe != vdifframes.end(); ++iframe) {
            outframes << iframe->first.refepoch << "\t" << iframe->first.refsecond << "\t" << iframe->first.frameno << "\t"
            << iframe->second.refepoch << "\t" << iframe->second.refsecond << "\t" << iframe->second.frameno << endl;
        }

        outframes.close();
    }
    return 0;
}
