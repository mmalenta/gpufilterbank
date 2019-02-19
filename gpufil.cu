#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "dada.hpp"
#include "errors.hpp"
#include "filterbank.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;

#define DEBUG 0
#define GPURUN 1
#define NACCUMULATE 4000
#define PERBLOCK 625
#define TIMESCALE 0.125
#define UNPACKFACTOR 4
#define VDIFSIZE 8000
#define FFTOUT 257
#define FFTUSE 256

#define COMPLEX 1
#define INCHANS 1
#define OUTCHANS 1024
#define NPOL 2
#define TIMEAVG 4
#define INBANDS 1

struct FrameInfo {
    unsigned int frameno;
    unsigned int refsecond;
    unsigned int refepoch;
};

struct Timing {
    float readtime;
    float scaletime;
    float filtime;
    float savetime;
    float totaltime;
    float intertime;
};
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

int main(int argc, char *argv[]) {

    string inpola = "";
    string inpolb = "";
    string outfil = "";
    string config = "";
    string dadastr = "";
    double readsec; 
    bool scaling = false;
    bool saveinter = false;

    if ((argc < 5) || (argv[1] == "-h") || (argv[1] == "--help")) {
        cout << "Incorrect number of arguments!" << endl;
        cout << "Command line options:" << endl
                << "-a <filename> - input file for polarisation a" << endl
                << "-b <filename> - input file for polarisation b" << endl
                << "-d <filename> - input DADA file" << endl
                << "-o <filename> - output filterbank file" << endl
                << "-c <filename> - input configuration file" << endl
                << "-r <number> - number of seconds to process - CURRENTLY NOT WORKING" << endl
                << "-s - enable scaling from 32 bits to 8 bits" << endl
                << "-i - enable saving the intermediate data products" << endl
                << "-h, --help - display this message" << endl;
        exit(EXIT_SUCCESS);
    }

    for (int iarg = 0; iarg < argc; ++iarg) {
        if (std::string(argv[iarg]) == "-a") {
            iarg++;
            inpola = std::string(argv[iarg]);
        } else if (std::string(argv[iarg]) == "-b") {
            iarg++;
            inpolb = std::string(argv[iarg]);
        } else if (std::string(argv[iarg]) == "-d") {
            iarg++;
            dadastr = std::string(argv[iarg]);
        } else if (std::string(argv[iarg]) == "-o") {
            iarg++;
            outfil = std::string(argv[iarg]);
        } else if (std::string(argv[iarg]) == "-c") {
            iarg++;
            config = std::string(argv[iarg]);
        } else if (std::string(argv[iarg]) == "-s") {
            cout << "Will scale the data to 8 bits" << endl;
            scaling = true;
        } else if (std::string(argv[iarg]) == "-i") {
            cout << "Will save the intermediate products" << endl;
            saveinter = true;
        } else if (std::string(argv[iarg]) == "-r") {
            iarg++;
            readsec = std::stod(argv[iarg]);
        }
    }

    if (!inpola.empty() && !dadastr.empty()) {
        cerr << "It's one or the other: DADA or VDIF, not both!" << endl;
        return 1;
    }

    if (!dadastr.empty()) {

        cout << "Input file: " << dadastr << endl;
        
        std::ifstream indada(dadastr.c_str(), std::ios_base::binary);        
        long long filesize = 0;
        indada.seekg(0, indada.end);
        filesize = indada.tellg() - 4096L;
        indada.seekg(0, indada.beg);

        // NOTE: 4 bytes per full time sample: 1 byte sampling, 2 polarisations, complex number
        size_t totalsamples = filesize / 4;
        if (filesize != totalsamples * 4) {
            std::cerr << "A non-integer number of time samples was read - something went very wrong!" << std::endl;
            return 1;
        }

        std::cout << "File size: " << filesize / 1024.0f / 1024.0f << "MiB with " << totalsamples << " time samples" << std::endl;
        // NOTE: That simply ensures that we only process the integer number of final filterbank channels
        totalsamples = (int)((float)totalsamples / (OUTCHANS * TIMEAVG)) * OUTCHANS * TIMEAVG;
        std::cout << "Will use first" << totalsamples << " samples" << std::endl;

        FilHead filhead = {};
        ReadDadaHeader(indada, filhead);
        if (!scaling) {
            filhead.nbits = 32;
        }
        filhead.nchans = OUTCHANS;
        filhead.tsamp = filhead.tsamp * OUTCHANS * TIMEAVG;
        PrintFilterbankHeader(filhead);
        
        size_t freemem = 0;
        size_t totalmem = 0;
        cudaCheckError(cudaMemGetInfo(&freemem, &totalmem));
        // NOTE: Let's liffh just 25% of what's free, because cuFFT happens...
        freemem = freemem * 0.25;
        std::cout << "Total memory: " << totalmem / 1024.0f / 1024.0f << "MiB, with " << freemem / 1024.0f / 1024.0f << "MiB free" << std::endl;
        
        // original file + original file cast to cufftComplex for FFT + output filterbank file saved as 32 bit float
        size_t needmem = 4 * totalsamples + 4 * totalsamples * 4 + totalsamples / OUTCHANS / TIMEAVG * OUTCHANS * 4;
        std::cout << "Need " << needmem / 1024.0f / 1024.0f << "MiB on the device" << std::endl;
        
        int nblocks = 0;
        size_t sampperblock = 0;
        size_t remsamp = 0;
        
        if (needmem < freemem) {
            std::cout << "Can store everything in global memory at once..." << std::endl;
            nblocks = 1;
            sampperblock = totalsamples;
        } else {
            std::cout << "We need to divide the job..." << std::endl;

            sampperblock = (int)((float)freemem / ((float)(OUTCHANS * TIMEAVG) * (4.0f + 16.0f + 4.0f / (float)TIMEAVG))) * OUTCHANS * TIMEAVG;
            nblocks = (int)(totalsamples / sampperblock);
            remsamp = totalsamples - nblocks * sampperblock;

            std::cout << "Will process the data in " << nblocks << " blocks, with "
                        << sampperblock << " samples per block";
            if (remsamp) {
                std::cout << " and an extra block with " << remsamp << " samples at the end";
            }
            std::cout << std::endl;
        }

        std::ofstream filfile(outfil.c_str(), std::ios_base::binary);
        WriteFilterbankHeader(filfile, filhead);

        /**** ####
        // STAGE: MEMORY AND FFT
        #### ****/
        // NOTE: Factor of 4 to account for 2 polarisations and complex components for every time sample
        size_t blockread = sampperblock * 4;
        size_t remread = remsamp * 4;
        
        cufftHandle fftplan;
        int fftsizes[1];
        fftsizes[0] = OUTCHANS;
        // NOTE: Factor of 2 to account for 2 polarisations
        int fftbatchsize = sampperblock * 2 / fftsizes[0];
        cufftCheckError(cufftPlanMany(&fftplan, 1, fftsizes, NULL, 1, OUTCHANS, NULL, 1, OUTCHANS, CUFFT_C2C, fftbatchsize));
 
        unsigned char *hostvoltage = new unsigned char[blockread];
        unsigned char *devicevoltage = new unsigned char[blockread];
        cudaCheckError(cudaMalloc((void**)&devicevoltage, blockread * sizeof(unsigned char)));

        cufftComplex *devicefft;
        cudaCheckError(cudaMalloc((void**)&devicefft, sampperblock * 2 * sizeof(cufftComplex)));

        size_t powersize = sampperblock / OUTCHANS * OUTCHANS / TIMEAVG;
        float *hostpower = new float[powersize];
        float *devicepower;
        cudaCheckError(cudaMalloc((void**)&devicepower, powersize * sizeof(float)))

        float *hostband = new float[OUTCHANS];
        float *deviceband;
        cudaCheckError(cudaMalloc((void**)&deviceband, OUTCHANS * sizeof(float)));

        // NOTE: Just in case I did something wrong
        indada.seekg(4096, indada.beg);

        for (int iblock = 0; iblock < nblocks; iblock++) {

            std::cout << "Processing block " << iblock << "..." << std::endl;

            indada.read(reinterpret_cast<char*>(hostvoltage), blockread * sizeof(unsigned char));

            cudaCheckError(cudaMemcpy(devicevoltage, hostvoltage, blockread * sizeof(unsigned char), cudaMemcpyHostToDevice));

            dim3 block (OUTCHANS, 1, 1);
            dim3 grid (64, 1, 1);

            UnpackDadaKernel<<<grid, block, 0, 0>>>(sampperblock, reinterpret_cast<uchar4*>(devicevoltage), devicefft);
            cudaCheckError(cudaGetLastError());

            cufftCheckError(cufftExecC2C(fftplan, devicefft, devicefft, CUFFT_FORWARD));

            DetectDadaKernel<<<grid, block, 0, 0>>>(sampperblock / OUTCHANS, devicefft, devicepower);
            cudaCheckError(cudaGetLastError());

            BandpassKernel<<<1, OUTCHANS, 0, 0>>>(sampperblock / OUTCHANS / TIMEAVG, devicepower, deviceband);
            cudaCheckError(cudaGetLastError());

            cudaCheckError(cudaMemcpy(hostpower, devicepower, powersize * sizeof(float), cudaMemcpyDeviceToHost));

            filfile.write(reinterpret_cast<char*>(hostpower), powersize * sizeof(float));
        } 
        
        cufftCheckError(cufftDestroy(fftplan));

        if (remsamp) {

            std::cout << "Processing the remainder block..." << std::endl;

            indada.read(reinterpret_cast<char*>(hostvoltage), remread * sizeof(unsigned char));

            cudaCheckError(cudaMemcpy(devicevoltage, hostvoltage, remread * sizeof(unsigned char), cudaMemcpyHostToDevice));

            dim3 block(OUTCHANS, 1, 1);
            dim3 grid (64, 1, 1);

            UnpackDadaKernel<<<grid, block, 0, 0>>>(remsamp, reinterpret_cast<uchar4*>(devicevoltage), devicefft);
            cudaCheckError(cudaGetLastError());

            cufftHandle fftplanrem;
            int fftrembatchsize = remsamp * 2 / fftsizes[0];
            cufftCheckError(cufftPlanMany(&fftplanrem, 1, fftsizes, NULL, 1, OUTCHANS, NULL, 1, OUTCHANS, CUFFT_C2C, fftrembatchsize));

            cufftCheckError(cufftExecC2C(fftplanrem, devicefft, devicefft, CUFFT_FORWARD));

            DetectDadaKernel<<<grid, block, 0, 0>>>(remsamp / OUTCHANS, devicefft, devicepower);
            cudaCheckError(cudaGetLastError());

            BandpassKernel<<<1, OUTCHANS, 0, 0>>>(remsamp / OUTCHANS / TIMEAVG, devicepower, deviceband);
            cudaCheckError(cudaGetLastError());

            cudaCheckError(cudaMemcpy(hostpower, devicepower, remsamp / OUTCHANS / TIMEAVG * OUTCHANS * sizeof(float), cudaMemcpyDeviceToHost));

            filfile.write(reinterpret_cast<char*>(hostpower), remsamp / OUTCHANS / TIMEAVG * OUTCHANS * sizeof(float));

            cufftCheckError(cufftDestroy(fftplanrem)); 
        }

        cudaCheckError(cudaMemcpy(hostband, deviceband, OUTCHANS * sizeof(float), cudaMemcpyDeviceToHost));

        std::ofstream bandout("bandpass.dat");

        if (bandout) {
            for (int ichan = 0; ichan < OUTCHANS; ++ichan) {
                bandout << hostband[ichan] << std::endl;
            }
        }

        bandout.close();
        filfile.close();
        indada.close();

        cudaFree(deviceband);
        cudaFree(devicepower);
        cudaFree(devicefft);
        cudaFree(devicevoltage);

        delete [] hostband;
        delete [] hostpower;
        delete [] hostvoltage;

    } 

    // NOTE: This code is in a bit of a bad state - need to clean it up
    // else if (!inpola.empty() & !inpolb.empty()) {

    //     cout << "Input files: " << inpola << " " << inpolb << endl;

    //     FilHead filhead;
    //     ReadFilterbankHeader(config, filhead);



        
    //     if (scaling) {
    //         filhead.nbits = 8;
    //     }
    
    //     // TODO: This will be wrong for R2C FFT
    //     filhead.tsamp = 1.0 / (2.0 * filhead.foff) * 2 * FFTUSE * TIMEAVG;
    //     // TODO: Make sure it is the middle of the top frequency channel
    //     filhead.fch1 = (filhead.fch1 + filhead.foff / 2.0f) * 1e-06;
    //     filhead.nchans = FFTUSE;
    //     filhead.foff = -1.0 * filhead.foff / FFTUSE * 1e-06 ;
    
    //     filhead.fch1 = filhead.fch1 + filhead.foff / 2.0;
    
    //     if (DEBUG) {
    //         cout << "Some header info:\n"
    //                 << "Raw file: " << filhead.rawfile << endl
    //                 << "Source name: " << filhead.source << endl
    //                 << "Azimuth: " << filhead.az << endl
    //                 << "Zenith angle: " << filhead.za << endl
    //                 << "Declination: " << filhead.dec << endl
    //                 << "Right ascension: " << filhead.ra << endl
    //                 << "Top channel frequency: " << filhead.fch1 << endl
    //                 << "Channel bandwidth: " << filhead.foff << endl
    //                 << "Number of channels: " << filhead.nchans << endl
    //                 << "Sampling time: " << filhead.tsamp << endl
    //                 << "Bits per sample: " << filhead.nbits << endl;
    //     }
    
    //     // TODO: Make sure there are correct values for bandwidth and sampling time in the header after taking averaging into account
    
    //     ifstream filepola(inpola.c_str(), ifstream::in | ifstream::binary);
    //     ifstream filepolb(inpolb.c_str(), ifstream::in | ifstream::binary);
    //     ofstream filfile(outfil.c_str(), ofstream::out | ofstream::binary);
    
    //     if (!filepola || !filepolb) {
    //     if (!filepola) {
    //             cout << "Could not open file " << inpola << endl;
    //         }
    //         if (!filepolb) {
    //             cout << "Could not open file " << inpolb << endl;
    //         }
    //         exit(EXIT_FAILURE);
    //     }
    //     // TODO: Can save the filterbank header straight away, after the first header is read
    //     unsigned char vdifheadpola[32];
    //     unsigned char vdifheadpolb[32];
    //     filepola.read(reinterpret_cast<char*>(vdifheadpola), 32);
    //     filepolb.read(reinterpret_cast<char*>(vdifheadpolb), 32);
    
    //     filepola.seekg(0, filepola.end);
    //     long long filelengtha = filepola.tellg();
    //     filepola.seekg(0, filepola.beg);
    
    //     filepolb.seekg(0, filepolb.end);
    //     long long filelengthb = filepolb.tellg();
    //     filepolb.seekg(0, filepolb.beg);
    
    //     unsigned int startframe;
    //     unsigned int startsecond;
    
    //     cout << filelengtha << " " << filelengthb << endl;
    
    //     startframe = (unsigned int)(vdifheadpola[4] | (vdifheadpola[5] << 8) | (vdifheadpola[6] << 16));	// frame number in this second
    //     startsecond = (unsigned int)(vdifheadpola[0] | (vdifheadpola[1] << 8) | (vdifheadpola[2] << 16) | ((vdifheadpola[3] & 0x3f) << 24));
    
    //     if (DEBUG) {
    //         cout << "Starting time: " << startsecond << ":" << startframe << endl;
    //     }
    
    //     // NOTE: Need to read headers in
    //     unsigned int toread = NACCUMULATE * 8000;
    //     // NOTE: No more headers after unpacking
    //     unsigned int unpackedsize = NACCUMULATE * VDIFSIZE * UNPACKFACTOR;
    //     unsigned int fftedsize = unpackedsize / (2 * FFTUSE) * FFTOUT;
    //     unsigned int powersize = unpackedsize / (2 * FFTUSE) * FFTUSE / TIMEAVG;
    
    //     cufftHandle fftplan;
    //     int fftsizes[1];
    //     fftsizes[0] = 2 * FFTUSE;
    //     int fftbatchsize = unpackedsize / fftsizes[0];
    //     cout << fftbatchsize << endl;
    //     cufftCheckError(cufftPlanMany(&fftplan, 1, fftsizes, NULL, 1, FFTUSE, NULL, 1, FFTUSE, CUFFT_R2C, fftbatchsize));
    
    //     unsigned char *tmppola = new unsigned char[toread];
    //     unsigned char *tmppolb = new unsigned char[toread];
    
    //     unsigned char *devpola;
    //     unsigned char *devpolb;
    //     unsigned char **datapol = new unsigned char*[NPOL];
    //     unsigned char **devpol;
    //     float **unpacked = new float*[NPOL];
    //     float **devunpacked;
    //     cufftComplex **ffted = new cufftComplex*[NPOL];
    //     cufftComplex **devffted;
        
    //     unsigned char *devpower;
    //     unsigned char *tmppower = new unsigned char[powersize * filhead.nbits / 8];
    
    //     if (GPURUN) {
    //         cudaCheckError(cudaMalloc((void**)&devpola, toread * sizeof(unsigned char)));
    //         cudaCheckError(cudaMalloc((void**)&devpolb, toread * sizeof(unsigned char)));
    
    //         cudaCheckError(cudaMalloc((void**)&devpol, NPOL * sizeof(unsigned char*)));
    //         cudaCheckError(cudaMalloc((void**)&datapol[0], toread * sizeof(unsigned char)));
    //         cudaCheckError(cudaMalloc((void**)&datapol[1], toread * sizeof(unsigned char)));
    //         cudaCheckError(cudaMemcpy(devpol, datapol, NPOL * sizeof(unsigned char*), cudaMemcpyHostToDevice));
    
    //         cudaCheckError(cudaMalloc((void**)&devunpacked, NPOL * sizeof(float*)));
    //         cudaCheckError(cudaMalloc((void**)&unpacked[0], unpackedsize * sizeof(float)));
    //         cudaCheckError(cudaMalloc((void**)&unpacked[1], unpackedsize * sizeof(float)));
    //         cudaCheckError(cudaMemcpy(devunpacked, unpacked, NPOL * sizeof(float*), cudaMemcpyHostToDevice));
    
    //         cudaCheckError(cudaMalloc((void**)&devffted, NPOL * sizeof(cufftComplex*)));
    //         cudaCheckError(cudaMalloc((void**)&ffted[0], fftedsize * sizeof(cufftComplex)));
    //         cudaCheckError(cudaMalloc((void**)&ffted[1], fftedsize * sizeof(cufftComplex)));
    //         cudaCheckError(cudaMemcpy(devffted, ffted, NPOL * sizeof(cufftComplex*), cudaMemcpyHostToDevice));
    
    //         cudaCheckError(cudaMalloc((void**)&devpower, powersize * (filhead.nbits / 8)));
    //     }
    
    //     vector<std::pair<FrameInfo, FrameInfo>> vdifframes;
    
    //     FrameInfo tmpframea, tmpframeb;
    //     int refsecond;
    //     int frameno;
    //     int epoch;
    
    //     WriteFilterbankHeader(filfile, filhead);
       
    //     Timing runtimes;
    //     runtimes.readtime = 0.0f;
    //     runtimes.scaletime = 0.0f;
    //     runtimes.filtime = 0.0f;
    //     runtimes.savetime = 0.0f;
    //     runtimes.totaltime = 0.0f;
    //     runtimes.intertime = 0.0f;
    
    //     std::chrono::time_point<std::chrono::steady_clock> readstart, readend, scalestart, scaleend, filstart, filend, savestart, saveend, interstart, interend;
    
    //     float *tmpunpackeda = new float[unpackedsize];
    //     float *tmpunpackedb = new float[unpackedsize];
    //     cufftComplex *tmpffta = new cufftComplex[fftedsize];
    //     cufftComplex *tmpfftb = new cufftComplex[fftedsize];
    
    //     bool saved = false;
    
    //     //float *dmeans;
    //     //float *dstdevs;
    //     //cudaCheckError(cudaMalloc((void**)&dmeans, FFTUSE * sizeof(float)));
    //     //cudaCheckError(cudaMalloc((void**)&dstdevs, FFTUSE * sizeof(float)));
    
    //     thrust::device_vector<float> dmeans, dstdevs;
    //     dmeans.resize(FFTUSE);
    //     dstdevs.resize(FFTUSE);
    //     thrust::fill(dmeans.begin(), dmeans.end(), 0.0f);
    //     thrust::fill(dstdevs.begin(), dstdevs.end(), 0.0f);
    //     float *pdmeans = thrust::raw_pointer_cast(dmeans.data());
    //     float *pdstdevs = thrust::raw_pointer_cast(dstdevs.data());    
    
    //     cout << "Size of the device vectors: " << dmeans.size() << " " << dstdevs.size() << endl;
    
    //     scalestart = std::chrono::steady_clock::now();
    
    //     // NOTE: Use first 5 accumulates of data to obtain scaling factors
    //     if (scaling) {
    
    //         size_t divfactors = 5 * powersize / FFTUSE;
    //         thrust::device_vector<float> dfactors; 
    //         dfactors.resize(divfactors + 1);
    //         thrust::sequence(dfactors.begin(), dfactors.end());
    //         thrust::transform(dfactors.begin(), dfactors.end(), dfactors.begin(), FactorFunctor());
    //         float *pdfactors = thrust::raw_pointer_cast(dfactors.data());
    
    //         //float *dfactors;
    //         //size_t divfactors = 5 * powersize / FFTUSE;
    //         //cudaCheckError(cudaMalloc((void**)&dfactors, divfactors * sizeof(float)));
    //         //int scalethreads = 1024;
    //         //int scaleblocks = (divfactors - 1) / scalethreads + 1;
    //         //cout << "Div factors blocks: " << scaleblocks << " and threads: " << scalethreads << endl;
    //         //InitDivFactors<<<scaleblocks, scalethreads>>>(dfactors, divfactors);
    //         //cudaCheckError(cudaDeviceSynchronize());
    //         //cudaCheckError(cudaGetLastError());
    //         size_t processed = 0;
    
    //         float *tmpdpower;
    //         cudaCheckError(cudaMalloc((void**)&tmpdpower, powersize * sizeof(float)));
    
    //     while((filepola.tellg() < (5 * NACCUMULATE * 8032)) && (filepolb.tellg() < (5 * NACCUMULATE * 8032))) {
    //             for (int iacc = 0; iacc < NACCUMULATE; ++iacc) {
    //                 filepola.read(reinterpret_cast<char*>(vdifheadpola), 32);
    //                 filepolb.read(reinterpret_cast<char*>(vdifheadpolb), 32);
    //                 filepola.read(reinterpret_cast<char*>(tmppola) + iacc * 8000, 8000);
    //                 filepolb.read(reinterpret_cast<char*>(tmppolb) + iacc * 8000, 8000);
    //             }
    
    //             cudaCheckError(cudaMemcpy(datapol[0], tmppola, NACCUMULATE * 8000 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    //             cudaCheckError(cudaMemcpy(datapol[1], tmppolb, NACCUMULATE * 8000 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    //             UnpackKernel<<<50, 1024, 0, 0>>>(devpol, devunpacked, toread);
    //             for (int ipol = 0; ipol < NPOL; ++ipol) {
    //                 cufftCheckError(cufftExecR2C(fftplan, unpacked[ipol], ffted[ipol]));
    //             }
    //             DetectKernel<<<25, FFTUSE, 0, 0>>>(devffted, tmpdpower);
    //             cudaCheckError(cudaDeviceSynchronize());
    //             GetScalingFactorsKernel<<<1, FFTUSE, 0, 0>>>(tmpdpower, pdmeans, pdstdevs, pdfactors, processed);
    //             processed += (powersize / FFTUSE);
    //             cudaCheckError(cudaDeviceSynchronize());
    //         }
    
    //         //float *hmeans = new float[FFTUSE];
    //         //float *hstdevs = new float[FFTUSE];
    
    //         //cudaCheckError(cudaMemcpy(hmeans, dmeans, FFTUSE * sizeof(float), cudaMemcpyDeviceToHost));
    //         //cudaCheckError(cudaMemcpy(hstdevs, dstdevs, FFTUSE * sizeof(float), cudaMemcpyDeviceToHost));
    
    //         thrust::host_vector<float> hmeans = dmeans;
    //         thrust::host_vector<float> hstdevs = dstdevs;
    
    //         std::ofstream statsfile("mean_stdev.dat");
    
    //         cout << "Size of host vector:" << hmeans.size() << endl;
     
    //         if (statsfile) {
    //             for (int ichan = 0; ichan < hmeans.size(); ++ichan) {
    //                 statsfile << hmeans[ichan] << " " << hstdevs[ichan] << endl;
    //             }
    //         } else {
    //             cerr << "Could not open the stats file" << endl;
    //         }
    
    //         statsfile.close();
    
    //         cudaFree(tmpdpower);
             
    //     }
    
    //     scaleend = std::chrono::steady_clock::now();
    
    //     runtimes.scaletime = std::chrono::duration<float>(scaleend - scalestart).count();
    
    //     filepola.seekg(0, filepola.beg);
    //     filepolb.seekg(0, filepolb.beg);
    
    //     std::ofstream unpackedfilea ((outfil + ".unp0").c_str(), std::ios_base::binary);
    //     std::ofstream unpackedfileb ((outfil + ".unp1").c_str(), std::ios_base::binary);
    //     std::ofstream fftfilea ((outfil + ".fft0").c_str(), std::ios_base::binary);
    //     std::ofstream fftfileb ((outfil + ".fft1").c_str(), std::ios_base::binary);
    
    //     while((filepola.tellg() < (filelengtha - NACCUMULATE * 8000)) && (filepolb.tellg() < (filelengthb - NACCUMULATE * 8000))) {
    //         //cout << filepola.tellg() << endl;
    //         // NOTE: This implementation
    //         for (int iacc = 0; iacc < NACCUMULATE; ++iacc) {
    //         readstart = std::chrono::steady_clock::now();
    //             filepola.read(reinterpret_cast<char*>(vdifheadpola), 32);
    //             filepolb.read(reinterpret_cast<char*>(vdifheadpolb), 32);
    //             filepola.read(reinterpret_cast<char*>(tmppola) + iacc * 8000, 8000);
    //             filepolb.read(reinterpret_cast<char*>(tmppolb) + iacc * 8000, 8000);
    //             readend = std::chrono::steady_clock::now();
    //             runtimes.readtime += std::chrono::duration<float>(readend - readstart).count();
    
    //             refsecond = (unsigned int)(vdifheadpola[0] | (vdifheadpola[1] << 8) | (vdifheadpola[2] << 16) | ((vdifheadpola[3] & 0x3f) << 24));
    //             frameno = (unsigned int)(vdifheadpola[4] | (vdifheadpola[5] << 8) | (vdifheadpola[6] << 16));
    //             epoch = (unsigned int)(vdifheadpola[7] & 0x3f);
    // //            frameno += (refsecond - startsecond) * 4000;
    
    //             tmpframea.frameno = frameno;
    //             tmpframea.refsecond = refsecond;
    //             tmpframea.refepoch = epoch;
    
    //             refsecond = (unsigned int)(vdifheadpolb[0] | (vdifheadpolb[1] << 8) | (vdifheadpolb[2] << 16) | ((vdifheadpolb[3] & 0x3f) << 24));
    //             frameno = (unsigned int)(vdifheadpolb[4] | (vdifheadpolb[5] << 8) | (vdifheadpolb[6] << 16));
    //             epoch = (unsigned int)(vdifheadpolb[7] & 0x3f);
    // //            frameno += (refsecond - startsecond) * 4000;
    
    //             tmpframeb.frameno = frameno;
    //             tmpframeb.refsecond = refsecond;
    //             tmpframeb.refepoch = epoch;
    
    //             vdifframes.push_back(std::make_pair(tmpframea, tmpframeb));
    
    //             // NOTE: Can use subtract startframe to put frame count at 0 and use that to save into the buffer
    
    //         }
     
    //        if (GPURUN) {
    //             filstart = std::chrono::steady_clock::now();
    //             cudaCheckError(cudaMemcpy(datapol[0], tmppola, NACCUMULATE * 8000 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    //             cudaCheckError(cudaMemcpy(datapol[1], tmppolb, NACCUMULATE * 8000 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    //             UnpackKernel<<<50, 1024, 0, 0>>>(devpol, devunpacked, toread);
    //             for (int ipol = 0; ipol < NPOL; ++ipol) {
    //                 cufftCheckError(cufftExecR2C(fftplan, unpacked[ipol], ffted[ipol]));
    //             }
    
    //             if (filhead.nbits == 8) {
    //                 DetectScaleKernel<<<25, FFTUSE, 0, 0>>>(devffted, reinterpret_cast<unsigned char*>(devpower), pdmeans, pdstdevs);
    //             } else if (filhead.nbits == 32) {
    //                 DetectKernel<<<25, FFTUSE, 0, 0>>>(devffted, reinterpret_cast<float*>(devpower));
    //             } else {
    //                 cerr << "Unsupported option! Will use float!" << endl;
    //                 DetectKernel<<<25, FFTUSE, 0, 0>>>(devffted, reinterpret_cast<float*>(devpower));
    //             }
    
    //             //PowerKernel<<<25, FFTUSE, 0, 0>>>(devffted, devpower);
    //             cudaCheckError(cudaDeviceSynchronize());
    //             cudaCheckError(cudaMemcpy(tmppower, devpower, powersize * filhead.nbits / 8, cudaMemcpyDeviceToHost));
                
    //             if (!saved) {
    //                 std::ofstream unpackedfile("unpacked.dat");
    //         cudaCheckError(cudaMemcpy(tmpunpackeda, unpacked[0], 2 * 8000 * 4 * sizeof(float), cudaMemcpyDeviceToHost));
    //         cudaCheckError(cudaMemcpy(tmpunpackedb, unpacked[1], 2 * 8000 * 4 * sizeof(float), cudaMemcpyDeviceToHost));
    //         for (int isamp = 0; isamp < 2 * 8000 * 4; ++isamp) {
    //                     unpackedfile << tmpunpackeda[isamp] << " " << tmpunpackedb[isamp] << endl;
    //                 }
    //                 unpackedfile.close();
    //                 saved = true;
    //             }
    
    //             filend = std::chrono::steady_clock::now();
    //             runtimes.filtime += std::chrono::duration<float>(filend - filstart).count();
                
    //             savestart = std::chrono::steady_clock::now(); 
    //             filfile.write(reinterpret_cast<char*>(tmppower), powersize * filhead.nbits / 8);
    //             saveend = std::chrono::steady_clock::now();
    //             runtimes.savetime += std::chrono::duration<float>(saveend - savestart).count();
    
       
    
    //             if (saveinter) {
    
    //                 interstart = std::chrono::steady_clock::now();
    
    //                 cudaCheckError(cudaMemcpy(tmpunpackeda, unpacked[0], unpackedsize * sizeof(float), cudaMemcpyDeviceToHost));
    //         cudaCheckError(cudaMemcpy(tmpunpackedb, unpacked[1], unpackedsize * sizeof(float), cudaMemcpyDeviceToHost));
    //                 /*for (int isamp = 0; isamp < unpackedsize; ++isamp) {
    //                     unpackedfile << tmpunpackeda[isamp] << " " << tmpunpackedb[isamp] << endl;
    //                 }*/
    
    //                 unpackedfilea.write(reinterpret_cast<char*>(tmpunpackeda), unpackedsize * sizeof(float));
    //                 unpackedfileb.write(reinterpret_cast<char*>(tmpunpackedb), unpackedsize * sizeof(float));
    
    //                 cudaCheckError(cudaMemcpy(tmpffta, ffted[0], fftedsize * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    //                 cudaCheckError(cudaMemcpy(tmpfftb, ffted[1], fftedsize * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    //                 /*for (int isamp = 0; isamp < fftedsize; ++isamp) {
    //                     fftfile << tmpffta[isamp].x << " " << tmpffta[isamp].y << " " << tmpfftb[isamp].x << " " << tmpfftb[isamp].y << endl;
    //                 }*/
    
    //                 fftfilea.write(reinterpret_cast<char*>(tmpffta), fftedsize * sizeof(cufftComplex));
    //                 fftfileb.write(reinterpret_cast<char*>(tmpfftb), fftedsize * sizeof(cufftComplex));
    
    //                 interend = std::chrono::steady_clock::now();
    //                 runtimes.intertime += std::chrono::duration<float>(interend - interstart).count();           
    
    //             }FilHead filhead;
    //     ReadFilterbankHeader(config, filhead);
        
    //     if (scaling) {
    //         filhead.nbits = 8;
    //     }
    
    //     // TODO: This will be wrong for R2C FFT
    //     filhead.tsamp = 1.0 / (2.0 * filhead.foff) * 2 * FFTUSE * TIMEAVG;
    //     // TODO: Make sure it is the middle of the top frequency channel
    //     filhead.fch1 = (filhead.fch1 + filhead.foff / 2.0f) * 1e-06;
    //     filhead.nchans = FFTUSE;
    //     filhead.foff = -1.0 * filhead.foff / FFTUSE * 1e-06 ;
    
    //     filhead.fch1 = filhead.fch1 + filhead.foff / 2.0;
    
    //     if (DEBUG) {
    //         cout << "Some header info:\n"
    //                 << "Raw file: " << filhead.rawfile << endl
    //                 << "Source name: " << filhead.source << endl
    //                 << "Azimuth: " << filhead.az << endl
    //                 << "Zenith angle: " << filhead.za << endl
    //                 << "Declination: " << filhead.dec << endl
    //                 << "Right ascension: " << filhead.ra << endl
    //                 << "Top channel frequency: " << filhead.fch1 << endl
    //                 << "Channel bandwidth: " << filhead.foff << endl
    //                 << "Number of channels: " << filhead.nchans << endl
    //                 << "Sampling time: " << filhead.tsamp << endl
    //                 << "Bits per sample: " << filhead.nbits << endl;
    //     }
    
    //     // TODO: Make sure there are correct values for bandwidth and sampling time in the header after taking averaging into account
    
    //     ifstream filepola(inpola.c_str(), ifstream::in | ifstream::binary);
    //     ifstream filepolb(inpolb.c_str(), ifstream::in | ifstream::binary);
    //     ofstream filfile(outfil.c_str(), ofstream::out | ofstream::binary);
    
    //     if (!filepola || !filepolb) {
    //     if (!filepola) {
    //             cout << "Could not open file " << inpola << endl;
    //         }
    //         if (!filepolb) {
    //             cout << "Could not open file " << inpolb << endl;
    //         }
    //         exit(EXIT_FAILURE);
    //     }
    //     // TODO: Can save the filterbank header straight away, after the first header is read
    //     unsigned char vdifheadpola[32];
    //     unsigned char vdifheadpolb[32];
    //     filepola.read(reinterpret_cast<char*>(vdifheadpola), 32);
    //     filepolb.read(reinterpret_cast<char*>(vdifheadpolb), 32);
    
    //     filepola.seekg(0, filepola.end);
    //     long long filelengtha = filepola.tellg();
    //     filepola.seekg(0, filepola.beg);
    
    //     filepolb.seekg(0, filepolb.end);
    //     long long filelengthb = filepolb.tellg();
    //     filepolb.seekg(0, filepolb.beg);
    
    //     unsigned int startframe;
    //     unsigned int startsecond;
    
    //     cout << filelengtha << " " << filelengthb << endl;
    
    //     startframe = (unsigned int)(vdifheadpola[4] | (vdifheadpola[5] << 8) | (vdifheadpola[6] << 16));	// frame number in this second
    //     startsecond = (unsigned int)(vdifheadpola[0] | (vdifheadpola[1] << 8) | (vdifheadpola[2] << 16) | ((vdifheadpola[3] & 0x3f) << 24));
    
    //     if (DEBUG) {
    //         cout << "Starting time: " << startsecond << ":" << startframe << endl;
    //     }
    
    //     // NOTE: Need to read headers in
    //     unsigned int toread = NACCUMULATE * 8000;
    //     // NOTE: No more headers after unpacking
    //     unsigned int unpackedsize = NACCUMULATE * VDIFSIZE * UNPACKFACTOR;
    //     unsigned int fftedsize = unpackedsize / (2 * FFTUSE) * FFTOUT;
    //     unsigned int powersize = unpackedsize / (2 * FFTUSE) * FFTUSE / TIMEAVG;
    
    //     cufftHandle fftplan;
    //     int fftsizes[1];
    //     fftsizes[0] = 2 * FFTUSE;
    //     int fftbatchsize = unpackedsize / fftsizes[0];
    //     cout << fftbatchsize << endl;
    //     cufftCheckError(cufftPlanMany(&fftplan, 1, fftsizes, NULL, 1, FFTUSE, NULL, 1, FFTUSE, CUFFT_R2C, fftbatchsize));
    
    //     unsigned char *tmppola = new unsigned char[toread];
    //     unsigned char *tmppolb = new unsigned char[toread];
    
    //     unsigned char *devpola;
    //     unsigned char *devpolb;
    //     unsigned char **datapol = new unsigned char*[NPOL];
    //     unsigned char **devpol;
    //     float **unpacked = new float*[NPOL];
    //     float **devunpacked;
    //     cufftComplex **ffted = new cufftComplex*[NPOL];
    //     cufftComplex **devffted;
        
    //     unsigned char *devpower;
    //     unsigned char *tmppower = new unsigned char[powersize * filhead.nbits / 8];
    
    //     if (GPURUN) {
    //         cudaCheckError(cudaMalloc((void**)&devpola, toread * sizeof(unsigned char)));
    //         cudaCheckError(cudaMalloc((void**)&devpolb, toread * sizeof(unsigned char)));
    
    //         cudaCheckError(cudaMalloc((void**)&devpol, NPOL * sizeof(unsigned char*)));
    //         cudaCheckError(cudaMalloc((void**)&datapol[0], toread * sizeof(unsigned char)));
    //         cudaCheckError(cudaMalloc((void**)&datapol[1], toread * sizeof(unsigned char)));
    //         cudaCheckError(cudaMemcpy(devpol, datapol, NPOL * sizeof(unsigned char*), cudaMemcpyHostToDevice));
    
    //         cudaCheckError(cudaMalloc((void**)&devunpacked, NPOL * sizeof(float*)));
    //         cudaCheckError(cudaMalloc((void**)&unpacked[0], unpackedsize * sizeof(float)));
    //         cudaCheckError(cudaMalloc((void**)&unpacked[1], unpackedsize * sizeof(float)));
    //         cudaCheckError(cudaMemcpy(devunpacked, unpacked, NPOL * sizeof(float*), cudaMemcpyHostToDevice));
    
    //         cudaCheckError(cudaMalloc((void**)&devffted, NPOL * sizeof(cufftComplex*)));
    //         cudaCheckError(cudaMalloc((void**)&ffted[0], fftedsize * sizeof(cufftComplex)));
    //         cudaCheckError(cudaMalloc((void**)&ffted[1], fftedsize * sizeof(cufftComplex)));
    //         cudaCheckError(cudaMemcpy(devffted, ffted, NPOL * sizeof(cufftComplex*), cudaMemcpyHostToDevice));
    
    //         cudaCheckError(cudaMalloc((void**)&devpower, powersize * (filhead.nbits / 8)));
    //     }
    
    //     vector<std::pair<FrameInfo, FrameInfo>> vdifframes;
    
    //     FrameInfo tmpframea, tmpframeb;
    //     int refsecond;
    //     int frameno;
    //     int epoch;
    
    //     WriteFilterbankHeader(filfile, filhead);
       
    //     Timing runtimes;
    //     runtimes.readtime = 0.0f;
    //     runtimes.scaletime = 0.0f;
    //     runtimes.filtime = 0.0f;
    //     runtimes.savetime = 0.0f;
    //     runtimes.totaltime = 0.0f;
    //     runtimes.intertime = 0.0f;
    
    //     std::chrono::time_point<std::chrono::steady_clock> readstart, readend, scalestart, scaleend, filstart, filend, savestart, saveend, interstart, interend;
    
    //     float *tmpunpackeda = new float[unpackedsize];
    //     float *tmpunpackedb = new float[unpackedsize];
    //     cufftComplex *tmpffta = new cufftComplex[fftedsize];
    //     cufftComplex *tmpfftb = new cufftComplex[fftedsize];
    
    //     bool saved = false;
    
    //     //float *dmeans;
    //     //float *dstdevs;
    //     //cudaCheckError(cudaMalloc((void**)&dmeans, FFTUSE * sizeof(float)));
    //     //cudaCheckError(cudaMalloc((void**)&dstdevs, FFTUSE * sizeof(float)));
    
    //     thrust::device_vector<float> dmeans, dstdevs;
    //     dmeans.resize(FFTUSE);
    //     dstdevs.resize(FFTUSE);
    //     thrust::fill(dmeans.begin(), dmeans.end(), 0.0f);
    //     thrust::fill(dstdevs.begin(), dstdevs.end(), 0.0f);
    //     float *pdmeans = thrust::raw_pointer_cast(dmeans.data());
    //     float *pdstdevs = thrust::raw_pointer_cast(dstdevs.data());    
    
    //     cout << "Size of the device vectors: " << dmeans.size() << " " << dstdevs.size() << endl;
    
    //     scalestart = std::chrono::steady_clock::now();
    
    //     // NOTE: Use first 5 accumulates of data to obtain scaling factors
    //     if (scaling) {
    
    //         size_t divfactors = 5 * powersize / FFTUSE;
    //         thrust::device_vector<float> dfactors; 
    //         dfactors.resize(divfactors + 1);
    //         thrust::sequence(dfactors.begin(), dfactors.end());
    //         thrust::transform(dfactors.begin(), dfactors.end(), dfactors.begin(), FactorFunctor());
    //         float *pdfactors = thrust::raw_pointer_cast(dfactors.data());
    
    //         //float *dfactors;
    //         //size_t divfactors = 5 * powersize / FFTUSE;
    //         //cudaCheckError(cudaMalloc((void**)&dfactors, divfactors * sizeof(float)));
    //         //int scalethreads = 1024;
    //         //int scaleblocks = (divfactors - 1) / scalethreads + 1;
    //         //cout << "Div factors blocks: " << scaleblocks << " and threads: " << scalethreads << endl;
    //         //InitDivFactors<<<scaleblocks, scalethreads>>>(dfactors, divfactors);
    //         //cudaCheckError(cudaDeviceSynchronize());
    //         //cudaCheckError(cudaGetLastError());
    //         size_t processed = 0;
    
    //         float *tmpdpower;
    //         cudaCheckError(cudaMalloc((void**)&tmpdpower, powersize * sizeof(float)));
    
    //     while((filepola.tellg() < (5 * NACCUMULATE * 8032)) && (filepolb.tellg() < (5 * NACCUMULATE * 8032))) {
    //             for (int iacc = 0; iacc < NACCUMULATE; ++iacc) {
    //                 filepola.read(reinterpret_cast<char*>(vdifheadpola), 32);
    //                 filepolb.read(reinterpret_cast<char*>(vdifheadpolb), 32);
    //                 filepola.read(reinterpret_cast<char*>(tmppola) + iacc * 8000, 8000);
    //                 filepolb.read(reinterpret_cast<char*>(tmppolb) + iacc * 8000, 8000);
    //             }
    
    //             cudaCheckError(cudaMemcpy(datapol[0], tmppola, NACCUMULATE * 8000 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    //             cudaCheckError(cudaMemcpy(datapol[1], tmppolb, NACCUMULATE * 8000 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    //             UnpackKernel<<<50, 1024, 0, 0>>>(devpol, devunpacked, toread);
    //             for (int ipol = 0; ipol < NPOL; ++ipol) {
    //                 cufftCheckError(cufftExecR2C(fftplan, unpacked[ipol], ffted[ipol]));
    //             }
    //             DetectKernel<<<25, FFTUSE, 0, 0>>>(devffted, tmpdpower);
    //             cudaCheckError(cudaDeviceSynchronize());
    //             GetScalingFactorsKernel<<<1, FFTUSE, 0, 0>>>(tmpdpower, pdmeans, pdstdevs, pdfactors, processed);
    //             processed += (powersize / FFTUSE);
    //             cudaCheckError(cudaDeviceSynchronize());
    //         }
    
    //         //float *hmeans = new float[FFTUSE];
    //         //float *hstdevs = new float[FFTUSE];
    
    //         //cudaCheckError(cudaMemcpy(hmeans, dmeans, FFTUSE * sizeof(float), cudaMemcpyDeviceToHost));
    //         //cudaCheckError(cudaMemcpy(hstdevs, dstdevs, FFTUSE * sizeof(float), cudaMemcpyDeviceToHost));
    
    //         thrust::host_vector<float> hmeans = dmeans;
    //         thrust::host_vector<float> hstdevs = dstdevs;
    
    //         std::ofstream statsfile("mean_stdev.dat");
    
    //         cout << "Size of host vector:" << hmeans.size() << endl;
     
    //         if (statsfile) {
    //             for (int ichan = 0; ichan < hmeans.size(); ++ichan) {
    //                 statsfile << hmeans[ichan] << " " << hstdevs[ichan] << endl;
    //             }
    //         } else {
    //             cerr << "Could not open the stats file" << endl;
    //         }
    
    //         statsfile.close();
    
    //         cudaFree(tmpdpower);
             
    //     }
    
    //     scaleend = std::chrono::steady_clock::now();
    
    //     runtimes.scaletime = std::chrono::duration<float>(scaleend - scalestart).count();
    
    //     filepola.seekg(0, filepola.beg);
    //     filepolb.seekg(0, filepolb.beg);
    
    //     std::ofstream unpackedfilea ((outfil + ".unp0").c_str(), std::ios_base::binary);
    //     std::ofstream unpackedfileb ((outfil + ".unp1").c_str(), std::ios_base::binary);
    //     std::ofstream fftfilea ((outfil + ".fft0").c_str(), std::ios_base::binary);
    //     std::ofstream fftfileb ((outfil + ".fft1").c_str(), std::ios_base::binary);
    
    //     while((filepola.tellg() < (filelengtha - NACCUMULATE * 8000)) && (filepolb.tellg() < (filelengthb - NACCUMULATE * 8000))) {
    //         //cout << filepola.tellg() << endl;
    //         // NOTE: This implementation
    //         for (int iacc = 0; iacc < NACCUMULATE; ++iacc) {
    //         readstart = std::chrono::steady_clock::now();
    //             filepola.read(reinterpret_cast<char*>(vdifheadpola), 32);
    //             filepolb.read(reinterpret_cast<char*>(vdifheadpolb), 32);
    //             filepola.read(reinterpret_cast<char*>(tmppola) + iacc * 8000, 8000);
    //             filepolb.read(reinterpret_cast<char*>(tmppolb) + iacc * 8000, 8000);
    //             readend = std::chrono::steady_clock::now();
    //             runtimes.readtime += std::chrono::duration<float>(readend - readstart).count();
    
    //             refsecond = (unsigned int)(vdifheadpola[0] | (vdifheadpola[1] << 8) | (vdifheadpola[2] << 16) | ((vdifheadpola[3] & 0x3f) << 24));
    //             frameno = (unsigned int)(vdifheadpola[4] | (vdifheadpola[5] << 8) | (vdifheadpola[6] << 16));
    //             epoch = (unsigned int)(vdifheadpola[7] & 0x3f);
    // //            frameno += (refsecond - startsecond) * 4000;
    
    //             tmpframea.frameno = frameno;
    //             tmpframea.refsecond = refsecond;
    //             tmpframea.refepoch = epoch;
    
    //             refsecond = (unsigned int)(vdifheadpolb[0] | (vdifheadpolb[1] << 8) | (vdifheadpolb[2] << 16) | ((vdifheadpolb[3] & 0x3f) << 24));
    //             frameno = (unsigned int)(vdifheadpolb[4] | (vdifheadpolb[5] << 8) | (vdifheadpolb[6] << 16));
    //             epoch = (unsigned int)(vdifheadpolb[7] & 0x3f);
    // //            frameno += (refsecond - startsecond) * 4000;
    
    //             tmpframeb.frameno = frameno;
    //             tmpframeb.refsecond = refsecond;
    //             tmpframeb.refepoch = epoch;
    
    //             vdifframes.push_back(std::make_pair(tmpframea, tmpframeb));
    
    //             // NOTE: Can use subtract startframe to put frame count at 0 and use that to save into the buffer
    
    //         }
     
    //        if (GPURUN) {
    //             filstart = std::chrono::steady_clock::now();
    //             cudaCheckError(cudaMemcpy(datapol[0], tmppola, NACCUMULATE * 8000 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    //             cudaCheckError(cudaMemcpy(datapol[1], tmppolb, NACCUMULATE * 8000 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    //             UnpackKernel<<<50, 1024, 0, 0>>>(devpol, devunpacked, toread);
    //             for (int ipol = 0; ipol < NPOL; ++ipol) {
    //                 cufftCheckError(cufftExecR2C(fftplan, unpacked[ipol], ffted[ipol]));
    //             }
    
    //             if (filhead.nbits == 8) {
    //                 DetectScaleKernel<<<25, FFTUSE, 0, 0>>>(devffted, reinterpret_cast<unsigned char*>(devpower), pdmeans, pdstdevs);
    //             } else if (filhead.nbits == 32) {
    //                 DetectKernel<<<25, FFTUSE, 0, 0>>>(devffted, reinterpret_cast<float*>(devpower));
    //             } else {
    //                 cerr << "Unsupported option! Will use float!" << endl;
    //                 DetectKernel<<<25, FFTUSE, 0, 0>>>(devffted, reinterpret_cast<float*>(devpower));
    //             }
    
    //             //PowerKernel<<<25, FFTUSE, 0, 0>>>(devffted, devpower);
    //             cudaCheckError(cudaDeviceSynchronize());
    //             cudaCheckError(cudaMemcpy(tmppower, devpower, powersize * filhead.nbits / 8, cudaMemcpyDeviceToHost));
                
    //             if (!saved) {
    //                 std::ofstream unpackedfile("unpacked.dat");
    //         cudaCheckError(cudaMemcpy(tmpunpackeda, unpacked[0], 2 * 8000 * 4 * sizeof(float), cudaMemcpyDeviceToHost));
    //         cudaCheckError(cudaMemcpy(tmpunpackedb, unpacked[1], 2 * 8000 * 4 * sizeof(float), cudaMemcpyDeviceToHost));
    //         for (int isamp = 0; isamp < 2 * 8000 * 4; ++isamp) {
    //                     unpackedfile << tmpunpackeda[isamp] << " " << tmpunpackedb[isamp] << endl;
    //                 }
    //                 unpackedfile.close();
    //                 saved = true;
    //             }
    
    //             filend = std::chrono::steady_clock::now();
    //             runtimes.filtime += std::chrono::duration<float>(filend - filstart).count();
                
    //             savestart = std::chrono::steady_clock::now(); 
    //             filfile.write(reinterpret_cast<char*>(tmppower), powersize * filhead.nbits / 8);
    //             saveend = std::chrono::steady_clock::now();
    //             runtimes.savetime += std::chrono::duration<float>(saveend - savestart).count();
    
       
    
    //             if (saveinter) {
    
    //                 interstart = std::chrono::steady_clock::now();
    
    //                 cudaCheckError(cudaMemcpy(tmpunpackeda, unpacked[0], unpackedsize * sizeof(float), cudaMemcpyDeviceToHost));
    //         cudaCheckError(cudaMemcpy(tmpunpackedb, unpacked[1], unpackedsize * sizeof(float), cudaMemcpyDeviceToHost));
    //                 /*for (int isamp = 0; isamp < unpackedsize; ++isamp) {
    //                     unpackedfile << tmpunpackeda[isamp] << " " << tmpunpackedb[isamp] << endl;
    //                 }*/
    
    //                 unpackedfilea.write(reinterpret_cast<char*>(tmpunpackeda), unpackedsize * sizeof(float));
    //                 unpackedfileb.write(reinterpret_cast<char*>(tmpunpackedb), unpackedsize * sizeof(float));
    
    //                 cudaCheckError(cudaMemcpy(tmpffta, ffted[0], fftedsize * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    //                 cudaCheckError(cudaMemcpy(tmpfftb, ffted[1], fftedsize * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    //                 /*for (int isamp = 0; isamp < fftedsize; ++isamp) {
    //                     fftfile << tmpffta[isamp].x << " " << tmpffta[isamp].y << " " << tmpfftb[isamp].x << " " << tmpfftb[isamp].y << endl;
    //                 }*/
    
    //                 fftfilea.write(reinterpret_cast<char*>(tmpffta), fftedsize * sizeof(cufftComplex));
    //                 fftfileb.write(reinterpret_cast<char*>(tmpfftb), fftedsize * sizeof(cufftComplex));
    
    //                 interend = std::chrono::steady_clock::now();
    //                 runtimes.intertime += std::chrono::duration<float>(interend - interstart).count();           
    
    //             }
    
    //         }
    //         cout << "Completed " << std::fixed << std::setprecision(2) << (float)filepola.tellg() / (float)(filelengtha - 1.0) * 100.0f << "%\r";
    //         cout.flush();
    //     }
    
    //     cout << endl;
    //     filfile.close();
    //     unpackedfilea.close();
    //     unpackedfileb.close();
    //     fftfilea.close();
    //     fftfileb.close();
    
    //     runtimes.totaltime = runtimes.readtime + runtimes.scaletime + runtimes.filtime + runtimes.savetime + runtimes.intertime;
    
    //     cout << "Total execution time: " << runtimes.totaltime << "s\n";
    //     cout << "\tScaling factors: " << runtimes.scaletime << "s\n";
    //     cout << "\tFile read: " << runtimes.readtime << "s\n";
    //     cout << "\tFilterbanking: " << runtimes.filtime << "s\n";
    //     cout << "\tFile write: " << runtimes.savetime << "s\n";
    //     if (saveinter) {
    //         cout << "\tIntermediate write: " << runtimes.intertime << "s\n";
    //     }
    
    //     if (DEBUG) {
    //         std::ofstream outframes("dataframes.dat");
    
    //         outframes << "Ref Epoch A\tRef second A\tRef frame A\tRef Epoch B\tRef second B\tRef frame b\n";
    //         for (auto iframe = vdifframes.begin(); iframe != vdifframes.end(); ++iframe) {
    //             outframes << iframe->first.refepoch << "\t" << iframe->first.refsecond << "\t" << iframe->first.frameno << "\t"
    //             << iframe->second.refepoch << "\t" << iframe->second.refsecond << "\t" << iframe->second.frameno << endl;
    //         }
    
    //         outframes.close();
    //     }


    // }

    return 0;
}
