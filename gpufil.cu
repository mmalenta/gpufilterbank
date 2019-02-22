#include <algorithm>
#include <chrono>
#include <cmath>
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

#include "constants.hpp"
#include "dada.hpp"
#include "errors.hpp"
#include "filterbank.hpp"
#include "kernels.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;

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

int main(int argc, char *argv[]) {

    string inpola = "";
    string inpolb = "";
    string outfil = "";
    string config = "";
    string dadastr = "";
    double readsec; 
    bool scaling = false;
    bool saveinter = false;

    std::vector<std::string> dadastrings; 

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
            for (int ifile = 0; ifile < 4; ++ifile) {
                iarg++;
                dadastr = std::string(argv[iarg]);
                dadastrings.push_back(dadastr);
            }
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

    if (!dadastrings.empty()) {

        std::cout << "Input files: ";
            for (auto &dadastring: dadastrings) {
                std::cout << dadastring << " ";
            }
        std::cout << std::endl;
        
        long long filesize = 0;

        for (auto &dadastring: dadastrings) {
            std::ifstream indada(dadastring.c_str(), std::ios_base::binary);

            if (indada) {

                indada.seekg(0, indada.end);
                if (!filesize) {
                    filesize = indada.tellg() - 4096L;
                }
                
                if (filesize != indada.tellg() - 4096L) {
                    std::cerr << "Files do not have the same size!" << std::endl;
                    exit(EXIT_FAILURE);
                }
                indada.close();

            } else {
                std::cerr << "Could not open file: " << dadastring << std::endl;
                exit(EXIT_FAILURE);
            }

        }

        /* std::ifstream indada(dadastr.c_str(), std::ios_base::binary);        
        long long filesize = 0;
        indada.seekg(0, indada.end);
        filesize = indada.tellg() - 4096L;
        indada.seekg(0, indada.beg);
        */ 
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

        size_t freemem = 0;
        size_t totalmem = 0;
        cudaCheckError(cudaMemGetInfo(&freemem, &totalmem));
        // NOTE: Let's liffh just 25% of what's free, because cuFFT happens...
        freemem = freemem * 0.25;
        std::cout << "Total memory: " << totalmem / 1024.0f / 1024.0f << "MiB, with " << freemem / 1024.0f / 1024.0f << "MiB free" << std::endl;
        
        // original file + original file cast to cufftComplex for FFT + output filterbank file saved as 32 bit float, all times the number of input files
        size_t needmem = (4 * totalsamples + 4 * totalsamples * 4 + totalsamples / OUTCHANS / TIMEAVG * OUTCHANS * 4) * dadastrings.size();
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

            sampperblock = (int)((float)freemem / (dadastrings.size() * (float)(OUTCHANS * TIMEAVG) * (4.0f + 16.0f + 4.0f / (float)TIMEAVG))) * OUTCHANS * TIMEAVG;
            nblocks = (int)(totalsamples / sampperblock);
            remsamp = totalsamples - nblocks * sampperblock;

            std::cout << "Will process the data in " << nblocks << " blocks, with "
                        << sampperblock << " samples per block "
                        << "(" << dadastrings.size() << " files per block)";
            if (remsamp) {
                std::cout << " and an extra block with " << remsamp << " samples at the end";
            }
            std::cout << std::endl;
        }

        /**** ####
        // STAGE: MEMORY AND FFT
        #### ****/
        // NOTE: Factor of 4 to account for 2 polarisations and complex components for every time sample
        size_t blockread = sampperblock * 4 * dadastrings.size();
        size_t perfileread = sampperblock * 4;
        size_t remread = remsamp * 4 * dadastrings.size();
        size_t perfilerem = remsamp * 4;
        
        // NOTE: This is a very annoying stage where cufftPlanMany uses ridiculous amount of temporary buffer and runs out of memory most of the time
        cufftHandle fftplan;
        int fftsizes[1];
        fftsizes[0] = OUTCHANS;
        // NOTE: Factor of 2 to account for 2 polarisations
        int fftbatchsize = sampperblock * 2 / fftsizes[0] * dadastrings.size();
        cufftCheckError(cufftPlanMany(&fftplan, 1, fftsizes, NULL, 1, OUTCHANS, NULL, 1, OUTCHANS, CUFFT_C2C, fftbatchsize));
 
        unsigned char *hostvoltage = new unsigned char[blockread];
        unsigned char *devicevoltage = new unsigned char[blockread];
        cudaCheckError(cudaMalloc((void**)&devicevoltage, blockread * sizeof(unsigned char)));

        cufftComplex *devicefft;
        cudaCheckError(cudaMalloc((void**)&devicefft, sampperblock * 2 * sizeof(cufftComplex)));

        size_t powersize = sampperblock / OUTCHANS * OUTCHANS / TIMEAVG * dadastrings.size();
        float *hostpower = new float[powersize];
        float *devicepower;
        cudaCheckError(cudaMalloc((void**)&devicepower, powersize * sizeof(float)))

        float *hostband = new float[OUTCHANS * dadastrings.size()];
        float *deviceband;
        cudaCheckError(cudaMalloc((void**)&deviceband, OUTCHANS * dadastrings.size() * sizeof(float)));
        

        size_t fullfillsize = nblocks * powersize * dadastrings.size() + remsamp / OUTCHANS / TIMEAVG * OUTCHANS * dadastrings.size();
        float *fullfil = new float[fullfillsize];

        std::vector<FilHead> filheaders;
        std::vector<std::ifstream> dadastreams;

        for (int ifile = 0; ifile < dadastrings.size(); ++ifile) {

            dadastreams.push_back(std::ifstream());

            dadastreams.back().open(dadastrings.at(ifile).c_str(), std::ios_base::binary);

            // std::ifstream indada(dadastrings.at(ifile).c_str(), std::ios_base::binary);
            
            FilHead filhead = {};
            ReadDadaHeader(dadastreams.back(), filhead);

            if (!scaling) {
                filhead.nbits = 32;
            }
            filhead.nchans = OUTCHANS;
            filhead.tsamp = filhead.tsamp * OUTCHANS * TIMEAVG;
        
            filheaders.push_back(filhead);
        
            PrintFilterbankHeader(filheaders.at(ifile));
            
            // NOTE: Just in case I did something wrong
            dadastreams.back().seekg(4096, dadastreams.back().beg);

        }

        std::ofstream filfile(outfil.c_str(), std::ios_base::binary);
        //WriteFilterbankHeader(filfile, filhead);


        // NOTE: So this approach has an extra complication that it does the baseband calculation and filtering over the whole input file
        // NOTE: This is not ideal as it requires multiple copies between host and devices, but avoids the situation when the input data is divided into too many blocks
        for (int iblock = 0; iblock < nblocks; iblock++) {

            std::cout << "Processing block " << iblock << "..." << std::endl;

            for (int ifile = 0; ifile < dadastrings.size(); ++ifile) {
                std::cout << "Reading file " << dadastrings.at(ifile) << "..." << std::endl;
               
                dadastreams.at(ifile).read(reinterpret_cast<char*>(hostvoltage + ifile * perfileread), perfileread * sizeof(unsigned char));
            }

            cudaCheckError(cudaMemcpy(devicevoltage, hostvoltage, blockread * sizeof(unsigned char), cudaMemcpyHostToDevice));

            dim3 block (1024, 1, 1);
            dim3 grid (64, 1, 1);

            UnpackDadaKernel<<<grid, block, 0, 0>>>(sampperblock * dadastrings.size(), reinterpret_cast<uchar4*>(devicevoltage), devicefft);
            cudaCheckError(cudaGetLastError());

            cufftCheckError(cufftExecC2C(fftplan, devicefft, devicefft, CUFFT_FORWARD));

            DetectDadaKernel<<<grid, block, 0, 0>>>(sampperblock / OUTCHANS, devicefft, devicepower, dadastrings.size());
            cudaCheckError(cudaGetLastError());

            BandpassKernel<<<dadastrings.size(), OUTCHANS, 0, 0>>>(sampperblock / OUTCHANS / TIMEAVG, devicepower, deviceband);
            cudaCheckError(cudaGetLastError());

            //cudaCheckError(cudaMemcpy(hostpower, devicepower, powersize * sizeof(float), cudaMemcpyDeviceToHost));

            //filfile.write(reinterpret_cast<char*>(hostpower), powersize * sizeof(float));

            cudaCheckError(cudaMemcpy(fullfil + powersize * dadastrings.size() * iblock, devicepower,
                                        powersize * dadastrings.size() * sizeof(float), cudaMemcpyDeviceToHost));
        } 
        
        cufftCheckError(cufftDestroy(fftplan));

        if (remsamp) {

            std::cout << "Processing the remainder block..." << std::endl;

            for (int ifile = 0; ifile < dadastrings.size(); ++ifile) {
                std::cout << "Reading file " << dadastrings.at(ifile) << "..." << std::endl;
               
                dadastreams.at(ifile).read(reinterpret_cast<char*>(hostvoltage + ifile * perfilerem), perfilerem * sizeof(unsigned char));
            }

            cudaCheckError(cudaMemcpy(devicevoltage, hostvoltage, remread * sizeof(unsigned char), cudaMemcpyHostToDevice));

            dim3 block(OUTCHANS, 1, 1);
            dim3 grid (64, 1, 1);

            UnpackDadaKernel<<<grid, block, 0, 0>>>(remsamp, reinterpret_cast<uchar4*>(devicevoltage), devicefft);
            cudaCheckError(cudaGetLastError());

            cufftHandle fftplanrem;
            int fftrembatchsize = remsamp * 2 / fftsizes[0];
            cufftCheckError(cufftPlanMany(&fftplanrem, 1, fftsizes, NULL, 1, OUTCHANS, NULL, 1, OUTCHANS, CUFFT_C2C, fftrembatchsize));

            cufftCheckError(cufftExecC2C(fftplanrem, devicefft, devicefft, CUFFT_FORWARD));

            DetectDadaKernel<<<grid, block, 0, 0>>>(remsamp / OUTCHANS, devicefft, devicepower, dadastrings.size());
            cudaCheckError(cudaGetLastError());

            BandpassKernel<<<1, OUTCHANS, 0, 0>>>(remsamp / OUTCHANS / TIMEAVG, devicepower, deviceband);
            cudaCheckError(cudaGetLastError());

            //cudaCheckError(cudaMemcpy(hostpower, devicepower, remsamp / OUTCHANS / TIMEAVG * OUTCHANS * sizeof(float), cudaMemcpyDeviceToHost));

            //filfile.write(reinterpret_cast<char*>(hostpower), remsamp / OUTCHANS / TIMEAVG * OUTCHANS * sizeof(float));

            cufftCheckError(cufftDestroy(fftplanrem));

            cudaCheckError(cudaMemcpy(fullfil + nblocks * powersize * dadastrings.size(), devicepower,
                                        remsamp / OUTCHANS / TIMEAVG * OUTCHANS * sizeof(float) * dadastrings.size(), cudaMemcpyDeviceToHost));
        }

        cudaCheckError(cudaMemcpy(hostband, deviceband, dadastrings.size() * OUTCHANS * sizeof(float), cudaMemcpyDeviceToHost));
        std::ofstream bandout("bandpass.dat");
        if (bandout) {
            for (int ichan = 0; ichan < dadastrings.size() * OUTCHANS; ++ichan) {
                bandout << hostband[ichan] << std::endl;
            }
        }
        bandout.close();

        // NOTE: Quick and dirty bandpass cleaning
        // NOTE: First adjust the levels between different bands

        float banddiff;

        for (int iband = 1; iband < dadastrings.size(); ++iband) {

            banddiff = hostband[OUTCHANS - 1] - hostband[iband * OUTCHANS];
            std::transform(hostband + iband * OUTCHANS, hostband + (iband + 1) * OUTCHANS,
                            hostband + iband * OUTCHANS,
                            [banddiff](float val) -> float { return val + banddiff; });            

        }

        // NOTE: And now get the running median of 32 channels
        // Saved in the 'middle of range' (rounded up)
        float *medianhostband = new float[OUTCHANS * dadastrings.size()];
        const int mediansize = 32;

        float currentmedian = 0.0f;

        for (int ichan = 16; ichan < OUTCHANS * dadastrings.size() - 16; ++ichan) {

            std::vector<float> subvector(hostband + ichan - 16, hostband + ichan + 16);
            std::sort(subvector.begin(), subvector.end());
            currentmedian = (subvector.at(16) + subvector.at(15)) / 2.0f;
            medianhostband[ichan] = currentmedian;

        }
      
        // NOTE: And now take care of leftover samples from the median at the start and end of the band
        // NOTE: Uses very simple linear interpolation - might move to something more sophisticated later, but this seems to do the job for now

        // NOTE: Start of the band
        for (int ichan = 15; ichan >= 0; --ichan) {
            medianhostband[ichan] = medianhostband[ichan + 1] + (medianhostband[ichan + 1] - medianhostband[ichan + 2]);
        }

        // NOTE: End of the band
        for (int ichan = OUTCHANS * dadastrings.size() - 16; ichan < OUTCHANS * dadastrings.size(); ++ichan) {
            medianhostband[ichan] = medianhostband[ichan - 1] + (medianhostband[ichan - 1] - medianhostband[ichan - 2]);
        }

        std::ofstream medianbandout("bandpass_median.dat");
        if (medianbandout) {
            for (int ichan = 0; ichan < dadastrings.size() * OUTCHANS; ++ichan) {
                medianbandout << medianhostband << std::endl;
            }
        }
        medianbandout.close();
        
        float *normalisedband = new float[OUTCHANS * dadastrings.size()];
        
        std::transform(hostband, hostband + OUTCHANS * dadastrings.size(), 
        medianhostband, normalisedband,
        [] (float band, float median) -> float { return band / median; });
        
        const int nrep = 4;
        int tmpcount = 0;
        const float threshold = 5.0;

        float normean = 0.0f;
        float normstd = 0.0f;

        float tmpmean = 0.0f;
        float tmpstd = 0.0f;

        for (int ichan = 0; ichan < dadastrings.size() * OUTCHANS; ++ichan) {
            float tmpdata = normalisedband[ichan];
            normean += tmpdata;
            normstd += tmpdata * tmpdata;
        }
        normstd = sqrtf(normstd / float(OUTCHANS * dadastrings.size()) - normean * normean);

        // NOTE: We estimate 'true' mean and standard deviation of normalised bandpass excluding outliers after every run
        for (int irep = 0 ; irep < nrep; ++irep) {

            for (int ichan = 0; ichan < dadastrings.size() * OUTCHANS; ++ichan) {
                if (normalisedband[ichan] <= (normean + threshold * normstd)) {
                    float tmpdata = normalisedband[ichan];
                    tmpmean += tmpdata;
                    tmpstd += tmpdata * tmpdata;
                    tmpcount++;
                }
            }

            tmpmean = tmpmean / (float)tmpcount;
            tmpstd = sqrt(tmpstd / (float)tmpcount - tmpmean * tmpmean);

            normean = tmpmean;
            normstd = tmpstd;

            tmpmean = 0.0f;
            tmpstd = 0.0f;
            tmpcount = 0;

        }

        /**** ####
        // STAGE: CLEANING THE DATA
        #### ****/
        // NOTE: Check which channels are offending and replace the original band channel with median
        std::vector<int> maskedchans;

        for (int ichan = 0; ichan < dadastrings.size() * OUTCHANS; ++ichan) {
            if (normalisedband[ichan] > (normean + threshold * normstd)) {
                maskedchans.push_back(ichan);
                hostband[ichan] = medianhostband[ichan];
            }
        }

        std::ofstream maskedout("masked_channels.dat");
        for (auto &chan: maskedchans) {
            maskedout << chan << std::endl;
        }
        maskedout.close();

        std::ofstream correctedout("corrected_band.dat");
        for (int ichan = 0; ichan < dadastrings.size() * OUTCHANS; ++ichan) {
            correctedout << hostband[ichan] << std::endl;
        }
        correctedout.close();        

        int* devicemask;
        cudaCheckError(cudaMalloc((void**)&devicemask, maskedchans.size() * sizeof(int)));

        

        // NOTE: Actual cleaning on the
        // NOTE: Need to scale the data somehow as well

        // NOTE: This will most likely be slow, very slow

        std::chrono::time_point<std::chrono::steady_clock> cleanstart, cleanend;
        cleanstart = std::chrono::steady_clock::now();
        for (auto &ichan: maskedchans) {

            for (size_t isamp = 0; isamp < totalsamples; ++isamp) {

                fullfil[isamp * OUTCHANS * dadastrings.size() + ichan] = hostband[ichan];

            }

        }
        cleanend = std::chrono::steady_clock::now();

        std::cout << "Took " << std::chrono::duration<double>(cleanend - cleanstart).count() << "s to clean the data..." << std::endl;

        filfile.write(reinterpret_cast<char*>(fullfil), totalsamples * OUTCHANS * dadastrings.size() * sizeof(float));

        /*
        for (int iblock = 0; iblock < nblocks; ++iblock) {

            cudaCheckError(cudaMemcpy(devicepower, fullfil + iblock * powersize * dadastrings.size(),
                                        powersize * dadastrings.size() * sizeof(float), cudaMemcpyHostToDevice));

            dim3 grid(1, 1, 1);
            dim3 block(1, 1, 1);

            MaskKernel<<<grid, block, 0, 0>>>(devicepower, devicemask, sampperblock / OUTCHANS / TIMEAVG, OUTCHANS * dadastrings.size());
            cudaCheckError(cudaGetLastError());            

            cudaCheckError(cudaMemcpy(fullfil + iblock * powersize * dadastrings.size(), devicepower,
                                        powersize * dadastrings.size() * sizeof(float), cudaMemcpyDeviceToHost));

        }

        if (remsamp) {

            cudaCheckError(cudaMemcpy(devicepower, fullfil + nblocks * powersize * dadastrings.size(),
                                        remsamp / OUTCHANS / TIMEAVG * OUTCHANS * dadastrings.size() * sizeof(float), cudaMemcpyHostToDevice));

            dim3 grid(1, 1, 1);
            dim3 block(1, 1, 1);

            MaskKernel<<<grid, block, 0, 0>>>(devicepower, devicemask, remsamp / OUTCHANS / TIMEAVG, OUTCHANS * dadastrings.size());
            cudaCheckError(cudaGetLastError());            

            cudaCheckError(cudaMemcpy(fullfil + nblocks * powersize * dadastrings.size(), devicepower,
                                        remsamp / OUTCHANS / TIMEAVG * OUTCHANS * dadastrings.size() * sizeof(float), cudaMemcpyDeviceToHost));

        }
        */
        cudaCheckError(cudaGetLastError());

        /**** ####
        // STAGE: Save the final filterbank file
        #### ****/



        /**** ####
        // STAGE: CLEANING UP
        #### ****/
        // NOTE: Factor of 4 to account for 2 polarisations and complex components for every time sample
        for (auto &dadastream: dadastreams) {
            dadastream.close();
        }

        filfile.close();

        cudaFree(devicemask);
        cudaFree(deviceband);
        cudaFree(devicepower);
        cudaFree(devicefft);
        cudaFree(devicevoltage);

        delete [] normalisedband;
        delete [] medianhostband;
        delete [] fullfil;
        delete [] hostband;
        delete [] hostpower;
        delete [] hostvoltage;

    } 

    return 0;

}
