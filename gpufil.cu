#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <numeric>
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

        const int fullchans = OUTCHANS * dadastrings.size();

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
        size_t voltagesamples = filesize / 4;
        if (filesize != voltagesamples * 4) {
            std::cerr << "A non-integer number of voltage samples was read - something went very wrong!" << std::endl;
            return 1;
        }

        std::cout << "Individual file size: " << filesize / 1024.0f / 1024.0f << "MiB with " << voltagesamples << " voltage samples" << std::endl;
        // NOTE: That simply ensures that we only process the integer number of final filterbank channels
        voltagesamples = (int)((float)voltagesamples / (OUTCHANS * TIMEAVG)) * OUTCHANS * TIMEAVG;
        std::cout << "Will use first" << voltagesamples << " samples" << std::endl;

        size_t freemem = 0;
        size_t totalmem = 0;
        cudaCheckError(cudaMemGetInfo(&freemem, &totalmem));
        // NOTE: Let's liffh just 25% of what's free, because cuFFT happens...
        freemem = freemem * 0.25;
        std::cout << "Total memory: " << totalmem / 1024.0f / 1024.0f << "MiB, with " << freemem / 1024.0f / 1024.0f << "MiB free" << std::endl;
        
        // original file + original file cast to cufftComplex for FFT + output filterbank file saved as 32 bit float, all times the number of input files
        size_t needmem = (4 * voltagesamples + 4 * voltagesamples * 4 + voltagesamples / OUTCHANS / TIMEAVG * OUTCHANS * 4) * dadastrings.size();
        std::cout << "Need " << needmem / 1024.0f / 1024.0f << "MiB on the device" << std::endl;
        
        int nblocks = 0;
        size_t voltagesamplesperblock = 0;
        size_t remvoltagesamples = 0;
        
        if (needmem < freemem) {
            std::cout << "Can store everything in global memory at once..." << std::endl;
            nblocks = 1;
            voltagesamplesperblock = voltagesamples;
        } else {
            std::cout << "We need to divide the job..." << std::endl;

            voltagesamplesperblock = (int)((float)freemem / (dadastrings.size() * (float)(OUTCHANS * TIMEAVG) * (4.0f + 16.0f + 4.0f / (float)TIMEAVG))) * OUTCHANS * TIMEAVG;
            nblocks = (int)(voltagesamples / voltagesamplesperblock);
            remvoltagesamples = voltagesamples - nblocks * voltagesamplesperblock;

            std::cout << "Will process the data in " << nblocks << " blocks, with "
                        << voltagesamplesperblock << " samples per block "
                        << "(" << dadastrings.size() << " files per block)";
            if (remvoltagesamples) {
                std::cout << " and an extra block with " << remvoltagesamples << " samples at the end";
            }
            std::cout << std::endl;
        }

        /**** ####
        // STAGE: MEMORY AND FFT
        #### ****/
        // NOTE: Factor of 4 to account for 2 polarisations and complex components for every time sample
        // NOTE: This is the amount of memory required to store whole block, with all DADA files included
        size_t blockread = voltagesamplesperblock * 4 * dadastrings.size();
        // NOTE: This is the amount of memory required to store only a single DADA file portion of the block
        size_t perfileread = voltagesamplesperblock * 4;
        size_t remread = remvoltagesamples * 4 * dadastrings.size();
        size_t perfilerem = remvoltagesamples * 4;
        
        size_t timesamplesperblockin = voltagesamplesperblock / OUTCHANS;
        size_t timesamplesperblockout = voltagesamplesperblock / OUTCHANS / TIMEAVG;

        size_t remtimesamplesin = remvoltagesamples / OUTCHANS;
        size_t remtimesamplesout = remvoltagesamples / OUTCHANS / TIMEAVG;

        // NOTE: This is a very annoying stage where cufftPlanMany uses ridiculous amount of temporary buffer and runs out of memory most of the time
        cufftHandle fftplan;
        int fftsizes[1];
        fftsizes[0] = OUTCHANS;
        // NOTE: Factor of 2 to account for 2 polarisations
        int fftbatchsize = (voltagesamplesperblock * 2 / fftsizes[0]) * dadastrings.size();
        cufftCheckError(cufftPlanMany(&fftplan, 1, fftsizes, NULL, 1, OUTCHANS, NULL, 1, OUTCHANS, CUFFT_C2C, fftbatchsize));
 
        unsigned char *hostvoltage = new unsigned char[blockread];
        unsigned char *devicevoltage = new unsigned char[blockread];
        cudaCheckError(cudaMalloc((void**)&devicevoltage, blockread * sizeof(unsigned char)));

        cufftComplex *devicefft;
        // NOTE: Factor of 2 to account for 2 polarisations
        cudaCheckError(cudaMalloc((void**)&devicefft, voltagesamplesperblock * 2 * dadastrings.size() * sizeof(cufftComplex)));

        // NOTE: That's detected, time averaged data per block, with all DADA files included
        size_t powersize = timesamplesperblockout * fullchans;
        float *hostpower = new float[powersize];
        float *devicepower;
        cudaCheckError(cudaMalloc((void**)&devicepower, powersize * sizeof(float)))

        float *hostband = new float[fullchans];
        float *deviceband;
        cudaCheckError(cudaMalloc((void**)&deviceband, fullchans * sizeof(float)));
        
        size_t fullfilsize = nblocks * powersize + remtimesamplesout * fullchans;
        float *fullfil = new float[fullfilsize];

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
            filhead.fch1 = filhead.fch1 + fabs(filhead.foff) / 2;
            filhead.foff = -1.0 * fabs(filhead.foff / OUTCHANS);

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

            UnpackDadaKernel<<<grid, block, 0, 0>>>(voltagesamplesperblock * dadastrings.size(), reinterpret_cast<uchar4*>(devicevoltage), devicefft);
            cudaDeviceSynchronize();
            cudaCheckError(cudaGetLastError());

            cufftCheckError(cufftExecC2C(fftplan, devicefft, devicefft, CUFFT_FORWARD));

            DetectDadaKernel<<<grid, block, 0, 0>>>(timesamplesperblockin, devicefft, devicepower, dadastrings.size());
            cudaDeviceSynchronize();
            cudaCheckError(cudaGetLastError());

            BandpassKernel<<<dadastrings.size(), OUTCHANS, 0, 0>>>(timesamplesperblockout, devicepower, deviceband);
            cudaDeviceSynchronize();
	        cudaCheckError(cudaGetLastError());

            //cudaCheckError(cudaMemcpy(hostpower, devicepower, powersize * sizeof(float), cudaMemcpyDeviceToHost));

            //filfile.write(reinterpret_cast<char*>(hostpower), powersize * sizeof(float));

            cudaCheckError(cudaMemcpy(fullfil + powersize * iblock, devicepower,
                                        powersize * sizeof(float), cudaMemcpyDeviceToHost));
        } 
        
        cufftCheckError(cufftDestroy(fftplan));

        if (remvoltagesamples) {

            std::cout << "Processing the remainder block..." << std::endl;

            for (int ifile = 0; ifile < dadastrings.size(); ++ifile) {
                std::cout << "Reading file " << dadastrings.at(ifile) << "..." << std::endl;
               
                dadastreams.at(ifile).read(reinterpret_cast<char*>(hostvoltage + ifile * perfilerem), perfilerem * sizeof(unsigned char));
            }

            cudaCheckError(cudaMemcpy(devicevoltage, hostvoltage, remread * sizeof(unsigned char), cudaMemcpyHostToDevice));

            dim3 block(OUTCHANS, 1, 1);
            dim3 grid (64, 1, 1);

            UnpackDadaKernel<<<grid, block, 0, 0>>>(remvoltagesamples * dadastrings.size(), reinterpret_cast<uchar4*>(devicevoltage), devicefft);
            cudaDeviceSynchronize();
            cudaCheckError(cudaGetLastError());

            cufftHandle fftplanrem;
            int fftrembatchsize = (remvoltagesamples * 2 / fftsizes[0]) * dadastrings.size();
            cufftCheckError(cufftPlanMany(&fftplanrem, 1, fftsizes, NULL, 1, OUTCHANS, NULL, 1, OUTCHANS, CUFFT_C2C, fftrembatchsize));

            cufftCheckError(cufftExecC2C(fftplanrem, devicefft, devicefft, CUFFT_FORWARD));

            DetectDadaKernel<<<grid, block, 0, 0>>>(remtimesamplesin, devicefft, devicepower, dadastrings.size());
            cudaDeviceSynchronize();
            cudaCheckError(cudaGetLastError());

            BandpassKernel<<<dadastrings.size(), OUTCHANS, 0, 0>>>(remtimesamplesout, devicepower, deviceband);
            cudaDeviceSynchronize();
            cudaCheckError(cudaGetLastError());

            //cudaCheckError(cudaMemcpy(hostpower, devicepower, remvoltagesamples / OUTCHANS / TIMEAVG * OUTCHANS * sizeof(float), cudaMemcpyDeviceToHost));

            //filfile.write(reinterpret_cast<char*>(hostpower), remvoltagesamples / OUTCHANS / TIMEAVG * OUTCHANS * sizeof(float));

            cufftCheckError(cufftDestroy(fftplanrem));

            cudaCheckError(cudaMemcpy(fullfil + nblocks * powersize, devicepower,
                                        remtimesamplesout * fullchans * sizeof(float), cudaMemcpyDeviceToHost));
        }

        cudaCheckError(cudaMemcpy(hostband, deviceband, fullchans * sizeof(float), cudaMemcpyDeviceToHost));
        std::ofstream bandout("bandpass.dat");
        if (bandout) {
            for (int ichan = 0; ichan < fullchans; ++ichan) {
                bandout << hostband[ichan] << std::endl;
            }
        }
        bandout.close();

        // NOTE: Quick and dirty bandpass cleaning
        // NOTE: First adjust the levels between different bands

        // NOTE: This cannot be done here - there's a possibility of massive RFI at the ends of bands
        
        // Saved in the 'middle of range' (rounded up)
        float *medianhostband = new float[OUTCHANS * dadastrings.size()];
        int mediansize = 32;

        int bandskip = 0;

        for (int iband = 0; iband < dadastrings.size(); ++iband) {

            float currentmedian = 0.0f;
            bandskip = iband * OUTCHANS;

            for (int ichan = 16; ichan < OUTCHANS - 16; ++ichan) {
                
                std::vector<float> subvector(hostband + bandskip + ichan - 16, hostband + bandskip + ichan + 16);
                std::sort(subvector.begin(), subvector.end());
                currentmedian = (subvector.at(16) + subvector.at(15)) / 2.0f;
                medianhostband[ichan + iband * OUTCHANS] = currentmedian;
    
            }

        }
        
        // NOTE: Median closer to the band edges
        // NOTE: Here we run an 8-point running median - allows us to have only 4 points at the edges that need extrapolating
        mediansize = 8;
        
        for (int iband = 0; iband < dadastrings.size(); ++iband) {

            float currentmedian = 0.0f;
            bandskip = iband * OUTCHANS;

            for (int ichan = 4; ichan < 16; ++ichan) {
                std::vector<float> subvector(hostband + bandskip + ichan - mediansize / 2, hostband + bandskip + ichan + mediansize / 2);
                std::sort(subvector.begin(), subvector.end());
                currentmedian = (subvector.at(4) + subvector.at(3)) / 2.0f;
                medianhostband[ichan + iband * OUTCHANS] = currentmedian;
            }

            for (int ichan = OUTCHANS - 16; ichan < OUTCHANS - 4; ++ichan) {
                std::vector<float> subvector(hostband + bandskip + ichan - mediansize / 2, hostband + bandskip + ichan + mediansize / 2);
                std::sort(subvector.begin(), subvector.end());
                currentmedian = (subvector.at(4) + subvector.at(3)) / 2.0f;
                medianhostband[ichan + iband * OUTCHANS] = currentmedian;
            }

        }

        for (int iband = 0; iband < dadastrings.size(); ++iband) {
        
            bandskip = iband * OUTCHANS;
            // NOTE: Remove the horrible DC compoment artifacts
            hostband[bandskip + 512] = medianhostband[bandskip + 512];
            // NOTE: Start of the band
            for (int ichan = 3; ichan >= 0; --ichan) {
                medianhostband[ichan + bandskip] = medianhostband[ichan + bandskip + 1] + (medianhostband[ichan + bandskip + 1] - medianhostband[ichan + bandskip + 2]);
            }
            // NOTE: End of the band
            for (int ichan = OUTCHANS - 4; ichan < OUTCHANS; ++ichan) {
                medianhostband[ichan + bandskip] = medianhostband[ichan + bandskip - 1] + (medianhostband[ichan + bandskip - 1] - medianhostband[ichan + bandskip - 2]);
            }
        }
        
        std::ofstream medianbandout("bandpass_median.dat");
        if (medianbandout) {
            for (int ichan = 0; ichan < dadastrings.size() * OUTCHANS; ++ichan) {
                medianbandout << medianhostband[ichan] << std::endl;
            }
        }
        medianbandout.close();

        float banddiff;
        float fulldiff;
        std::vector<float> banddiffs;
        for (int iband = 1; iband < dadastrings.size(); ++iband) {

            banddiff = medianhostband[iband * OUTCHANS - 1] - medianhostband[iband * OUTCHANS];

            if (iband == 1) {
                banddiffs.push_back(banddiff);
            } else {
                banddiffs.push_back(banddiffs.back() + banddiff);
            }
            banddiff = banddiffs.back();

            std::transform(hostband + iband * OUTCHANS, hostband + (iband + 1) * OUTCHANS,
                            hostband + iband * OUTCHANS,
                            [banddiff](float val) -> float { return val + banddiff; });            

            std::transform(medianhostband + iband * OUTCHANS, medianhostband + (iband + 1) * OUTCHANS,
                            medianhostband + iband * OUTCHANS,
                            [banddiff](float val) -> float { return val + banddiff; });            

        }

        std::ofstream adjbandout("adjusted_band.dat");
        if (adjbandout) {
            for (int ichan = 0; ichan < fullchans; ++ichan) {
                adjbandout << hostband[ichan] << std::endl;
            }
        }
        adjbandout.close();

        // NOTE: And now get the running median of 32 channels
      
        // NOTE: And now take care of leftover samples from the median at the start and end of the band
        // NOTE: Uses very simple linear interpolation - might move to something more sophisticated later, but this seems to do the job for now
        
        float *normalisedband = new float[OUTCHANS * dadastrings.size()];
        
        std::transform(hostband, hostband + OUTCHANS * dadastrings.size(), 
        medianhostband, normalisedband,
        [] (float band, float median) -> float { return band / median; });
        
        std::ofstream normbandout("normalised_band.dat");
        if (normbandout) {
            for (int ichan = 0; ichan < fullchans; ++ichan) {
                normbandout << normalisedband[ichan] << std::endl;
            }
        }
        normbandout.close();

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
        normean = normean / float(OUTCHANS * dadastrings.size());
        normstd = sqrtf(normstd / float(OUTCHANS * dadastrings.size()) - normean * normean);

        std::ofstream statsout("band_stats.dat");

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

            if (statsout) {
                statsout << normean << " " << normstd << std::endl;
            }

            tmpmean = 0.0f;
            tmpstd = 0.0f;
            tmpcount = 0;

        }

        statsout.close();

        /**** ####
        // STAGE: CLEANING THE DATA
        #### ****/
        // NOTE: Check which channels are offending and replace the original band channel with median
        // TODO: Also need to adjust the levels of 3 bands - this has to be done on the GPU now
        std::vector<int> maskedchans;

        for (int ichan = 0; ichan < dadastrings.size() * OUTCHANS; ++ichan) {
            if (normalisedband[ichan] > (normean + threshold * normstd)) {
                maskedchans.push_back(ichan);
                hostband[ichan] = medianhostband[ichan];
            }
        }

        // NOTE: Add DC masking
        for (int iband = 0; iband < dadastrings.size(); ++iband) {
            maskedchans.push_back(512 + iband * 1024);
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

        // NOTE: Actual cleaning on the
        // NOTE: Need to scale the data somehow as well
        // NOTE: This will most likely be slow, very slow

        std::chrono::time_point<std::chrono::steady_clock> cleanstart, cleanend;

        size_t fulltimesamples = fullfilsize / fullchans;
        for (auto &idiff: banddiffs) {
            idiff /= (float)fulltimesamples;
        }

        cleanstart = std::chrono::steady_clock::now();
        for (auto &ichan: maskedchans) {

            std::cout << "Masking channel " << ichan << "..." << std::endl;

            for (size_t isamp = 0; isamp < fulltimesamples; ++isamp) {

                fullfil[isamp * fullchans + ichan] = hostband[ichan] / (float)fulltimesamples;

            }

        }
        cleanend = std::chrono::steady_clock::now();

        std::cout << "Took " << std::chrono::duration<double>(cleanend - cleanstart).count() << "s to clean the data..." << std::endl;
        //std::cout << "Will write " << fullfilsize * sizeof(float) / 1024.0f / 1024.0f
		//	<< "MiB to the disk" << std::endl;
        //filfile.write(reinterpret_cast<char*>(fullfil), fullfilsize * sizeof(float));
        // NOTE: Need to do stuff on the GPU anyway - need to adjust the band in the original data anyway
        // NOTE: Need to ajdust if more than one band only 
        // NOTE: Some of this code is an abomination - move to Thrust
        if (dadastrings.size() > 1) {
            
            float *devicediffs;
            cudaCheckError(cudaMalloc((void**)&devicediffs, (dadastrings.size() - 1) * sizeof(float)));
            cudaCheckError(cudaMemcpy(devicediffs, banddiffs.data(), (dadastrings.size() - 1) * sizeof(float), cudaMemcpyHostToDevice));

            for (int iblock = 0; iblock < nblocks; ++iblock) {
                
                std::cout << "Adjusting bands in block " << iblock << " out of " << nblocks << "..." << std::endl;
                
                // TODO: Think about copying and processing bands excluding the first one
                // NOTE: Thats should save 25% of resources
                cudaCheckError(cudaMemcpy(devicepower, fullfil + iblock * powersize,
                                            powersize * sizeof(float), cudaMemcpyHostToDevice));
                
                dim3 block(OUTCHANS, 1, 1);
                dim3 grid(64, 1, 1);
                
                AdjustKernel<<<grid, block, 0, 0>>>(devicepower, devicediffs, dadastrings.size(), timesamplesperblockout);
                cudaDeviceSynchronize();                
                cudaCheckError(cudaGetLastError());            
                
                cudaCheckError(cudaMemcpy(fullfil + iblock * powersize, devicepower,
                                            powersize * sizeof(float), cudaMemcpyDeviceToHost));
                
            }
            
            if (remvoltagesamples) {
                
                std::cout << "Adjusting bands in the remainder block..." << std::endl;
                
                cudaCheckError(cudaMemcpy(devicepower, fullfil + nblocks * powersize ,
                                            remtimesamplesout * OUTCHANS * sizeof(float), cudaMemcpyHostToDevice));
                
                dim3 block(OUTCHANS, 1, 1);
                dim3 grid(64, 1, 1);
                
                AdjustKernel<<<grid, block, 0, 0>>>(devicepower, devicediffs, dadastrings.size(), remtimesamplesout);
                cudaDeviceSynchronize();
                cudaCheckError(cudaGetLastError());            
                
                cudaCheckError(cudaMemcpy(fullfil + nblocks * powersize, devicepower,
                                            remtimesamplesout * OUTCHANS * sizeof(float), cudaMemcpyDeviceToHost));
                
            }
        

            cudaCheckError(cudaFree(devicediffs));

        }

        cudaCheckError(cudaGetLastError());

        /**** ####
        // STAGE: Save the final filterbank file
        #### ****/
        std::cout << "Will write " << fullfilsize * sizeof(float) / 1024.0f / 1024.0f
			<< "MiB to the disk" << std::endl;
        filfile.write(reinterpret_cast<char*>(fullfil), fullfilsize * sizeof(float));

        /**** ####
        // STAGE: CLEANING UP
        #### ****/
        // NOTE: Factor of 4 to account for 2 polarisations and complex components for every time sample
        for (auto &dadastream: dadastreams) {
            dadastream.close();
        }

        filfile.close();

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

    cudaCheckError(cudaDeviceReset());

    return 0;

}