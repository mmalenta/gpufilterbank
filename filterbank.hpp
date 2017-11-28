#ifndef _H_VDIFIL_FIL
#define _H_VDIFIL_FIL

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

struct FilHead {
    std::string rawfile;
    std::string source;

    double az;                      // azimuth angle in deg
    double dec;                     // source declination
    double fch1;                    // frequency of the top channel in MHz
    double foff;                    // channel bandwidth in MHz
    double ra;                      // source right ascension
    double rdm;                     // reference DM
    double tsamp;                   // sampling time in seconds
    double tstart;                  // observation start time in MJD format
    double za;                      // zenith angle in deg

    int datatype;                  // data type ID
    int ibeam;                      // beam number
    int machineid;
    int nbeams;
    int nbits;
    int nchans;
    int nifs;
    int telescopeid;
};

inline void ReadFilterbankHeader(std::string config, FilHead &head) {

    std::ifstream inconfig(config.c_str());
    std::string line;
    std::string paraname;
    std::string paravalue;

    if (inconfig) {
        while(std::getline(inconfig, line)) {
            std::istringstream ossline(line);
            ossline >> paraname >> paravalue;

            if (paraname == "RAWFILE") {
                head.rawfile = paravalue;
            } else if (paraname == "SOURCE") {
                head.source = paravalue;
            } else if (paraname == "AZ") {
                head.az = (double)std::stof(paravalue);
            } else if (paraname == "DEC") {
                head.dec = (double)std::stof(paravalue);
            } else if (paraname == "FCENT") {
                head.fch1 = (double)std::stof(paravalue);
            } else if (paraname == "FOFF") {
                head.foff = (double)std::stof(paravalue);
            } else if (paraname == "RA") {
                head.ra = (double)std::stof(paravalue);
            } else if (paraname == "REFDM") {
                head.rdm = (double)std::stof(paravalue);
            } else if (paraname == "TSAMP") {
                head.tsamp = (double)std::stof(paravalue);
            } else if (paraname == "TSTART") {
                head.tstart = (double)std::stof(paravalue);
            } else if (paraname == "ZA") {
                head.za = (double)std::stof(paravalue);
            } else if (paraname == "DATATYPE") {
                head.datatype = (int)std::stoi(paravalue);
            } else if (paraname == "BEAMNO") {
                head.ibeam = (int)std::stoi(paravalue);
            } else if (paraname == "MACHINEID") {
                head.machineid = (int)std::stoi(paravalue);
            } else if (paraname == "NOBEAMS") {
                head.nbeams = (int)std::stoi(paravalue);
            } else if (paraname == "OUTBITS") {
                head.nbits = (int)std::stoi(paravalue);
            } else if (paraname == "NOCHANS") {
                head.nchans = (int)std::stoi(paravalue);
            } else if (paraname == "NOIFS") {
                head.nifs = (int)std::stoi(paravalue);
            } else if (paraname == "TELESCOPEID") {
                head.telescopeid = (int)std::stoi(paravalue);
            } else {
                std::cerr << "Unrecognised option " << paravalue << std::endl;
            }
        }
    }

}

inline void WriteFilterbankHeader(std::ofstream &outfile, FilHead &head) {

    int length{0};
    char field[60];

    length = 12;
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "HEADER_START");
    outfile.write(field, length * sizeof(char));

    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "telescope_id");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.telescopeid, sizeof(int));

    length = 11;
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "rawdatafile");
    outfile.write(field, length * sizeof(char));
    length = head.rawfile.size();
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, head.rawfile.c_str());
    outfile.write(field, length * sizeof(char));

    length = 11;
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "source_name");
    outfile.write(field, length * sizeof(char));
    length = head.source.size();
    strcpy(field, head.source.c_str());
    outfile.write((char*)&length, sizeof(int));
    outfile.write(field, length * sizeof(char));

    length = 10;
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "machine_id");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.machineid, sizeof(int));

    length = 9;
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "data_type");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.datatype, sizeof(int));

    length = 8;
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "az_start");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.az, sizeof(double));

    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "za_start");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.za, sizeof(double));

    length = 7;
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "src_raj");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.ra, sizeof(double));

    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "src_dej");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.dec, sizeof(double));

    length = 6;
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "tstart");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.tstart, sizeof(double));

    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "nchans");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.nchans, sizeof(int));

    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "nbeams");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.nbeams, sizeof(int));

    length = 5;
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "tsamp");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.tsamp, sizeof(double));

    // bits per time sample
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "nbits");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.nbits, sizeof(int));

    // reference dm - not really sure what it does
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "refdm");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.rdm, sizeof(double));

    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "ibeam");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.ibeam, sizeof(int));

    length = 4;
    // the frequency of the top channel
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "fch1");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.fch1, sizeof(double));

    // channel bandwidth
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "foff");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.foff, sizeof(double));

    // number of if channels
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "nifs");
    outfile.write(field, length * sizeof(char));
    outfile.write((char*)&head.nifs, sizeof(int));

    length = 10;
    outfile.write((char*)&length, sizeof(int));
    strcpy(field, "HEADER_END");
    outfile.write(field, length * sizeof(char));

}

#endif
