#ifndef _H_GPUFIL_DADA
#define _H_GPUFIL_DADA

#include <fstream>
#include <string>
#include <sstream>

#include "filterbank.hpp"

// NOTE: This will have to do for now
#define HEADER_SIZE 4096

inline std::string ReadDadaValue(std::string param, std::stringstream &header) {
    
    size_t position = header.str().find(param);
    if (position != std::string::npos)
    {
        header.seekg(position + param.length());
        std::string value;
        header >> value;
        return value;
    } else {
      return "";
    }
}

inline void ReadDadaHeader(std::ifstream &indada, FilHead &head) {

    std::string dadaline;
    std::stringstream headerstream;

    unsigned char *header = new unsigned char[HEADER_SIZE];    

    indada.read(reinterpret_cast<char*>(header), HEADER_SIZE);

    headerstream << header;

    head.nbits = std::stoi(ReadDadaValue("NBIT", headerstream));    
    head.source = (ReadDadaValue("SOURCE", headerstream));
    head.tstart = std::stod(ReadDadaValue("MJD_START", headerstream));
    head.tsamp = std::stod(ReadDadaValue("TSAMP", headerstream));

}

#endif