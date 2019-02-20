FROM nvidia/cuda:9.1-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    git \
    vim && \
    apt-get clean

WORKDIR /software
RUN git clone https://github.com/mmalenta/gpufilterbank.git
RUN cd gpufilterbank && \
    git checkout pbandada && \
    mkdir bin && \
    nvcc gpufil.cu -o bin/gpufil.o -lcufft -lcuda -std=c++11

WORKDIR /data



