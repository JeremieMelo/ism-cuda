# cbert/Makefile

ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_75,code=compute_75

OPTIONS=-O3  -rdc=true -use_fast_math -lcurand  -Xcompiler -Wall -lineinfo

all: auction_cuda_main

auction_cuda_main: src/auction_cuda_main.cu
	mkdir -p bin
	nvcc -w $(ARCH) $(OPTIONS) -o bin/auction_cuda_main src/auction_cuda_main.cu

clean:
	rm -rf bin
