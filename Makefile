########################################
#@Author: Jiaqi Gu (jqgu@utexas.edu)
#@Date: 2020-10-22 12:33:39
#@LastEditors: Jiaqi Gu (jqgu@utexas.edu)
#@LastEditTime: 2020-10-22 12:33:39
########################################
# Makefile
ROOT = ${PWD}
OUTPUT_DIR = $(ROOT)/bin
TARGET = auction_cuda

RM = rm -rf

SRCS :=
INCLUDES :=
LIBS :=
OBJS :=
DEPS :=
CFLAGS :=

SRCS += $(ROOT)/src/*.cu
INCLUDES += $(ROOT)/src
NVCC = nvcc

ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_75,code=compute_75


OPTIONS = -O3  -rdc=true -use_fast_math -lcurand  -Xcompiler -Wall -lineinfo

NVCC_FLAGS = $(ARCH) $(OPTIONS)

all: auction_cuda

auction_cuda: $(SRCS) $(INCLUDES)
	mkdir -p bin
	$(NVCC) -w $(NVCC_FLAGS) -o $(OUTPUT_DIR)/$(TARGET) $(SRCS) -I $(INCLUDES)

clean:
	$(RM) $(OUTPUT_DIR)
