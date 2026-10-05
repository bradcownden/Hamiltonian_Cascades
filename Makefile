# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anxo Biasi, Brad Cownden, Oleg Evnin, Alvaro Iturbe Jabaloyes
#
# Build: make [ARCH=-arch=sm_80] [BOOST_INC=/path/to/boost/include]

NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -O3 -std=c++17
ARCH = -arch=sm_75 # Use sm_75 for Tesla gpus, sm_80 for A100
# Run ./deviceQuery from cuda samples to get "CUDA Capability Major/Minus version number"

# Override on each machine/OS, e.g.:
#   make BOOST_INC=/usr/include
#   make BOOST_INC=/opt/homebrew/include
BOOST_INC ?= /usr/local/include
INCFLAGS = -I$(BOOST_INC)

# Needed when bigfloat resolves to boost::multiprecision::float128.
# Override on platforms/toolchains that do not use libquadmath.
MATH_LIBS ?= -lquadmath
# cuRAND: used by rk_sm.cu 
MATH_LIBS += -lcurand

CU_SRCS = main.cu io.cu kernels.cu rk.cu rk_sm.cu
CPP_SRCS = initial_data.cpp
HDRS = main.cuh initial_data.hpp complex128.hpp bigfloat.hpp

all: cascades

cascades: $(CU_SRCS) $(CPP_SRCS) $(HDRS)
	@echo "\nInitialize compilation...\n"
	$(NVCC) $(CFLAGS) $(ARCH) $(INCFLAGS) $(CU_SRCS) $(CPP_SRCS) $(MATH_LIBS) -o cascades
	@echo "\nDone!\n"
	@echo "=================== Compilation Complete ===================\n"

clean:
	rm -f cascades
