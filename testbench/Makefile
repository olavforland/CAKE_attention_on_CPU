CXX = /opt/homebrew/opt/llvm/bin/clang++
CFLAGS = -O3 -march=native -fopenmp -Wall -Wfatal-errors -std=c++17

CAKE_INCLUDE = $(CAKE_HOME)/include

OPENBLAS_INCLUDE = /opt/homebrew/opt/openblas/include
OPENBLAS_LIB = /opt/homebrew/opt/openblas/lib

INCLUDES = -I$(CAKE_INCLUDE) -I/usr/local/include -I$(OPENBLAS_INCLUDE)
LDFLAGS = -L$(CAKE_HOME) -Wl,-rpath,$(CAKE_HOME) -L/usr/local/lib -L$(OPENBLAS_LIB) -lcake -lopenblas -lm -fopenmp

all: compare_gemm

compare_gemm: compare_gemm.cpp
	$(CXX) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f compare_gemm

.PHONY: all clean