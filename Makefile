CXX = /opt/homebrew/opt/llvm/bin/clang++
CFLAGS = -O3 -march=native -fopenmp -Wall -Wfatal-errors -std=c++17 -fPIC -fvisibility=default

# CAKE library paths
CAKE_SRC := $(CAKE_HOME)/src

SRC_FILES = $(wildcard $(CAKE_SRC)/*.cpp)
SRC_FILES += $(wildcard $(CAKE_SRC)/kernels/armv8/*.cpp)
SRC_FILES := $(filter-out $(CAKE_SRC)/linear.cpp, $(SRC_FILES))
SRC_FILES := $(filter-out $(CAKE_SRC)/autotune_sa.cpp, $(SRC_FILES))
SRC_FILES := $(filter-out $(CAKE_SRC)/transpose.cpp, $(SRC_FILES))

INCLUDES = -Iinclude -I/usr/local/include


LIBS = -fopenmp
LDFLAGS = -L/usr/local/lib -lm -fopenmp

OUTPUT = libcake.dylib

all: $(OUTPUT)

# $(CXX) $(CFLAGS) $(INCLUDES) -shared $^ -o $@ $(LIBS) $(LDFLAGS)
$(OUTPUT): $(SRC_FILES)
	$(CXX) $(CFLAGS) $(INCLUDES) -shared $^ \
		-Wl,-install_name,@rpath/libcake.dylib \
		-o $@ \
		$(LIBS) $(LDFLAGS)

clean:
	rm -f $(OUTPUT)

.PHONY: all clean