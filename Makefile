vpath %.cpp src/
vpath %.cpp tests/
vpath %.h include/
vpath %.o build/

.PHONY: all clean test

# CC compiler options:

##This is the CPU compilation section
#CC = g++ 
#LIBRARY_PATH= 
#CC_FLAGS= -g -w -O3 -std=c++17 -fPIC -fopenmp
#CC_INCLUDES = -I /usr/include

#AMD GPU Compilation Section#CC = /opt/rocm-5.2.5/bin/hipcc #AMD GPU compilation
#CC = /opt/rocm-5.2.5/bin/hipcc #AMD GPU compilation
#CC_FLAGS = -g -O3 -std=c++17 -fgpu-rdc
#INCLUDES = /opt/rocm-5.2.5/lib
#LIBRARY_PATH = 
#LIBRARIES = -L /opt/rocm-5.2.5/hiprand/lib

#Nvidia GPU Compilation Section
#using NVCC
BASEGPUPATH = /usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.6.0-u4jzhgn5buvcnkwuqrep25mluzkhzi3j
#BASEGPUPATH = /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers
#BASEGPUPATH = /opt/nvidia/hpc_sdk/Linux_x86_64/23.7/compilers

CC = $(BASEGPUPATH)/bin/nvcc
SM = 80#nvidia A100
#SM = 90#nvidia H100
#SM = 70#Nvidia V100
NVCC_FLAGS = -rdc=true -gencode arch=compute_$(SM),code=compute_$(SM)
TYPE_FLAG = -x cu
CC_FLAGS= -g -O3 -std=c++17 $(NVCC_FLAGS)
CC_INCLUDES = -I $(BASEGPUPATH)/include
LIBRARY_PATH = -L $(BASEGPUPATH)/lib64
LIBRARIES = -lcudart -lcurand
#end nvidia GPU compilation section


## Project file structure ##
MAIN = main
SOURCES = optionsParser.cpp double3.cpp integrator.cpp particle.cpp simulation.cpp quaternion.cpp floquet.cpp
INCLUDES = $(SOURCES:.cpp=.h)
OBJECTS = $(MAIN).o $(SOURCES:.cpp=.o)
BUILD = build
TEST = tests
TEST_MAIN = test_main
TEST_SOURCES = integrator_validation.cpp particle_validation.cpp floquet_validation.cpp

all: $(MAIN)

test: $(TEST_MAIN)

$(MAIN): $(OBJECTS)
	$(CC) $(CC_FLAGS) $(CC_INCLUDES) $(LIBRARY_PATH) $(addprefix $(BUILD)/,$(SOURCES:.cpp=.o)) -o $@ $(BUILD)/$(MAIN).o $(LIBRARIES)
	$(info Created executable $(MAIN))

$(TEST_MAIN): $(TEST_MAIN).o $(SOURCES:.cpp=.o) $(TEST_SOURCES:.cpp=.o)
	$(CC) $(CC_FLAGS) $(CC_INCLUDES) $(LIBRARY_PATH) $(addprefix $(BUILD)/,$(SOURCES:.cpp=.o)) $(addprefix $(BUILD)/,$(TEST_SOURCES:.cpp=.o)) -o $@ $(BUILD)/$(TEST_MAIN).o $(LIBRARIES)
	$(info Created executable $(TEST_MAIN))

$(TEST_MAIN).o: $(TEST_MAIN).cpp $(INCLUDES) | $(BUILD)
	$(CC) $(TYPE_FLAG) $(CC_FLAGS) $(CC_INCLUDES) $(LIBRARY_PATH) -c $< -o $(BUILD)/$@

$(MAIN).o : $(MAIN).cpp $(INCLUDES) | $(BUILD)
	$(CC) $(TYPE_FLAG) $(CC_FLAGS) $(CC_INCLUDES) $(LIBRARY_PATH) -c $< -o $(BUILD)/$@

%.o : %.cpp | $(BUILD)
	$(CC) $(TYPE_FLAG) $(CC_FLAGS) $(CC_INCLUDES) $(LIBRARY_PATH) -c $< -o $(BUILD)/$@

$(BUILD):
	mkdir $(BUILD)

clean:
	rm -f $(BUILD)/*.o $(MAIN)
