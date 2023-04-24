vpath %.cpp src/
vpath %.h include/
# CC compiler options:

##This is the CPU compilation section
#CC = g++ 
#CC_FLAGS= -g -w -O3 -std=c++17 -fPIC -fopenmp 

#AMD GPU Compilation Section
#CC = /opt/rocm-5.2.5/bin/hipcc #AMD GPU compilation
#CC_FLAGS = -g -O3 -std=c++17 -fgpu-rdc
#INCLUDES = /opt/rocm-5.2.5/lib
#LIBRARY_PATH = 
#LIBRARIES = -L /opt/rocm-5.2.5/hiprand/lib

#Nvidia GPU Compilation Section
#using NVCC
BASEGPUPATH = /usr/local/cuda-11.6
CC = $(BASEGPUPATH)/bin/nvcc
SM = 86
NVCC_FLAGS = -rdc=true -gencode arch=compute_$(SM),code=compute_$(SM)
TYPE_FLAG = -x cu
CC_FLAGS= -g -O3 -std=c++17 $(NVCC_FLAGS)
CC_INCLUDES = -I $(BASEGPUPATH)/include
LIBRARY_PATH = -L $(BASEGPUPATH)/lib64 
LIBRARIES = -lcudart -lcurand

#end nvidia GPU compilation section


## Project file structure ##
MAIN = main
SOURCES = double3.cpp optionsParser.cpp integrator.cpp particle.cpp simulation.cpp
INCLUDES = $(SOURCES:.cpp=.h)
OBJECTS = $(MAIN).o $(SOURCES:.cpp=.o)
BUILD = build/
EXECS =

all: $(MAIN)

$(MAIN): $(OBJECTS)
	$(CC) $(CC_FLAGS) $(CC_INCLUDES) $(LIBRARY_PATH) $(addprefix $(BUILD),$(SOURCES:.cpp=.o)) -o $(EXECS)$@ $(BUILD)$(MAIN).o $(LIBRARIES)

$(MAIN).o : $(MAIN).cpp $(INCLUDES)
	$(CC) $(TYPE_FLAG) $(CC_FLAGS) $(CC_INCLUDES) $(LIBRARY_PATH) -c $< -o $(BUILD)$@

%.o : %.cpp %.h
	$(CC) $(TYPE_FLAG) $(CC_FLAGS) $(CC_INCLUDES) $(LIBRARY_PATH) -c $< -o $(BUILD)$@

$(shell mkdir -p $(BUILD) $(EXECS))  

clean:
	rm -f bin/* *.o $(MAIN)
