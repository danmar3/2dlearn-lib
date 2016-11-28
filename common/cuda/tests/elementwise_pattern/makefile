# define the C compiler to use
CXX = nvcc

# ------------------------------- RELEASE SETUP ----------------------------- #

CUDA_PATH = $(patsubst %/,%,$(CUDA_HOME))
EIGEN_PATH = $(patsubst %/,%,$(EIGEN_HOME))
TWODLEARN_PATH = $(patsubst %/,%,$(TWODLEARN_HOME))

CXXFLAGS = -std=c++11 -O2

INCLUDES = -I $(EIGEN_PATH)/ -I $(addprefix $(CUDA_PATH)/,include/) -I $(TWODLEARN_PATH)/

LFLAGS = 

LIBS = -L$(addprefix $(CUDA_PATH)/,lib64/) -lcublas -lcudart


todo: release_1_cmd

# --------------------------- elementwise_pattern --------------------------- #

AUX_SRC_1 = $(TWODLEARN_PATH)/twodlearn/common/cuda/elementwise_pattern_cu.cu

SRC_1 = elementwise_pattern.cu

RELEASE_1 = elementwise_pattern

release_1_cmd:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) -o $(RELEASE_1) $(SRC_1)
#$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) $(AUX_SRC_1) $(SRC_1) -o $(RELEASE_1)
clean:
	rm $(RELEASE_1)
