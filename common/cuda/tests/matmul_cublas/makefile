# define the C compiler to use
CXX = g++

# ------------------------------- RELEASE SETUP ----------------------------- #

CUDA_PATH = $(patsubst %/,%,$(CUDA_HOME))
EIGEN_PATH = $(patsubst %/,%,$(EIGEN_HOME))
TWODLEARN_PATH = $(patsubst %/,%,$(TWODLEARN_HOME))

CXXFLAGS = -std=c++11 -O2

INCLUDES = -I $(EIGEN_PATH)/ -I $(addprefix $(CUDA_PATH)/,include/) -I $(TWODLEARN_PATH)/

LFLAGS = 

LIBS = -L$(addprefix $(CUDA_PATH)/,lib64/) -lcublas -lcudart


todo: matmul_cublas matmul_cublas_2
matmul_cublas:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) matmul_cublas.cc -o matmul_cublas
matmul_cublas_2:	
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) matmul_cublas_2.cc -o matmul_cublas_2
clean:
	rm matmul_cublas matmul_cublas_2
