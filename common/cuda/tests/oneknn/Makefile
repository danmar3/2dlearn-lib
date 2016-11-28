CXX = nvcc

CUDA_PATH = $(patsubst %/,%,$(CUDA_HOME))
EIGEN_PATH = $(patsubst %/,%,$(EIGEN_HOME))
TWODLEARN_PATH = $(patsubst %/,%,$(TWODLEARN_HOME))

CXXFLAGS = -std=c++11 -O2

INCLUDES =  -I $(EIGEN_PATH)/ -I $(addprefix $(CUDA_PATH)/,include/) -I $(TWODLEARN_PATH)/ 
LFLAGS = 
LIBS = -L$(addprefix $(CUDA_PATH)/,lib64/) -lcublas -lcudart

todo: main_gpu
main_gpu:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) main_gpu.cu -o main_gpu -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
clean:
	rm main_gpu
