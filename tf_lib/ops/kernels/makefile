# define the C compiler to use
CC = nvcc

# ------------------------------- RELEASE SETUP ----------------------------- #
CUDA_PATH = $(patsubst %/,%,$(CUDA_HOME))
EIGEN_PATH = $(patsubst %/,%,$(EIGEN_HOME))
TWODLEARN_PATH = $(patsubst %/,%,$(TWODLEARN_HOME))
TENSORFLOW_PATH = $(patsubst %/,%,$(TENSORFLOW_HOME))

CFLAGS_R = -std=c++11 -shared

INCLUDES_R = -I $(addprefix $(TENSORFLOW_PATH)/,include/) -I $(TWODLEARN_PATH)/ -I $(addprefix $(CUDA_PATH)/,include/)

LFLAGS_R = -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

LIBS_R = 

OBJECTS_R = $(SOURCES:.cpp=.o)

all: bin_dir my_matmul gmm_model
#all: bin_dir gmm_model

bin_dir:
	mkdir -p bin

my_matmul:
	$(CC) $(CFLAGS_R) my_matmul_op.cu -o bin/my_matmul_op.so $(LFLAGS_R) $(INCLUDES_R)

gmm_model:
	$(CC) $(CFLAGS_R) gmm_model_op.cu -o bin/gmm_model_op.so $(LFLAGS_R) $(INCLUDES_R)
