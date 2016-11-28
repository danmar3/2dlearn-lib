# define the C compiler to use
CC = g++

# ------------------------------- RELEASE SETUP ----------------------------- #

CFLAGS_R = -std=c++11 -O2

INCLUDES_R = -I /home/marinodl/libraries/eigen_3.2.9/ -I/usr/local/cuda-7.5/include

LFLAGS_R = 

LIBS_R = -L/usr/local/cuda-7.5/lib64 -lcublas -lcudart

all: 
	$(CC) $(CFLAGS_R) $(INCLUDES_R) $(LIBS_R) matmul_cublasxt.cc -o matmul_cublasxt
