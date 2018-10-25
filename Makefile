TARGET = matrixAddition

NVCC = /usr/local/cuda/bin/nvcc -arch=sm_37
CUDA_FLAGS= -I ./h -I /usr/local/cuda-7.0/include -I /usr/local/cuda_sdk/NVIDIA_CUDA-7.0_Samples/common/inc

SRCDIR = ./src/
CUDA_SOURCES = matrixAddition.cu
C_SOURCES =

CUDA_OBJECTS = matrixAddition.o
C_OBJECTS =

all: $(TARGET)

$(TARGET): $(CUDA_OBJECTS) $(C_OBJECTS)
	$(NVCC) -o $(TARGET) $(CUDA_FLAGS) $(CUDA_OBJECTS) $(C_OBJECTS)
	rm -f *.o *.bak *~

%.o: $(SRCDIR)%.cu
	$(NVCC) -c $(CUDA_FLAGS) $(SRCDIR)$*.cu

%.o: $(SRCDIR)%.cpp
	$(NVCC) -c $(CUDA_FLAGS) $(SRCDIR)$*.cpp

clean:
	rm -f *.o *.bak *.out
	rm -f $(TARGET)
