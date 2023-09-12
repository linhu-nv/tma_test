
all: tma_test  

tma_test: tma_gather_scatter_test.cu
	nvcc  -O3 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86  -rdc=true tma_gather_scatter_test.cu -o tma_test #-g -G

clean:
	rm -f tma_test *.o


