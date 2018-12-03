NVCC ?= /usr/local/cuda/bin/nvcc
DOCKER ?= docker
PRE ?= jam
VER ?= latest

CU_SRCS += \
./pii.cu

CU_DEPS += \
./pii.d

OBJS += \
./pii.o

.PHONY: clean all

all: matching

build-container: Dockerfile Makefile pii.cu
	$(DOCKER) build -t $(PRE)/matching:$(VER) .
	touch build-container

matching-pii:  build-container
	CID=$$($(DOCKER) create $(PRE)/matching:$(VER)) ; \
	$(DOCKER) cp $$CID:/usr/src/app/matching matching-pii ; \
	$(DOCKER) rm $$CID
	touch -c matching-pii

%.o: %.cu
	$(NVCC) -O3 -gencode arch=compute_30,code=sm_30  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	$(NVCC) --compile -O3 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"

matching: $(OBJS)
	$(NVCC) --cudart static -link -o  "matching" $(OBJS)

clean:
	rm -f $(CU_DEPS) $(OBJS) matching matching-pii build-container
