#CPP = clang++
#CPPFLAGS = -Wall -O3 -m64 -DNDEBUG # -g
ifneq ($(DARWIN),)
ifeq ($(XCODE_GE_5),1)
  GCC ?= clang++
else
  GCC ?= g++
endif
else
  GCC ?= g++
endif
#CPP = /usr/local/cuda/bin/nvcc -ccbin clang++ -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -w -O2 -m64  -Xcompiler -arch -Xcompiler x86_64     -gencode arch=compute_30,code=sm_30  -gencode arch=compute_35,code=sm_35   -x cu -dc
CPP = /usr/local/cuda/bin/nvcc -ccbin $(GCC) -I/usr/local/cuda/samples/common/inc -w -O2 -m64 -gencode arch=compute_20,code=sm_20   -x cu -dc
LDFLAGS = -lm

all: libmesh_query.a

libmesh_query.a: mesh_query.o predicates.o bounding_box_tree.o
	ar r $@ mesh_query.o predicates.o bounding_box_tree.o;
	ranlib $@

mesh_query.o: mesh_query.cpp mesh_query.h bounding_box_tree.h bounding_box.h predicates.h vec.h util.h
	$(CPP) $(CPPFLAGS) -o $@ -c mesh_query.cpp

predicates.o: predicates.cpp predicates.h
	$(CPP) $(CPPFLAGS) -o $@ -c predicates.cpp

bounding_box_tree.o: bounding_box_tree.cpp bounding_box_tree.h bounding_box.h vec.h util.h
	$(CPP) $(CPPFLAGS) -o $@ -c bounding_box_tree.cpp

clean:
	-rm libmesh_query.a mesh_query.o predicates.o bounding_box_tree.o
