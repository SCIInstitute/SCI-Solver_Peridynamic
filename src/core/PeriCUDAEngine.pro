TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt


INCLUDEPATH += /Developer/NVIDIA/CUDA-7.0/include
INCLUDEPATH += /Developer/NVIDIA/CUDA-7.0/samples/common/inc
INCLUDEPATH += /Users/nghia/SPHP/Libraries/cusplibrary-0.4.0

DISTFILES += \
    ChangeLog.txt \
    Makefile \
    Makefile.profile

HEADERS += \
    cutil_math_ext.h \
    cyPoint.h \
    cyTriMesh.h \
    dataIO.h \
    kdtree.h \
    monitor.h \
    parameters.h \
    simulator.h \
    utilities.h \
    simulator.cuh \
    memory_manager.h \
    implicit_euler.cuh \
    definitions.h \
    newmark_beta.cuh \
    cg_solver.cuh

SOURCES += \
    dataIO.cpp \
    monitor.cpp \
    parameters.cpp \
    simulator.cpp \
    memory_manager.cpp

