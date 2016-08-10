TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    scene.cpp \
    mesh_query0.1/bounding_box_tree.cpp \
    mesh_query0.1/mesh_query.cpp \
    mesh_query0.1/predicates.cpp

HEADERS += \
    scene.h \
    mesh_query0.1/bounding_box_tree.h \
    mesh_query0.1/bounding_box.h \
    mesh_query0.1/mesh_query.h \
    mesh_query0.1/predicates.h \
    mesh_query0.1/util.h \
    mesh_query0.1/vec.h \
    cyPoint.h \
    cyTriMesh.h

INCLUDEPATH += /Developer/NVIDIA/CUDA-7.0/include
INCLUDEPATH += /Developer/NVIDIA/CUDA-7.0/samples/common/inc
INCLUDEPATH += /Users/nghia/SPHP/Libraries/cusplibrary-0.4.0
INCLUDEPATH += /Users/nghia/SPHP

QMAKE_CXXFLAGS +=-std=gnu++11  -stdlib=libc++

DISTFILES += \
    Makefile \
    ChangeLog.txt \
    Makefile.profile
