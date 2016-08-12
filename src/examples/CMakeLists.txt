include_directories(${CMAKE_CURRENT_DIRECTORY})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/core)
FILE(GLOB SRC *.cu *.c *.cpp *.cc mesh_query0.1/*.cpp)
FILE(GLOB HDR *.cuh *.h *.hpp *.hh mesh_query0.1/*.h)

CUDA_ADD_EXECUTABLE(BunnyBreak ${SRC} ${HDR})
LINK_LIBRARIES(BunnyBreak PERIDYNAMIC_CORE)
