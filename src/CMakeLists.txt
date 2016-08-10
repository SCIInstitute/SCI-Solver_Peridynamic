CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)
PROJECT(PeriDynamic)

Set(CMAKE_BUILD_TYPE Debug)
cmake_policy(SET CMP0020 NEW)
if("${CMAKE_VERSION}" VERSION_GREATER 2.8.1333)
  cmake_policy(SET CMP0043 NEW)
endif()
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_SOURCE_DIR})

OPTION(BUILD_GUI "Builds the viewer." ON)

if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

if (BUILD_GUI)
  add_subdirectory(gui)
endif()
