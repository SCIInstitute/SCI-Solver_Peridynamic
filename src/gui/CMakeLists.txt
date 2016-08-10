
file(GLOB src  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB hdr  ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB rscs ${CMAKE_CURRENT_SOURCE_DIR}/*.qrc)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
# Find QT5, Glew, GLut, OpenGL

find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/core)
qt5_wrap_ui(ui_src ${uis})

get_target_property(QtWidgets_location Qt5::Widgets LOCATION)
get_target_property(QtOpenGL_location  Qt5::OpenGL LOCATION)
get_target_property(QtGui_location     Qt5::Gui LOCATION)
get_target_property(QtCore_location    Qt5::Core LOCATION)

add_executable(PeriViz ${UI_SRC} ${src} ${hdr} ${rscs} )
target_link_libraries(PeriViz #CORE_LIB
  Qt5::Widgets
  Qt5::OpenGL
  Qt5::Gui
  Qt5::Core
 ${OPENGL_LIBRARIES})
