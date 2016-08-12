
file(GLOB src  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB hdr  ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
# Find QT5, Glew, GLut, OpenGL

find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/core)

get_target_property(QtWidgets_location Qt5::Widgets LOCATION)
get_target_property(QtOpenGL_location  Qt5::OpenGL LOCATION)
get_target_property(QtGui_location     Qt5::Gui LOCATION)
get_target_property(QtCore_location    Qt5::Core LOCATION)

# For Apple set the icns file containing icons
IF(APPLE)
  # set how it shows up in the Info.plist file
  SET(MACOSX_BUNDLE_ICON_FILE ${CMAKE_SOURCE_DIR}/Resources/icons/app.icns)
  # set where in the bundle to put the icns file
  SET_SOURCE_FILES_PROPERTIES( ${CMAKE_SOURCE_DIR}/Resources/icons/app.icns PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
  # include the icns file in the target
  SET(src ${src} ${CMAKE_SOURCE_DIR}/Resources/icons/app.icns)
ELSEIF(WIN32)
  ADD_DEFINITIONS(-DICON_RC_FILE="${CMAKE_SOURCE_DIR}/Resources/icons/app.ico")
  SET(src ${src} ${CMAKE_SOURCE_DIR}/Resources/icons/PeriViz.rc)
ENDIF(APPLE)

add_executable(PeriViz MACOSX_BUNDLE
  ${UI_SRC} ${src} ${hdr} ${rscs} )
target_link_libraries(PeriViz
  Qt5::Widgets
  Qt5::OpenGL
  Qt5::Gui
  Qt5::Core
  ${OPENGL_LIBRARIES})
#copy resources to the binary directory
add_custom_command(TARGET PeriViz POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "${CMAKE_SOURCE_DIR}/Resources/textures"
  "$<TARGET_FILE_DIR:PeriViz>/textures")
add_custom_command(TARGET PeriViz POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "${CMAKE_SOURCE_DIR}/Resources/shaders"
  "$<TARGET_FILE_DIR:PeriViz>/shaders")
add_custom_command(TARGET PeriViz POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "${CMAKE_SOURCE_DIR}/Resources/icons"
  "$<TARGET_FILE_DIR:PeriViz>/icons")
add_custom_command(TARGET PeriViz POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "${CMAKE_SOURCE_DIR}/Resources/textures"
  "${CMAKE_CURRENT_BINARY_DIR}/textures")
add_custom_command(TARGET PeriViz POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "${CMAKE_SOURCE_DIR}/Resources/shaders"
  "${CMAKE_CURRENT_BINARY_DIR}/shaders")
add_custom_command(TARGET PeriViz POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "${CMAKE_SOURCE_DIR}/Resources/icons"
  "${CMAKE_CURRENT_BINARY_DIR}/icons")
