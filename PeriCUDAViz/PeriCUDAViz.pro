#-------------------------------------------------
#
# Project created by QtCreator 2014-11-19T14:51:05
#
#-------------------------------------------------

QT       += core gui

CONFIG+=c++11
CONFIG+=warn_off
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = PeriCUDAViz
TEMPLATE = app
QMAKE_MAC_SDK = macosx10.11

SOURCES += main.cpp\
    controller.cpp \
    mainwindow.cpp \
    renderer.cpp \
    parameters.cpp \
    datareader.cpp \
    unitcube.cpp \
    unitplane.cpp \
    colorselector.cpp \
    dualviewport_renderer.cpp

HEADERS  += \
    mainwindow.h \
    controller.h \
    renderer.h \
    parameters.h \
    datareader.h \
    unitcube.h \
    unitplane.h \
    colorselector.h \
    dualviewport_renderer.h

SHARED_FOLDER = /Users/nghia/Qt5.6.0/Examples/Qt-5.6/qtbase/widgets/painting/shared
#SHARED_FOLDER = C:/Qt/Qt5.4.1/Examples/Qt-5.4/widgets/painting/shared

include($$SHARED_FOLDER/shared.pri)

RESOURCES += \
    shaders.qrc \
    textures.qrc \
    icons.qrc

macx:ICON = $${PWD}/icons/app.icns
