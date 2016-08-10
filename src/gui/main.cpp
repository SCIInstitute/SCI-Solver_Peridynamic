/**************************************************************************
* main.cpp
*
* Created on: 02/20/2015
*     Author: nghiatruong
*
**************************************************************************/

#include "mainwindow.h"
//#include "arthurwidgets.h"
#include <QDesktopWidget>
#include <QApplication>


int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    QApplication::setWindowIcon(QIcon(":/icons/app.png"));

    QSurfaceFormat format;
    format.setVersion(4, 1);
    format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);

    MainWindow mainWindow;

    //QStyle* arthurStyle = new ArthurStyle();
    //mainWindow.setStyle(arthurStyle);
    QList<QWidget*> widgets = mainWindow.findChildren<QWidget*>();
    foreach (QWidget * w, widgets)
    {
        QString className = QString(w->metaObject()->className());

        if(className == "QScrollBar" || className == "QComboBox" ||
           className == "QCheckBox")
        {
            continue;
        }


       // w->setStyle(arthurStyle);
    }

    mainWindow.show();
    mainWindow.setGeometry(QStyle::alignedRect(Qt::LeftToRight, Qt::AlignCenter,
                                               mainWindow.size(),
                                               qApp->desktop()->availableGeometry()));


    return a.exec();
}
