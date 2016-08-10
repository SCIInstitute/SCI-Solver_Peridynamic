//------------------------------------------------------------------------------------------
//
//
// Created on: 3/19/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <QtGui>
#include <QtWidgets>
#include "colorselector.h"

ColorSelector::ColorSelector(QWidget *parent) : QWidget(parent),
    currentColor(Qt::blue)
{
    setMouseTracking(true);
    setAutoFillBackground(true);
}

//------------------------------------------------------------------------------------------
void ColorSelector::mousePressEvent(QMouseEvent *)
{
    QColor color = QColorDialog::getColor(currentColor, this);
    if(color.isValid())
    {
        setColor(color);

        emit colorChanged((float) color.red() / 255.0f, (float) color.green() / 255.0f,
                               (float) color.blue() / 255.0f);
    }
}

//------------------------------------------------------------------------------------------
void ColorSelector::mouseReleaseEvent(QMouseEvent *)
{
    qDebug() << "released";
}

//------------------------------------------------------------------------------------------
void ColorSelector::mouseMoveEvent(QMouseEvent *)
{
//    qDebug() << "move";
}

//------------------------------------------------------------------------------------------
bool ColorSelector::event(QEvent *e)
{
    return QWidget::event(e);
}

//------------------------------------------------------------------------------------------
void ColorSelector::enterEvent(QEvent *)
{
      QApplication::setOverrideCursor(QCursor(Qt::PointingHandCursor));
}

//------------------------------------------------------------------------------------------
void ColorSelector::leaveEvent(QEvent *)
{
    QApplication::restoreOverrideCursor();
}

//------------------------------------------------------------------------------------------
void ColorSelector::setColor(QColor _color)
{
    QPalette palette = this->palette();
    palette.setColor(QPalette::Window, _color);
    this->setPalette(palette);

    currentColor = _color;
}
