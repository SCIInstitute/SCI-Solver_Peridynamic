//------------------------------------------------------------------------------------------
//
//
// Created on: 3/19/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef COLORSELECTOR_H
#define COLORSELECTOR_H

#include <QWidget>

class ColorSelector : public QWidget
{
    Q_OBJECT
public:
    explicit ColorSelector(QWidget *parent = 0);
    void setColor(QColor _color);
    bool event(QEvent *);

signals:
    void colorChanged(float _r, float _g, float _b);

public slots:

    // QWidget interface
protected:
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
    void mouseMoveEvent(QMouseEvent *);
    void enterEvent(QEvent *);
    void leaveEvent(QEvent *);

private:
    QColor currentColor;
};

#endif // COLORSELECTOR_H
