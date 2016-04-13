//------------------------------------------------------------------------------------------
//
//
// Created on: 2/6/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <QOpenGLWidget>
#include <QList>
#include <QVector3D>
#include <QVector2D>

#ifndef UNITPLANE_H
#define UNITPLANE_H


class UnitPlane
{
public:
    UnitPlane();
    ~UnitPlane();

    int getNumVertices();
    int getNumIndices();
    int getVertexOffset();
    int getTexCoordOffset();
    int getIndexOffset();


    GLfloat* getVertices();
    GLfloat* getRandomVertexColors();
    GLfloat* getNormals();
    GLfloat* getTexureCoordinates(float _scale);
    GLushort* getIndices();


private:
    void clearData();

    QList<QVector3D> vertexList;
    QList<QVector3D> colorList;
    QList<QVector2D> texCoordList;
    QList<QVector3D> normalsList;
    GLfloat* vertices;
    GLfloat* colors;
    GLfloat* texCoord;
    GLfloat* normals;
    static GLushort indices[];
};

#endif // UNITPLANE_H
