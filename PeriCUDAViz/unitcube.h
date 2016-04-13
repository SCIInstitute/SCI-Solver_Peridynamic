//------------------------------------------------------------------------------------------
//
//
// Created on: 1/21/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <QOpenGLWidget>
#include <QList>
#include <QVector3D>
#include <QVector2D>
#include <QtGui>

#ifndef UNITCUBE_H
#define UNITCUBE_H


class UnitCube
{
public:
    class CubeFaceTriangle
    {
    public:
        bool findSharedVertices(QVector3D* _otherTriangleVertices, QVector3D* _result)
        {
            int numSharedVertex = 0;
            int sharedIndices[2];

            for(int i = 0; i < 3; ++i)
            {
                for(int j = 0; j < 3; ++j)
                {
                    if((vertices[i] - _otherTriangleVertices[j]).lengthSquared() < 1e-8)
                    {
                        sharedIndices[numSharedVertex] = i;
                        ++numSharedVertex;
                    }
                }
            }

            if(numSharedVertex != 2)
            {
                return false;
            }

//            qDebug() << numSharedVertex;

            if(sharedIndices[1] - sharedIndices[0] == 1)
            {
                _result[0] = vertices[sharedIndices[0]];
                _result[1] = vertices[sharedIndices[1]];
            }
            else
            {
                _result[0] = vertices[sharedIndices[1]];
                _result[1] = vertices[sharedIndices[0]];
            }

            return true;
        }

        QVector3D vertices[3];
        int indices[3];
        QVector3D faceNormal;
    };

    UnitCube();
    ~UnitCube();

    int getNumVertices();
    int getNumIndices();
    int getNumLineIndices();
    int getVertexOffset();
    int getTexCoordOffset();
    int getIndexOffset();
    int getLineIndexOffset();
    int getNumFaceTriangles();


    GLfloat* getVertices();
    GLfloat* getVertexColors();
    GLfloat* getNormals();
    GLfloat* getNegativeNormals();
    GLfloat* getTexureCoordinates(float _scale);
    GLushort* getIndices();
    GLushort* getLineIndices();
    CubeFaceTriangle getFace(int _faceIndex);


private:
    void clearData();

    QList<QVector3D> vertexList;
    QList<QVector3D> colorList;
    QList<QVector2D> texCoordList;
    QList<QVector3D> normalsList;
    QList<CubeFaceTriangle> faceList;
    GLfloat* vertices;
    GLfloat* transformedVertices;
    GLfloat* colors;
    GLfloat* texCoord;
    GLfloat* normals;
    GLfloat* negNormals;
    static GLushort indices[];
    static GLushort lineIndices[];
};

#endif // UNITCUBE_H
