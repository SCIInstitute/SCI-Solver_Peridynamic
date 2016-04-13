//------------------------------------------------------------------------------------------
//
//
// Created on: 2/6/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <QMatrix4x4>

#include "unitplane.h"

GLushort UnitPlane::indices[] = {0,  1,  2,
                                 2, 3, 0
                                };

inline float rand_float()
{
    return (rand() / RAND_MAX);
}

UnitPlane::UnitPlane():
    vertices(NULL),
    colors(NULL),
    texCoord(NULL),
    normals(NULL)
{
    // v0
    vertexList.append(QVector3D(-1.0f, 0.0f,  -1.0f));
    colorList.append(QVector3D(rand_float(), rand_float(), rand_float()));
    normalsList.append(QVector3D(0.0f, 1.0f,  0.0f));
    texCoordList.append(QVector2D(0.0f, 0.0f));

    // v1
    vertexList.append(QVector3D(1.0f, 0.0f,  -1.0f));
    colorList.append(QVector3D(rand_float(), rand_float(), rand_float()));
    normalsList.append(QVector3D(0.0f, 1.0f,  0.0f));
    texCoordList.append(QVector2D(1.0f, 0.0f));

    // v2
    vertexList.append(QVector3D(1.0f, 0.0f,  1.0f));
    colorList.append(QVector3D(rand_float(), rand_float(), rand_float()));
    normalsList.append(QVector3D(0.0f, 1.0f,  0.0f));
    texCoordList.append(QVector2D(1.0f, 1.0f));

    // v3
    vertexList.append(QVector3D(-1.0f, 0.0f,  1.0f));
    colorList.append(QVector3D(rand_float(), rand_float(), rand_float()));
    normalsList.append(QVector3D(0.0f, 1.0f,  0.0f));
    texCoordList.append(QVector2D(0.0f, 1.0f));

}

//------------------------------------------------------------------------------------------
UnitPlane::~UnitPlane()
{
    clearData();
}

//------------------------------------------------------------------------------------------
int UnitPlane::getNumVertices()
{
    return vertexList.size();
}

//------------------------------------------------------------------------------------------
int UnitPlane::getNumIndices()
{
    return (sizeof(indices) / sizeof(GLushort));
}

//------------------------------------------------------------------------------------------
int UnitPlane::getVertexOffset()
{
    return (sizeof(GLfloat) * getNumVertices() * 3);
}

//------------------------------------------------------------------------------------------
int UnitPlane::getTexCoordOffset()
{
    return (sizeof(GLfloat) * getNumVertices() * 2);
}

//------------------------------------------------------------------------------------------
int UnitPlane::getIndexOffset()
{
    return sizeof(indices);
}

//------------------------------------------------------------------------------------------
GLfloat* UnitPlane::getVertices()
{
    if(!vertices)
    {
        vertices = new GLfloat[vertexList.size() * 3];

        for(int i = 0; i < vertexList.size(); ++i)
        {
            QVector3D vertex = vertexList.at(i);
            vertices[3 * i] = vertex.x();
            vertices[3 * i + 1] = vertex.y();
            vertices[3 * i + 2] = vertex.z();
        }

    }

    return vertices;
}

//------------------------------------------------------------------------------------------
GLfloat* UnitPlane::getRandomVertexColors()
{
    if(!colors)
    {
        colors = new GLfloat[colorList.size() * 3];

        for(int i = 0; i < colorList.size(); ++i)
        {
            QVector3D color = colorList.at(i);
            colors[3 * i] = color.x();
            colors[3 * i + 1] = color.y();
            colors[3 * i + 2] = color.z();
        }
    }

    return colors;
}

//------------------------------------------------------------------------------------------
GLfloat* UnitPlane::getNormals()
{
    if(!normals)
    {
        normals = new GLfloat[normalsList.size() * 3];

        for(int i = 0; i < normalsList.size(); ++i)
        {
            QVector3D normal = normalsList.at(i);
            normals[3 * i] = normal.x();
            normals[3 * i + 1] = normal.y();
            normals[3 * i + 2] = normal.z();
        }

    }

    return normals;
}

//------------------------------------------------------------------------------------------
GLfloat* UnitPlane::getTexureCoordinates(float _scale)
{
    if(!texCoord)
    {
        texCoord = new GLfloat[texCoordList.size() * 2];
    }

    for(int i = 0; i < texCoordList.size(); ++i)
    {
        QVector2D tex = texCoordList.at(i);
        texCoord[2 * i] = tex.x() * _scale;
        texCoord[2 * i + 1] = tex.y() * _scale;
    }


    return texCoord;
}

//------------------------------------------------------------------------------------------
GLushort* UnitPlane::getIndices()
{
    return indices;
}

//------------------------------------------------------------------------------------------
void UnitPlane::clearData()
{
    vertexList.clear();
    normalsList.clear();
    texCoordList.clear();

    if(vertices)
    {
        delete[] vertices;
    }

    if(colors)
    {
        delete[] colors;
    }

    if(normals)
    {
        delete[] normals;
    }

    if(texCoord)
    {
        delete[] texCoord;
    }
}
