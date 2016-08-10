//------------------------------------------------------------------------------------------
//
//
// Created on: 1/21/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#include "unitcube.h"

#include <QMatrix4x4>


GLushort UnitCube::indices[] = {0,  1,  2,   // Face 0 - triangle strip ( v0,  v1,  v2,  v3)
                                2, 1, 3,
                                4, 5,  6,  // Face 1 - triangle strip ( v4,  v5,  v6,  v7)
                                6, 5, 7,
                                8, 9, 10, // Face 2 - triangle strip ( v8,  v9, v10, v11)
                                10, 9, 11,
                                12, 13, 14, // Face 3 - triangle strip (v12, v13, v14, v15)
                                14, 13, 15,
                                16, 17, 18, // Face 4 - triangle strip (v16, v17, v18, v19)
                                18, 17, 19,
                                20, 21, 22,      // Face 5 - triangle strip (v20, v21, v22, v23)
                                22, 21, 23
                               };

GLushort UnitCube::lineIndices[] = {0, 1,
                                    1, 3,
                                    3, 2,
                                    2, 0,   // Face 0 - triangle strip ( v0,  v1,  v2,  v3)
                                    4, 5,
                                    5, 7,
                                    7, 6,
                                    6, 4,  // Face 1 - triangle strip ( v4,  v5,  v6,  v7)
                                    8, 9,
                                    9, 11,
                                    11, 10,
                                    10, 8, // Face 2 - triangle strip ( v8,  v9, v10, v11)
                                    12, 13,
                                    13, 15,
                                    15, 14,
                                    14, 12, // Face 3 - triangle strip (v12, v13, v14, v15)
                                    16, 17,
                                    17, 19,
                                    19, 18,
                                    18, 16, // Face 4 - triangle strip (v16, v17, v18, v19)
                                    20, 21,
                                    21, 23,
                                    23, 22,
                                    22, 20      // Face 5 - triangle strip (v20, v21, v22, v23)
                                   };


UnitCube::UnitCube():
    vertices(NULL),
    transformedVertices(NULL),
    colors(NULL),
    texCoord(NULL),
    normals(NULL),
    negNormals(NULL)
{

    // Vertex data for face 0
    // v0
    vertexList.append(QVector3D(-1.0f, -1.0f,  1.0f));
    colorList.append(QVector3D(0.0f, 0.0f,  1.0f));
    normalsList.append(QVector3D(0.0f, 0.0f,  1.0f));
    texCoordList.append(QVector2D(0.0f, 0.0f));

    // v1
    vertexList.append(QVector3D(1.0f, -1.0f,  1.0f));
    colorList.append(QVector3D(1.0f, 0.0f,  1.0f));
    normalsList.append(QVector3D(0.0f, 0.0f,  1.0f));
    texCoordList.append(QVector2D(1.0f, 0.0f));

    // v2
    vertexList.append(QVector3D(-1.0f, 1.0f,  1.0f));
    colorList.append(QVector3D(0.0f, 1.0f,  1.0f));
    normalsList.append(QVector3D(0.0f, 0.0f,  1.0f));
    texCoordList.append(QVector2D(0.0f, 1.0f));

    // v3
    vertexList.append(QVector3D(1.0f, 1.0f,  1.0f));
    colorList.append(QVector3D(1.0f, 1.0f,  1.0f));
    normalsList.append(QVector3D(0.0f, 0.0f,  1.0f));
    texCoordList.append(QVector2D(1.0f, 1.0f));


    // Vertex data for face 1
    // v4
    vertexList.append(QVector3D(1.0f, -1.0f,  1.0f));
    colorList.append(QVector3D(1.0f, 0.0f,  1.0f));
    normalsList.append(QVector3D(1.0f, 0.0f,  0.0f));
    texCoordList.append(QVector2D(0.0f, 0.0f));

    // v5
    vertexList.append(QVector3D(1.0f, -1.0f,  -1.0f));
    colorList.append(QVector3D(1.0f, 0.0f,  0.0f));
    normalsList.append(QVector3D(1.0f, 0.0f,  0.0f));
    texCoordList.append(QVector2D(1.0f, 0.0f));

    // v6
    vertexList.append(QVector3D(1.0f, 1.0f,  1.0f));
    colorList.append(QVector3D(1.0f, 1.0f,  1.0f));
    normalsList.append(QVector3D(1.0f, 0.0f,  0.0f));
    texCoordList.append(QVector2D(0.0f, 1.0f));

    // v7
    vertexList.append(QVector3D(1.0f, 1.0f,  -1.0f));
    colorList.append(QVector3D(1.0f, 1.0f,  0.0f));
    normalsList.append(QVector3D(1.0f, 0.0f,  0.0f));
    texCoordList.append(QVector2D(1.0f, 1.0f));


    // Vertex data for face 2
    // v8
    vertexList.append(QVector3D(1.0f, -1.0f,  -1.0f));
    colorList.append(QVector3D(1.0f, 0.0f,  0.0f));
    normalsList.append(QVector3D(0.0f, 0.0f,  -1.0f));
    texCoordList.append(QVector2D(0.0f, 0.0f));

    // v9
    vertexList.append(QVector3D(-1.0f, -1.0f,  -1.0f));
    colorList.append(QVector3D(0.0f, 0.0f,  0.0f));
    normalsList.append(QVector3D(0.0f, 0.0f,  -1.0f));
    texCoordList.append(QVector2D(1.0f, 0.0f));

    // v10
    vertexList.append(QVector3D(1.0f, 1.0f,  -1.0f));
    colorList.append(QVector3D(1.0f, 1.0f,  0.0f));
    normalsList.append(QVector3D(0.0f, 0.0f,  -1.0f));
    texCoordList.append(QVector2D(0.0f, 1.0f));

    // v11
    vertexList.append(QVector3D(-1.0f, 1.0f,  -1.0f));
    colorList.append(QVector3D(0.0f, 1.0f,  0.0f));
    normalsList.append(QVector3D(0.0f, 0.0f,  -1.0f));
    texCoordList.append(QVector2D(1.0f, 1.0f));

    // Vertex data for face 3
    // v12
    vertexList.append(QVector3D(-1.0f, -1.0f,  -1.0f));
    colorList.append(QVector3D(0.0f, 0.0f,  0.0f));
    normalsList.append(QVector3D(-1.0f, 0.0f,  0.0f));
    texCoordList.append(QVector2D(0.0f, 0.0f));

    // v13
    vertexList.append(QVector3D(-1.0f, -1.0f,  1.0f));
    colorList.append(QVector3D(0.0f, 0.0f,  1.0f));
    normalsList.append(QVector3D(-1.0f, 0.0f,  0.0f));
    texCoordList.append(QVector2D(1.0f, 0.0f));

    // v14
    vertexList.append(QVector3D(-1.0f, 1.0f,  -1.0f));
    colorList.append(QVector3D(0.0f, 1.0f,  0.0f));
    normalsList.append(QVector3D(-1.0f, 0.0f,  0.0f));
    texCoordList.append(QVector2D(0.0f, 1.0f));

    // v15
    vertexList.append(QVector3D(-1.0f, 1.0f,  1.0f));
    colorList.append(QVector3D(0.0f, 1.0f,  1.0f));
    normalsList.append(QVector3D(-1.0f, 0.0f,  0.0f));
    texCoordList.append(QVector2D(1.0f, 1.0f));

    // Vertex data for face 4
    // v16
    vertexList.append(QVector3D(-1.0f, -1.0f,  -1.0f));
    colorList.append(QVector3D(0.0f, 0.0f,  0.0f));
    normalsList.append(QVector3D(0.0f, -1.0f,  0.0f));
    texCoordList.append(QVector2D(0.0f, 0.0f));

    // v17
    vertexList.append(QVector3D(1.0f, -1.0f,  -1.0f));
    colorList.append(QVector3D(1.0f, 0.0f,  0.0f));
    normalsList.append(QVector3D(0.0f, -1.0f,  0.0f));
    texCoordList.append(QVector2D(1.0f, 0.0f));

    // v18
    vertexList.append(QVector3D(-1.0f, -1.0f,  1.0f));
    colorList.append(QVector3D(0.0f, 0.0f,  1.0f));
    normalsList.append(QVector3D(0.0f, -1.0f,  0.0f));
    texCoordList.append(QVector2D(0.0f, 1.0f));

    // v19
    vertexList.append(QVector3D(1.0f, -1.0f,  1.0f));
    colorList.append(QVector3D(1.0f, 0.0f,  1.0f));
    normalsList.append(QVector3D(0.0f, -1.0f,  0.0f));
    texCoordList.append(QVector2D(1.0f, 1.0f));


    // Vertex data for face 5
    // v20
    vertexList.append(QVector3D(-1.0f, 1.0f,  1.0f));
    colorList.append(QVector3D(0.0f, 1.0f,  1.0f));
    normalsList.append(QVector3D(0.0f, 1.0f,  0.0f));
    texCoordList.append(QVector2D(0.0f, 0.0f));

    // v21
    vertexList.append(QVector3D(1.0f, 1.0f,  1.0f));
    colorList.append(QVector3D(1.0f, 1.0f,  1.0f));
    normalsList.append(QVector3D(0.0f, 1.0f,  0.0f));
    texCoordList.append(QVector2D(1.0f, 0.0f));

    // v22
    vertexList.append(QVector3D(-1.0f, 1.0f,  -1.0f));
    colorList.append(QVector3D(0.0f, 1.0f,  0.0f));
    normalsList.append(QVector3D(0.0f, 1.0f,  0.0f));
    texCoordList.append(QVector2D(0.0f, 1.0f));

    // v23
    vertexList.append(QVector3D(1.0f, 1.0f,  -1.0f));
    colorList.append(QVector3D(1.0f, 1.0f,  0.0f));
    normalsList.append(QVector3D(0.0f, 1.0f,  0.0f));
    texCoordList.append(QVector2D(1.0f, 1.0f));



    for(int i = 0; i < 12; ++i)
    {
        CubeFaceTriangle face;
        int index0 = indices[i * 3];
        int index1 = indices[i * 3 + 1];
        int index2 = indices[i * 3 + 2];
        face.faceNormal = normalsList.at(index0);
        face.vertices[0] = vertexList.at(index0);
        face.vertices[1] = vertexList.at(index1);
        face.vertices[2] = vertexList.at(index2);
        face.indices[0] = index0;
        face.indices[1] = index1;
        face.indices[2] = index2;

        faceList.append(face);
    }
}

//------------------------------------------------------------------------------------------
UnitCube::~UnitCube()
{
    clearData();

}

//------------------------------------------------------------------------------------------
int UnitCube::getNumVertices()
{
    return vertexList.size();
}

//------------------------------------------------------------------------------------------
int UnitCube::getNumIndices()
{
    return (sizeof(indices) / sizeof(GLushort));
}

//------------------------------------------------------------------------------------------
int UnitCube::getNumLineIndices()
{
    return (sizeof(lineIndices) / sizeof(GLushort));
}

//------------------------------------------------------------------------------------------
int UnitCube::getVertexOffset()
{
    return (sizeof(GLfloat) * getNumVertices() * 3);
}

//------------------------------------------------------------------------------------------
int UnitCube::getTexCoordOffset()
{
    return (sizeof(GLfloat) * getNumVertices() * 2);
}

//------------------------------------------------------------------------------------------
int UnitCube::getIndexOffset()
{
    return sizeof(indices);
}

//------------------------------------------------------------------------------------------
int UnitCube::getLineIndexOffset()
{
    return sizeof(lineIndices);
}

//------------------------------------------------------------------------------------------
int UnitCube::getNumFaceTriangles()
{
    return faceList.size();
}

//------------------------------------------------------------------------------------------
GLfloat* UnitCube::getVertices()
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
GLfloat* UnitCube::getVertexColors()
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
GLfloat* UnitCube::getNormals()
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
GLfloat* UnitCube::getNegativeNormals()
{
    if(!negNormals)
    {
        negNormals = new GLfloat[normalsList.size() * 3];
    }

    for(int i = 0; i < normalsList.size(); ++i)
    {
        QVector3D normal = normalsList.at(i);
        negNormals[3 * i] = -normal.x();
        negNormals[3 * i + 1] = -normal.y();
        negNormals[3 * i + 2] = -normal.z();
    }

    return negNormals;
}

//------------------------------------------------------------------------------------------
GLfloat* UnitCube::getTexureCoordinates(float _scale)
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
GLushort* UnitCube::getIndices()
{
    return indices;
}

//------------------------------------------------------------------------------------------
GLushort* UnitCube::getLineIndices()
{
    return lineIndices;
}

//------------------------------------------------------------------------------------------
UnitCube::CubeFaceTriangle UnitCube::getFace(int _faceIndex)
{
    return faceList.at(_faceIndex);
}

//------------------------------------------------------------------------------------------
void UnitCube::clearData()
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

//------------------------------------------------------------------------------------------
