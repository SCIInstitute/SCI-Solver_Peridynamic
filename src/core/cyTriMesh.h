// cyCodeBase by Cem Yuksel
// [www.cemyuksel.com]
//-------------------------------------------------------------------------------
///
/// \file		cyTriMesh.h 
/// \author		Cem Yuksel
/// \version	1.2
/// \date		October 23, 2013
///
/// \brief Triangular Mesh class.
///
//-------------------------------------------------------------------------------

#ifndef _CY_TRIMESH_H_INCLUDED_
#define _CY_TRIMESH_H_INCLUDED_

//-------------------------------------------------------------------------------

#include "cyPoint.h"
#include <stdio.h>

//-------------------------------------------------------------------------------

/// Triangular Mesh Class

class cyTriMesh
{
public:
	/// Triangular Mesh Face
	class cyTriFace
	{
	public:
		unsigned int v[3];	// vertex indices
	};

protected:
	cyPoint3f		*v;		///< vertices
	cyTriFace		*f;		///< faces
	cyPoint3f		*vn;	///< vertex normal
	cyTriFace		*fn;	///< normal faces
	cyPoint3f		*vt;	///< texture vertices
	cyTriFace		*ft;	///< texture faces

	unsigned int	nv;		///< number of vertices
	unsigned int	nf;		///< number of faces
	unsigned int	nvn;	///< number of vertex normals
	unsigned int	nvt;	///< number of texture vertices

	cyPoint3f boundMin, boundMax;	///< bounding box

public:

	cyTriMesh() : v(NULL), f(NULL), vn(NULL), fn(NULL), vt(NULL), ft(NULL), nv(0), nf(0), nvn(0), nvt(0), boundMin(0,0,0), boundMax(0,0,0) {}
	virtual ~cyTriMesh() { Clear(); }

	///@name Component Access Methods
	cyPoint3f&			V(int i)		{ return v[i]; }	///< returns the i^th vertex
	const cyPoint3f&	V(int i) const	{ return v[i]; }	///< returns the i^th vertex
	cyTriFace&			F(int i)		{ return f[i]; }	///< returns the i^th face
	const cyTriFace&	F(int i) const	{ return f[i]; }	///< returns the i^th face
	cyPoint3f&			VN(int i)		{ return vn[i]; }	///< returns the i^th vertex normal
	const cyPoint3f&	VN(int i) const	{ return vn[i]; }	///< returns the i^th vertex normal
	cyTriFace&			FN(int i)		{ return fn[i]; }	///< returns the i^th normal face
	const cyTriFace&	FN(int i) const	{ return fn[i]; }	///< returns the i^th normal face
	cyPoint3f&			VT(int i)		{ return vt[i]; }	///< returns the i^th vertex texture
	const cyPoint3f&	VT(int i) const	{ return vt[i]; }	///< returns the i^th vertex texture
	cyTriFace&			FT(int i)		{ return ft[i]; }	///< returns the i^th texture face
	const cyTriFace&	FT(int i) const	{ return ft[i]; }	///< returns the i^th texture face

	unsigned int		NV() const		{ return nv; }		///< returns the number of vertices
	unsigned int		NF() const		{ return nf; }		///< returns the number of faces
	unsigned int		NVN() const		{ return nvn; }		///< returns the number of vertex normals
	unsigned int		NVT() const		{ return nvt; }		///< returns the number of texture vertices

	bool HasNormals() const { return NVN() > 0; }			///< returns true if the mesh has vertex normals
	bool HasTextureVertices() const { return NVT() > 0; }	///< returns true if the mesh has texture vertices

	///@name Set Component Count
	void Clear() { SetNumVertex(0); SetNumFaces(0); SetNumNormals(0); SetNumTexVerts(0); boundMin.Zero(); boundMax.Zero(); }
	void SetNumVertex  (unsigned int n) { Allocate(n,v,nv); }
	void SetNumFaces   (unsigned int n) { if ( Allocate(n,f,nf) ) { if (fn) Allocate(n,fn); if (ft) Allocate(n,ft); } }
	void SetNumNormals (unsigned int n) { Allocate(n,vn,nvn); if (!fn) Allocate(nf,fn); }
	void SetNumTexVerts(unsigned int n) { Allocate(n,vt,nvt); if (!ft) Allocate(nf,ft); }

	///@name Get Property Methods
	bool		IsBoundBoxReady() const { return boundMin.x!=0 && boundMin.y!=0 && boundMin.z!=0 && boundMax.x!=0 && boundMax.y!=0 && boundMax.z!=0; }
	cyPoint3f	GetBoundMin() const { return boundMin; }		///< Returns the minimum values of the bounding box
	cyPoint3f	GetBoundMax() const { return boundMax; }		///< Returns the maximum values of the bounding box
	cyPoint3f	GetPoint   (int faceID, const cyPoint3f &bc) const { return Interpolate(faceID,v,f,bc); }	///< Returns the point on the given face with the given barycentric coordinates (bc).
	cyPoint3f	GetNormal  (int faceID, const cyPoint3f &bc) const { return Interpolate(faceID,vn,fn,bc); }	///< Returns the the surface normal on the given face at the given barycentric coordinates (bc). The returned vector is not normalized.
	cyPoint3f	GetTexCoord(int faceID, const cyPoint3f &bc) const { return Interpolate(faceID,vt,ft,bc); }	///< Returns the texture coordinate on the given face at the given barycentric coordinates (bc).

	///@name Compute Methods
	void ComputeBoundingBox();						///< Computes the bounding box
	void ComputeNormals(bool clockwise=false);		///< Computes and stores vertex normals

	///@name Load and Save methods
	bool LoadFromFileObj( const char *filename );	///< Loads the mesh from an OBJ file. Automatically converts all faces to triangles.

private:
	template <class T> void Allocate(unsigned int n, T* &t) { if (t) delete [] t; if (n>0) t = new T[n]; else t=NULL; }
	template <class T> bool Allocate(unsigned int n, T* &t, unsigned int &nt) { if (n==nt) return false; nt=n; Allocate(n,t); return true; }
	static cyPoint3f Interpolate( int i, const cyPoint3f *v, const cyTriFace *f, const cyPoint3f &bc ) { return v[f[i].v[0]]*bc.x + v[f[i].v[1]]*bc.y + v[f[i].v[2]]*bc.z; }
	static int  ReadLine( FILE *fp, int size, char *buffer );
	static void ReadVertex( const char *buffer, cyPoint3f &v ) { sscanf( buffer+2, "%f %f %f", &v.x, &v.y, &v.z ); }
};

//-------------------------------------------------------------------------------

inline void cyTriMesh::ComputeBoundingBox()
{
	boundMin=v[0];
	boundMax=v[0];
	for ( unsigned int i=1; i<nv; i++ ) {
		if ( boundMin.x > v[i].x ) boundMin.x = v[i].x;
		if ( boundMin.y > v[i].y ) boundMin.y = v[i].y;
		if ( boundMin.z > v[i].z ) boundMin.z = v[i].z;
		if ( boundMax.x < v[i].x ) boundMax.x = v[i].x;
		if ( boundMax.y < v[i].y ) boundMax.y = v[i].y;
		if ( boundMax.z < v[i].z ) boundMax.z = v[i].z;
	}
}

inline void cyTriMesh::ComputeNormals(bool clockwise)
{
	SetNumNormals(nv);
	for ( unsigned int i=0; i<nvn; i++ ) vn[i].Set(0,0,0);	// initialize all normals to zero
	for ( unsigned int i=0; i<nf; i++ ) {
		cyPoint3f N = (v[f[i].v[1]]-v[f[i].v[0]]) ^ (v[f[i].v[2]]-v[f[i].v[0]]);	// face normal (not normalized)
		if ( clockwise ) N = -N;
		vn[f[i].v[0]] += N;
		vn[f[i].v[1]] += N;
		vn[f[i].v[2]] += N;
		fn[i] = f[i];
	}
	for ( unsigned int i=0; i<nvn; i++ ) vn[i].Normalize();
}

inline bool cyTriMesh::LoadFromFileObj( const char *filename )
{
	FILE *fp = fopen(filename,"r");
    if ( !fp ) {
        printf("Could not open obj file: %s\n", filename);
        exit(-1);
//        return false;
    }

	Clear();

	unsigned int numVerts=0, numTVerts=0, numNormals=0, numFaces=0;

	const int bufsize = 1024;
	char buffer[bufsize];

	while ( int rb = ReadLine( fp, bufsize, buffer ) ) {
		switch ( buffer[0] ) {
			case 'v':
				switch ( buffer[1] ) {
					case ' ' :
					case '\t': numVerts++; break;
					case 't' : numTVerts++; break;
					case 'n' : numNormals++; break;
				}
				break;
			case 'f': {
					int nFaceVerts = 0; // count face vertices
					bool inspace = true;
					for ( int i=2; i<rb-1; i++ ) {
						if ( buffer[i] == ' ' || buffer[i] == '\t' ) inspace = true;
						else if ( inspace ) { nFaceVerts++; inspace = false; }
					}
					if ( nFaceVerts > 2 ) numFaces += nFaceVerts-2; // non-triangle faces will be multiple triangle faces
				}
				break;
		}
		if ( feof(fp) ) break;
	}

	if ( numFaces == 0 ) return true; // No faces found
	SetNumVertex(numVerts);
	SetNumFaces(numFaces);
	SetNumNormals(numNormals);
	SetNumTexVerts(numTVerts);

	unsigned int readVerts = 0;
	unsigned int readTVerts = 0;
	unsigned int readNormals = 0;
	unsigned int readFaces = 0;

	rewind(fp);
	while ( int rb = ReadLine( fp, bufsize, buffer ) ) {
		switch ( buffer[0] ) {
		case 'v':
			switch ( buffer[1] ) {
				case ' ' :
				case '\t': ReadVertex(buffer, v[readVerts++]); break;
				case 't' : ReadVertex(buffer, vt[readTVerts++]); break;
				case 'n' : ReadVertex(buffer, vn[readNormals++]); break;
			}
			break;
		case 'f': {
				int facevert = -1;
				bool inspace = true;
				int type = 0;
				unsigned int index;
				for ( int i=2; i<rb-1; i++ ) {
					if ( buffer[i] == ' ' || buffer[i] == '\t' ) inspace = true;
					else {
						if ( inspace ) {
							inspace=false;
							type=0;
							index=0;
							switch ( facevert ) {
								case -1:
									// initialize face
									f[readFaces].v[0] = f[readFaces].v[1] = f[readFaces].v[2] = 0;
									if ( ft ) ft[readFaces].v[0] = ft[readFaces].v[1] = ft[readFaces].v[2] = 0;
									if ( fn ) fn[readFaces].v[0] = fn[readFaces].v[1] = fn[readFaces].v[2] = 0;
								case 0:
								case 1:
									facevert++;
									break;
								case 2:
									// copy the first two vertices from the previous face
									readFaces++;
									f[readFaces].v[0] = f[readFaces-1].v[0];
									f[readFaces].v[1] = f[readFaces-1].v[2];
									if ( ft ) {
										ft[readFaces].v[0] = ft[readFaces-1].v[0];
										ft[readFaces].v[1] = ft[readFaces-1].v[2];
									}
									if ( fn ) {
										fn[readFaces].v[0] = fn[readFaces-1].v[0];
										fn[readFaces].v[1] = fn[readFaces-1].v[2];
									}
									break;
							}
						}
						if ( buffer[i] == '/' ) { type++; index=0; }
						if ( buffer[i] >= '0' && buffer[i] <= '9' ) {
							index = index*10 + (buffer[i]-'0');
							switch ( type ) {
								case 0: f[readFaces].v[facevert] = index-1; break;
								case 1: if (ft) ft[readFaces].v[facevert] = index-1; break;
								case 2: if (fn) fn[readFaces].v[facevert] = index-1; break;
							}
						}
					}
				}
				readFaces++;
			}
			break;
		}
		if ( feof(fp) ) break;
	}

	fclose(fp);
	return true;
}

inline int cyTriMesh::ReadLine( FILE *fp, int size, char *buffer )
{
	int i;
	for ( i=0; i<size; i++ ) {
		buffer[i] = fgetc(fp);
		if ( feof(fp) || buffer[i] == '\n' || buffer[i] == '\r' ) {
			buffer[i] = '\0';
			return i+1;
		}
	}
	return i;
}

//-------------------------------------------------------------------------------

namespace cy {
	typedef cyTriMesh TriMesh;
}

//-------------------------------------------------------------------------------

#endif

