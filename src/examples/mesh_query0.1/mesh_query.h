#ifndef MESH_QUERY_H
#define MESH_QUERY_H

// Although a plain C API is given, it requires the standard C++ library.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MeshObject MeshObject;

// Returns a pointer to a newly constructed mesh object, given
// the number of vertices, a pointer to the array of vertex coordinates
// (three doubles per vertex), the number of triangles, and a pointer to the
// array of triangle vertex indices (three ints per triangle).
// Note that the vertex and triangle data is not copied, so it is up to the
// user to preserve the arrays over the lifetime of the mesh object.
MeshObject*
construct_mesh_object(int num_vertices,
                      const double *positions,
                      int num_triangles,
                      const int *triangles);

// Cleans up a previously allocated mesh object.
void
destroy_mesh_object(MeshObject *mesh);

// Checks if the given point is inside the mesh, assuming the mesh specified
// was watertight. If the point is exactly on the mesh, the result might be
// labeled either inside or outside; otherwise this is an exact test.
bool
point_inside_mesh(const double point[3],
                  const MeshObject *mesh);

// Checks if the segment from point0 to point1 intersects some triangle
// in the mesh object; if so, the index of the first intersecting triangle
// found is set, along with barycentric coordinates of the intersection point:
//   s*point0+t*point1 = a*vertex0+b*vertex1+c*vertex2
// with s+t = a+b+c = 1, and vertex012 being the vertices of the indicated
// triangle. The barycentric coordinates are approximate, but the boolean
// return value is exact. Really degenerate cases are not counted (i.e. if
// the segment is really a point, if the segment lies in the plane of the
// triangle, or the triangle is degenerate), but mildly strange cases (i.e.
// an endpoint of a good segment lying on a good triangle, or a good segment
// passing through an edge or vertex of a good triangle) are counted as
// intersections.
bool
segment_intersects_mesh(const double point0[3],
                        const double point1[3],
                        const MeshObject *mesh,
                        int *triangle_index,
                        double *s,
                        double *t,
                        double *a,
                        double *b,
                        double *c);

#ifdef __cplusplus
}; // extern "C"
#endif

#endif
