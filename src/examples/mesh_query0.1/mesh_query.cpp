#include <cfloat>
#include "mesh_query.h"
#include "bounding_box_tree.h"
#include "predicates.h"

//==============================================================================
// helper functions for ray casts along positive z axis

// returns true if the ray cast from p along positive z axis hits box
static bool
box_zcast(const Vec3d& p,
          const BoundingBox& box)
{
    return p[0]>=box.xmin[0] && p[0]<=box.xmax[0]
        && p[1]>=box.xmin[1] && p[1]<=box.xmax[1]
        && p[2]<=box.xmax[2];
}

// helper function for tri_zcast below...
// Here we are given a robust 2d orientation of qrs in xy projection, which
// must be positive.
static bool
tri_zcast_inner(const Vec3d& p,
                double qrs,
                const Vec3d& q,
                const Vec3d& r,
                const Vec3d& s)
{
    assert(qrs>0);

    // first check if point is above or below triangle in z
    double pqrs=orient3d(p.v, q.v, r.v, s.v);
    if(pqrs>=0) return false; // point is on or above triangle - no intersection

    // then check if point lies outside triangle in 2D xy projection
    double pqr=orient2d(p.v, q.v, r.v);
    if(pqr<0) return false;
    double prs=orient2d(p.v, r.v, s.v);
    if(prs<0) return false;
    double psq=orient2d(p.v, s.v, q.v);
    if(psq<0) return false;

    // note: the following tests are somewhat redundant, but it's a pretty
    // tiny optimization to eliminate the redundancy compared to the loss in
    // clarity.

    // check if point is strictly inside the triangle in xy
    if(pqr>0 && prs>0 && psq>0) return true;

    // check if point is strictly on edge qr
    if(pqr==0 && prs>0 && psq>0){
        if(q[1]<r[1]) return false;
        if(q[1]>r[1]) return true;
        if(q[0]<r[0]) return true;
        assert(q[0]>r[0]); // q!=r because triangle is not degenerate
        return false;
    }

    // check if point is strictly on edge rs
    if(prs==0 && pqr>0 && psq>0){
        if(r[1]<s[1]) return false;
        if(r[1]>s[1]) return true;
        if(r[0]<s[0]) return true;
        assert(r[0]>s[0]); // r!=s because triangle is not degenerate
        return false;
    }
    
    // check if point is strictly on edge sq
    if(psq==0 && pqr>0 && prs>0){
        if(s[1]<q[1]) return false;
        if(s[1]>q[1]) return true;
        if(s[0]<q[0]) return true;
        assert(s[0]>q[0]); // r!=s because triangle is not degenerate
        return false;
    }
   
    // check if point is on vertex q
    if(p[0]==q[0] && p[1]==q[1]){
        return q[1]>=r[1] && q[1]<s[1];
    }
   
    // check if point is on vertex r
    if(p[0]==r[0] && p[1]==r[1]){
        return r[1]>=s[1] && r[1]<q[1];
    }
   
    // check if point is on vertex s
    if(p[0]==s[0] && p[1]==s[1]){
        return s[1]>=q[1] && s[1]<r[1];
    }

    assert(false); // we should have covered all cases at this point
    return false; // just to quiet compiler warnings
}

// returns true if the ray cast from p along positive z axis hits triangle in
// exactly one spot, with edge cases handled appropriately
static bool
tri_zcast(const Vec3d& p,
          const Vec3d& q,
          const Vec3d& r,
          const Vec3d& s)
{
    // robustly find orientation of qrs in 2D xy projection
    double qrs=orient2d(q.v, r.v, s.v);
    if(qrs>0)
        return tri_zcast_inner(p, qrs, q, r, s);
    else if(qrs<0)
        return tri_zcast_inner(p, -qrs, q, s, r); // flip triangle to reorient
    else
        return false; // triangle is degenerate in 2D projection - ignore
}

//==============================================================================
// helper functions for segment intersecting triangles

// returns false only if the segment for sure can't intersect the box
static bool
segment_box_intersect(const Vec3d& p,
                      const Vec3d& q,
                      const BoundingBox& box)
{
    // these are conservative bounds on error factor from rounding
    const double lo=1-5*DBL_EPSILON, hi=1+5*DBL_EPSILON;
    double s=0, t=1; // bounds on parameter for intersection interval
    for(unsigned int i=0; i<3; ++i){
        if(p[i]<q[i]){
            double d=q[i]-p[i];
            double s0=lo*(box.xmin[i]-p[i])/d, t0=hi*(box.xmax[i]-p[i])/d;
            if(s0>s) s=s0;
            if(t0<t) t=t0;
        }else if(p[i]>q[i]){
            double d=q[i]-p[i];
            double s0=lo*(box.xmax[i]-p[i])/d, t0=hi*(box.xmin[i]-p[i])/d;
            if(s0>s) s=s0;
            if(t0<t) t=t0;
        }else{
            if(p[i]<box.xmin[i] || p[i]>box.xmax[i]) return false;
        }
        if(s>t) return false;
    }
    return true;
}

// determine if segment pq intersects triangle uvw, setting approximate
// barycentric coordinates if so.
static bool
segment_tri_intersect(const Vec3d& p,
                      const Vec3d& q,
                      const Vec3d& u,
                      const Vec3d& v,
                      const Vec3d& w,
                      double* s,
                      double* t,
                      double* a,
                      double* b,
                      double* c)
{
    // find where segment hits plane of triangle
    double puvw=orient3d(p.v, u.v, v.v, w.v),
           uvwq=orient3d(u.v, v.v, w.v, q.v);
    if((puvw<=0 && uvwq>=0) || (puvw>=0 && uvwq<=0))
        return false; // either no intersection, or a degenerate one
    if(puvw<0 || uvwq<0){
        double pqvw=orient3d(p.v, q.v, v.v, w.v);
        if(pqvw>0) return false;
        double puqw=orient3d(p.v, u.v, q.v, w.v);
        if(puqw>0) return false;
        double puvq=orient3d(p.v, u.v, v.v, q.v);
        if(puvq>0) return false;
        *s=uvwq/(puvw+uvwq);
        *t=puvw/(puvw+uvwq);
        *a=pqvw/(pqvw+puqw+puvq);
        *b=puqw/(pqvw+puqw+puvq);
        *c=puvq/(pqvw+puqw+puvq);
        return true;
    }else{ //(puvw>0 || uvwq>0)
        double pqvw=orient3d(p.v, q.v, v.v, w.v);
        if(pqvw<0) return false;
        double puqw=orient3d(p.v, u.v, q.v, w.v);
        if(puqw<0) return false;
        double puvq=orient3d(p.v, u.v, v.v, q.v);
        if(puvq<0) return false;
        *s=uvwq/(puvw+uvwq);
        *t=puvw/(puvw+uvwq);
        *a=pqvw/(pqvw+puqw+puvq);
        *b=puqw/(pqvw+puqw+puvq);
        *c=puvq/(pqvw+puqw+puvq);
        return true;
    }
}

//==============================================================================
// the actual accelerated mesh class

struct MeshObject
{
    int n;
    const Vec3d *x;
    int nt;
    const Vec3i *tri;
    BoundingBoxTree tree;

    MeshObject(int n_,
               const double *x_,
               int nt_,
               const int *tri_)
        : n(n_), x((const Vec3d*)x_), nt(nt_), tri((const Vec3i*)tri_)
    {
        assert(x && tri && n>=0 && nt>=0);
        if(nt==0) return;
        std::vector<BoundingBox> box(nt);
        for(int t=0; t<nt; ++t){
            int i, j, k; assign(tri[t], i, j, k);
            box[t].build_from_points(x[i], x[j], x[k]);
        }
        tree.construct_from_leaf_boxes(nt, &box[0]);
    }

    bool
    inside(const double point[3]) const
    {
        Vec3d p(point);
        // quick check on root bounding box
        if(!tree.box.contains(p)) return false;
        // count intersections along a ray-cast, check parity for inside/outside
        int intersection_count=0;
        // we cast ray along positive z axis
        std::vector<const BoundingBoxTree*> stack;
        stack.push_back(&tree);
        while(!stack.empty()){
            const BoundingBoxTree *node=stack.back();
            stack.pop_back();
            // check any triangles in this node
            for(unsigned int i=0; i<node->index.size(); ++i){
                int t=node->index[i];
                if(tri_zcast(p, x[tri[t][0]], x[tri[t][1]], x[tri[t][2]]))
                    ++intersection_count;
            }
            // check any subtrees for this node
            for(unsigned int i=0; i<node->children.size(); ++i){
                if(box_zcast(p, node->children[i]->box))
                    stack.push_back(node->children[i]);
            }
        }
        return intersection_count%2;
    }

    bool
    intersects(const Vec3d &p,
               const Vec3d &q,
               int *triangle_index,
               double *s,
               double *t,
               double *a,
               double *b,
               double *c) const
    {
        std::vector<const BoundingBoxTree*> stack;
        stack.push_back(&tree);
        while(!stack.empty()){
            const BoundingBoxTree *node=stack.back();
            stack.pop_back();
            if(!segment_box_intersect(p, q, node->box))
                continue; // no need to go further with this node
            // check any triangles in this node
            for(unsigned int i=0; i<node->index.size(); ++i){
                int u, v, w; assign(tri[node->index[i]], u, v, w);
                if(segment_tri_intersect(p, q, x[u], x[v], x[w],
                                         s, t, a, b, c)){
                    *triangle_index=node->index[i];
                    return true;
                }
            }
            // check any subtrees for this node
            for(unsigned int i=0; i<node->children.size(); ++i)
                stack.push_back(node->children[i]);
        }
        return false;
    }
};

//==============================================================================
// the plain C wrapper API

MeshObject*
construct_mesh_object(int num_vertices,
                      const double *positions,
                      int num_triangles,
                      const int *triangles)
{
    exactinit(); // really only need to call this once, but safe to call always
    return new MeshObject(num_vertices, positions, num_triangles, triangles);
}

void
destroy_mesh_object(MeshObject *mesh)
{
    delete mesh;
}

bool
point_inside_mesh(const double point[3],
                  const MeshObject *mesh)
{
    return mesh->inside(point);
}

bool
segment_intersects_mesh(const double point0[3],
                        const double point1[3],
                        const MeshObject *mesh,
                        int *triangle_index,
                        double *s,
                        double *t,
                        double *a,
                        double *b,
                        double *c)
{
    return mesh->intersects(Vec3d(point0), Vec3d(point1),
                            triangle_index, s, t, a, b, c);
}
