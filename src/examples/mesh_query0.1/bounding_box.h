#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "vec.h"

// An axis-aligned bounding box, represented with two vectors---the minimum
// coordinate corner and the maximum coordinate corner.
struct BoundingBox
{
   Vec3d xmin, xmax;

   // the default is a zero-extent box at the origin
   BoundingBox(void) : xmin(0,0,0), xmax(0,0,0) {}

   // or if you give it a single point, a zero-extent box at the point
   BoundingBox(const Vec3d& x) : xmin(x), xmax(x) {}

   // if you construct it with two vectors, it assumes they are the min
   // and the max corners
   BoundingBox(const Vec3d& xmin_,
               const Vec3d& xmax_)
      : xmin(xmin_), xmax(xmax_)
   { assert(xmin[0]<=xmax[0] && xmin[1]<=xmax[1] && xmin[2]<=xmax[2]); }

   // you can also build a box from a list of three points, finding the
   // min and max for each axis.
   void
   build_from_points(const Vec3d& x0,
                     const Vec3d& x1,
                     const Vec3d& x2)
   { minmax(x0, x1, x2, xmin, xmax); }

   bool
   contains(const Vec3d& x) const
   {
      return xmin[0]<=x[0] && x[0]<=xmax[0]
          && xmin[1]<=x[1] && x[1]<=xmax[1]
          && xmin[2]<=x[2] && x[2]<=xmax[2];
   }

   // you can enlarge an existing box to include a new point (if it doesn't
   // contain it already)
   void
   include_point(const Vec3d& x)
   { update_minmax(x, xmin, xmax); }

   // or you can even enlarge it to include another box (if it doesn't contain
   // it already)
   void
   include_box(const BoundingBox& box)
   {
      for(unsigned int i=0; i<3; ++i){
         if(box.xmin[i]<xmin[i]) xmin[i]=box.xmin[i];
         if(box.xmax[i]>xmax[i]) xmax[i]=box.xmax[i];
      }
   }

   // increases the bounding box by a factor of 1+epsilon in each axis:
   // useful for dealing with rounding errors!
   void
   enlarge(double epsilon)
   {
      double a=(epsilon/2)*(xmax[0]-xmin[0]);
      xmin[0]-=a; xmax[0]+=a;
      double b=(epsilon/2)*(xmax[1]-xmin[1]);
      xmin[1]-=b; xmax[1]+=b;
      double c=(epsilon/2)*(xmax[2]-xmin[2]);
      xmin[2]-=c; xmax[2]+=c;
   }

   Vec3d
   world_space(double fraction0,
               double fraction1,
               double fraction2)
   {
      return Vec3d((1-fraction0)*xmin[0]+fraction0*xmax[0],
                   (1-fraction1)*xmin[1]+fraction1*xmax[1],
                   (1-fraction2)*xmin[2]+fraction2*xmax[2]);
   }
};

inline BoundingBox
box_intersection(const BoundingBox& a,
                 const BoundingBox& b)
{
   return BoundingBox(max_union(a.xmin, b.xmin), min_union(a.xmax, b.xmax));
}

// an IO operator so you can more easily dump out a box when debugging
inline std::ostream&
operator<<(std::ostream &out,
           const BoundingBox& box)
{
   out<<"("<<box.xmin<<" ~ "<<box.xmax<<")";
   return out;
}

#endif
