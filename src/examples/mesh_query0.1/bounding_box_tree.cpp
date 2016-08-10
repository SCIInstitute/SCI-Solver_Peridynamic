#include "bounding_box_tree.h"

void BoundingBoxTree::
construct_from_leaf_boxes(unsigned int n,
                          const BoundingBox* input_box,
                          unsigned int max_depth,
                          unsigned int boxes_per_leaf)
{
   assert(n>=1 && input_box && boxes_per_leaf>0);
   clear();
   // Set up a list of all point indices (initially in order 0, 1, ..., n-1)
   // and box centroids.
   unsigned int *global_index=new unsigned int[n];
   Vec3d* global_centroid=new Vec3d[n];
   for(unsigned int i=0; i<n; ++i){
      global_index[i]=i;
      global_centroid[i]=0.5*(input_box[i].xmin+input_box[i].xmax);
   }
   // Now hand this off to the recursive function which does the actual work
   construct_recursively(n, input_box, max_depth, boxes_per_leaf, 0, n,
                         global_index, global_centroid);
   // and clean up after ourselves!
   delete[] global_index;
   delete[] global_centroid;
}

// The idea here is that local_index is a pointer to the start of an array of
// num_indices integers (possibly a subset of a larger array) which is what
// this node is going to contain. The function decides if this where we should
// stop, making this node a leaf, or if it should instead partition the indices
// into two children split along an axis. To make the partitioning easy and
// efficient, it actually modifies the local_index array to have one child's
// indices all at the start and the other's indices after that kind of like
// quicksort.
void BoundingBoxTree::
construct_recursively(unsigned int n,
                      const BoundingBox* input_box,
                      unsigned int max_depth,
                      unsigned int boxes_per_leaf,
                      unsigned int current_depth,
                      unsigned int num_indices,
                      unsigned int* local_index,
                      Vec3d* centroid)
{
   assert(num_indices>0);
   // Find bounding box for everything contained in this node.
   box=input_box[local_index[0]];
   for(unsigned int i=1; i<num_indices; ++i)
      box.include_box(input_box[local_index[i]]);

   // Check for base case or recurse.
   if(current_depth==max_depth || num_indices<=boxes_per_leaf){
      // Base case of recursion (keep *this as a leaf, with the given points).
      index.resize(num_indices);
      for(unsigned int i=0; i<num_indices; ++i) index[i]=local_index[i];

   }else{ // See if we can split up the indices between two child nodes.
      // Figure out which axis to split along - from bounding box of centroids
      BoundingBox centroid_box(centroid[local_index[0]]);
      Vec3d mean(centroid[local_index[0]]);
      for(unsigned int i=1; i<num_indices; ++i){
         centroid_box.include_point(centroid[local_index[i]]);
         mean+=centroid[local_index[i]];
      }
      mean*=1./num_indices;
      // check if all the centroids were equal
      if(centroid_box.xmax==centroid_box.xmin){
         // leave this is as a leaf, since we can't tell the centroids apart
         index.resize(num_indices);
         for(unsigned int i=0; i<num_indices; ++i) index[i]=local_index[i];

      }else{
         // otherwise, split along a selected axis
         int axis=0;
         for(unsigned int a=1; a<3; ++a){
            if(centroid_box.xmax[a]-centroid_box.xmin[a]
                  > centroid_box.xmax[axis]-centroid_box.xmin[axis])
               axis=a;
         }
         // Pick mean as a splitting value.
         // Note, since axis bounds were tight, there must be at least one
         // point on either side of split.
         double split=mean[axis];
         if(split<=centroid_box.xmin[axis] || split>=centroid_box.xmax[axis])
            split=0.5*(centroid_box.xmin[axis]+(double)centroid_box.xmax[axis]);
         // Do the quicksort-style in-place split of the indices
         int i=0, j=num_indices-1;
         for(;;){
            while((unsigned int)i<num_indices
                  && centroid[local_index[i]][axis]<=split)
               ++i;
            while(j>=0 && centroid[local_index[j]][axis]>split)
               --j;
            if(i<j){
               std::swap(local_index[i], local_index[j]);
               ++i;
               --j;
            }else{
               break;
            }
         }
         assert((unsigned int)i<num_indices);
         // now indices[0] to indices[i-1] inclusive should all be
         // located <=split (i boxes) and indices[i] to indices[num_indices-1]
         // inclusive should be located >split (num_indices-i boxes) 
         children.resize(2,0);
         children[0]=new BoundingBoxTree;
         children[0]->construct_recursively(n, input_box, max_depth,
                                            boxes_per_leaf, current_depth+1,
                                            i, local_index, centroid);
         children[1]=new BoundingBoxTree;
         children[1]->construct_recursively(n, input_box, max_depth,
                                            boxes_per_leaf, current_depth+1,
                                            num_indices-i, local_index+i,
                                            centroid);
      }
   }
}
