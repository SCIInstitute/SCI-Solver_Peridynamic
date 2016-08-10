#ifndef BOUNDING_BOX_TREE_H
#define BOUNDING_BOX_TREE_H

// This provides a bounding volume hierarchy - in particular, a tree of nested
// axis-aligned bounding boxes around some given geometry.
// It has a construct_from_leaf_boxes() function which takes as input an array
// of bounding boxes for the leaves of the tree (it's up to you, the user, to
// provide those) and uses a heuristic method discussed in class to build the
// hierarchy around those.
//
// An object of type BoundingBoxTree represents one node of the tree, whether
// root or leaf or something in between. Every node has a bounding box ("box")
// which is supposed to contain all the geometry in it and its children. Every
// node has an array "index" of integers labeling the input boxes that the user
// specified; probably only leaves of the tree will have non-empty "index"
// arrays, but don't count on it. Every node also has an array "children" of
// pointers to the children nodes of this box, which would be empty for a leaf.

#include <vector>
#include "bounding_box.h"

struct BoundingBoxTree
{
    BoundingBox box;
    std::vector<int> index;
    std::vector<BoundingBoxTree*> children;

    BoundingBoxTree(void) {}

    virtual
    ~BoundingBoxTree(void)
    { clear(); }

    void
    clear(void)
    {
        index.clear();
        for(unsigned int i=0; i<children.size(); ++i) delete children[i];
        children.clear();
    }

    void
    construct_from_leaf_boxes(unsigned int n,
                              const BoundingBox* input_box,
                              unsigned int max_depth=30,
                              unsigned int boxes_per_leaf=1);

    protected:
    void
    construct_recursively(unsigned int n,
                          const BoundingBox* input_box,
                          unsigned int max_depth,
                          unsigned int boxes_per_leaf,
                          unsigned int current_depth,
                          unsigned int num_indices,
                          unsigned int* local_index,
                          Vec3d* local_centroid);

    private:
    // since this class allocates its own storage (children nodes)
    // we can't allow the default copy constructor or assignment operator
    BoundingBoxTree(const BoundingBoxTree& other_tree)
    { assert(false); }

    BoundingBoxTree& operator=(const BoundingBoxTree& other_tree)
    { assert(false); return *this; }
};

#endif
