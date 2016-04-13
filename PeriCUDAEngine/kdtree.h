//------------------------------------------------------------------------------------------
//
//
// Created on: 2/4/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#include "definitions.h"
#include "stdint.h"

#ifndef KDTREE
#define KDTREE


// Start KD.  From here down to End KD is all the machinery needed to avoid
// an N^2 search for all particles within distance delta of each particle.
class Point
{
public:
    Point();
    Point(real_t, real_t, real_t, int);
    real_t operator[](int);
    real_t dist_pt_to_box(Point, Point);
    void dump();
    int get_index()
    {
        return index;
    }
private:
    real_t x, y, z;
    int index;
};

Point::Point() :
    x(0.0), y(0.0), z(0.0), index(-1)
{
}
Point::Point(real_t newx, real_t newy, real_t newz, int newi) :
    x(newx), y(newy), z(newz), index(newi)
{
}

real_t Point::operator[](int comp)
{
    switch (comp)
    {
    case 0:
        return x;

    case 1:
        return y;

    case 2:
        return z;
    }

    return 0.0;
}

void Point::dump()
{
    printf("%d %f %f %f\n", index, x, y, z);
}

real_t dist(Point a, Point b)
{
    real_t howfar;
    int j;

    howfar = 0.0;

    for (j = 0; j < 3; j++)
    {
        howfar += (a[j] - b[j]) * (a[j] - b[j]);
    }

    return (sqrt(howfar));
}

// distance from point (class data) to a box; 0 if inside
real_t Point::dist_pt_to_box(Point m, Point M)
{
// start with two copies of class values
    Point target(x, y, z, index), hit(x, y, z, index);

    if (target.x < m.x)
    {
        hit.x = m.x;
    }
    else if (target.x > M.x)
    {
        hit.x = M.x;
    }

    if (target.y < m.y)
    {
        hit.y = m.y;
    }
    else if (target.y > M.y)
    {
        hit.y = M.y;
    }

    if (target.z < m.z)
    {
        hit.z = m.z;
    }
    else if (target.z > M.z)
    {
        hit.z = M.z;
    }

    return (dist(hit, target));
}

class KDtree;
class KDNode
{
public:
    KDNode(Point*, int, Point, Point);
    friend class KDtree;
private:
    KDNode* left;
    KDNode* right;
    int axis;		// 0==x, 1==y, 2==z
    real_t split;
    int isleaf;
    Point* point;
    int count;
    Point box_m, box_M;	// corners of bounding box for this node
};

#define NodeNULL (KDNode *)(0)
KDNode::KDNode(Point* newp, int newcount, Point nbox_m, Point nbox_M) :
    left(NodeNULL), right(NodeNULL), axis(-1), split(0.0), isleaf(1), point(
        newp), count(newcount), box_m(nbox_m), box_M(nbox_M)
{
}

class KDtree
{
public:
    KDtree(int _numParticles, KDNode* firstnode, int ipn, int* _bondList, int* _bondCount,
           real_t _pd_horizon);
    void buildtree(KDNode*);
    void printtree(KDNode*);
    void find_neighbors(Point, KDNode*);
    void add_bond(int p, int q, real_t r);
private:
    KDNode* root;
    int max_items_per_node;
    int* bond_list;
    int* bond_list_top;
    int numParticles;
    real_t pd_horizon;
};


KDtree::KDtree(int _numParticles, KDNode* firstnode, int ipn, int* _bondList, int* _bondCount,
               real_t _pd_horizon):
    numParticles(_numParticles), root(firstnode), max_items_per_node(ipn), bond_list(_bondList), bond_list_top(_bondCount),
    pd_horizon(_pd_horizon)
{
}

// find median using component axis of array of points, point[size]
real_t get_median(Point* point, int size, int axis)
{
    int left, right, i, j, k;
    Point hold, key;

    if (size % 2 == 0)
    {
        k = size / 2 - 1;
    }
    else
    {
        k = size / 2;
    }

    left = 0;
    right = size - 1;

    while (left < right)
    {
        key = point[k];
        i = left;
        j = right;

        do
        {
            while (point[i][axis] < key[axis])
            {
                i++;
            }

            while (point[j][axis] > key[axis])
            {
                j--;
            }

            if (i <= j)
            {
                // swap large i with small j
                hold = point[i];
                point[i] = point[j];
                point[j] = hold;
                i++;
                j--;
            }
        }
        while (i <= j);

        if (j < k)
        {
            left = i;
        }

        if (k < i)
        {
            right = j;
        }
    }

    return (point[k][axis]);
}

void KDtree::buildtree(KDNode* treeptr)
{
    int longaxis, lc;
    real_t maxlength, length, median, upcorner[3], dncorner[3];
    Point* leftarray, *rightarray, thispt;
    int leftarray_count, j;

    if ((treeptr)->isleaf)
    {
        lc = (treeptr)->count;

        // this is the only way out
        if (lc <= max_items_per_node)
        {
            return;
        }

        maxlength = 0;

        for (j = 0; j < 3; j++)
        {
            upcorner[j] = treeptr->box_M[j];
            dncorner[j] = treeptr->box_m[j];

            if ((length = (upcorner[j] - dncorner[j])) > maxlength)
            {
                longaxis = j;
                maxlength = length;
            }
        }

        median = get_median((treeptr)->point, lc, longaxis);
        (treeptr)->split = median;
        (treeptr)->axis = longaxis;
        (treeptr)->isleaf = 0;

        if (lc % 2 == 0)
        {
            leftarray_count = lc / 2;
        }
        else
        {
            leftarray_count = lc / 2 + 1;
        }

        leftarray = treeptr->point;
        rightarray = &(treeptr->point[leftarray_count]);
        upcorner[longaxis] = dncorner[longaxis] = median;
        (treeptr)->left = new KDNode(&leftarray[0], leftarray_count,
                                     treeptr->box_m,
                                     Point(upcorner[0], upcorner[1], upcorner[2], -1));
        (treeptr)->right = new KDNode(&rightarray[0], lc / 2,
                                      Point(dncorner[0], dncorner[1], dncorner[2], -1),
                                      treeptr->box_M);
    }

    buildtree((treeptr)->left);
    buildtree((treeptr)->right);
    return;
}

void KDtree::printtree(KDNode* root)
{
    if (root->isleaf)
    {
        printf("found leaf node with count %d\n", root->count);
        return;
    }

    printf("interior node; split value %f on axis %d\n", root->split,
           root->axis);
    printtree(root->left);
    printtree(root->right);
}

void KDtree::find_neighbors(Point target, KDNode* treeptr)
{
    int i, j, k;
    real_t r;

    if (target.dist_pt_to_box(treeptr->box_m, treeptr->box_M) > pd_horizon)
    {
        return;
    }

    if (treeptr->isleaf)
    {
        j = target.get_index();
        for (i = 0; i < treeptr->count; i++)
        {
            r = dist(target, treeptr->point[i]);
            if (r < pd_horizon)
            {
                k = treeptr->point[i].get_index();

                if (j != k)
                {
                    add_bond(j, k, r);
                }
            }
        }
        return;
    }

    find_neighbors(target, treeptr->left);
    find_neighbors(target, treeptr->right);
    return;
}

void KDtree::add_bond(int p, int q, real_t r)
{
    bond_list_top[p]++;

    if(bond_list_top[p] >= MAX_PD_BOND_COUNT)
    {
        bond_list_top[p]--;
        printf("WARNINGGGGGGG: Number of bonds exceeds MAX_PD_BOND_COUNT = %d\n", MAX_PD_BOND_COUNT);
    }
    else
    {
        bond_list[bond_list_top[p] * numParticles + p] = q;
//        printf("bond: %d-%d\n", p, q);
//        fflush(stdout);
    }
}

#endif // KDTREE

