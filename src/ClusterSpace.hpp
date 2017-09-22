#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <string>
#include <math.h>
#include "Structure.hpp"
#include "OrbitList.hpp"
#include "LocalOrbitlistGenerator.hpp"
#include "ClusterCounts.hpp"
using namespace Eigen;


/**
This is the cluster space object. 

It will have the definition of the cluster space a cluster expansion is based on.

*/

class ClusterSpace
{
public:
ClusterSpace(int Mi, const OrbitList primOrbitList)
{
    _Mi = Mi;
    _primitive_orbitlist = primOrbitList;
};


///Generate the clustervector on the input structure
std::vector<double> generateClustervector(const Structure &) const;

private:

///Currently we have constant Mi for development but will later change to dynamic Mi 
int _Mi;

///Primitive cell/structure    
Structure _primitive_structure;

///Primitive orbitlist based on the structure and the global cutoffs
OrbitList _primitive_orbitlist;

///Unique id for this clusterspace
int clusterSpace_ID;

///The radial cutoffs for neigbhor inclusion. First element in vector is pairs, then triplets etc.
std::vector<double> _clusterCutoffs;

/**
This maps a orbit to other orbits i.e. of _equalSitesList[0] = [1,2,3] then orbit zero should be seen as equal to orbit 1,2 and three.
This will mean that when retrieving the cluster vector the zeroth element will be a combination of orbit 0,1,2 and 3.
*/
std::map<int,std::vector<int>> _equalOrbitsList;

//std::map<std::pair<int,int>, double> _defaultClusterFunction; 

///return the default clusterfunction of the input element when this site can have Mi elements
double defaultClusterFunction(const int Mi, const int element) const;

///Return the full cluster product of entire cluster (elements vector). Assuming all sites have same Mi
double getClusterProduct(const int Mi, const std::vector<int> &elements) const;
};