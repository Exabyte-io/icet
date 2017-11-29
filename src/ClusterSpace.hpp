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
#include "PeriodicTable.hpp"
using namespace Eigen;

/**
This is the cluster space object. 

It will have the definition of the cluster space a cluster expansion is based on.

*/

class ClusterSpace
{
  public:
    ClusterSpace(std::vector<int> Mi, std::vector<std::string> elements, const OrbitList primOrbitList)
    {
        _Mi = Mi;        
        _primitive_orbitlist = primOrbitList;        
        _primitive_structure = primOrbitList.getPrimitiveStructure();
        _primitive_structure.setNumberOfAllowedComponents(_Mi);        
        initElementMap(elements);
        _isClusterspaceInitialized= false;
    };

    void initElementMap(std::vector<std::string> elements)
    {
        // std::sort(elements.begin(), elements.end());
        std::vector<int> intElements;
        for (const auto el : elements)
        {
            intElements.push_back(PeriodicTable::strInt[el]);
        }
        std::sort(intElements.begin(), intElements.end());
        
        for (size_t i = 0; i < elements.size(); i++)
        {
            _elementRepresentation[intElements[i]] = i;
        }

        _elements = intElements;
    }

    ///Generate the clustervector on the input structure
    std::vector<double> generateClustervector(const Structure &) const;

    ///Return the full cluster product of entire cluster (elements vector). Assuming all sites have same Mi
    double getClusterProduct(const std::vector<int> &mcVector, const std::vector<int> &Mi, const std::vector<int> &elements) const;

    ///setup  _clusterSpaceInfo
    void setupClusterspaceInfo();

    ///Returns clusterspace information (orbit index and mc vector)
    std::pair<int, std::vector<int>> getClusterSpaceInfo(const unsigned int);
        
    ///Gets the clusterspace size, i.e. the length of a clustervector
    size_t getClusterSpaceSize();

    ///Returns the cutoffs
    std::vector<double> getCutoffs() const
    {
        return _clusterCutoffs;
    }

    ///Get elements in str format
    std::vector<std::string> getAtomicNumbers() const
    {
        std::vector<std::string> elements;
        for(const auto &intEl : _elements )
        {
            elements.push_back(PeriodicTable::intStr[intEl]);
        }
        return elements;
    }
    ///returns a orbit from the orbitlist
    Orbit getOrbit(const size_t index) const
    {
        return _primitive_orbitlist.getOrbit(index);
    }

    OrbitList getOrbitList() const
    {
        return _primitive_orbitlist;
    }   

    ///returns the primitive structure    
    Structure getPrimitiveStructure() const
    {
        return _primitive_structure;
    }

    ClusterCounts getNativeClusters(const Structure &structure) const;
    
  private:    
    ///Currently we have constant Mi for development but will later change to dynamic Mi
    std::vector<int> _Mi;

    ///Primitive cell/structure
    Structure _primitive_structure;

    ///Primitive orbitlist based on the structure and the global cutoffs
    OrbitList _primitive_orbitlist;

    ///Unique id for this clusterspace
    int clusterSpace_ID;

    ///The radial cutoffs for neigbhor inclusion. First element in vector is pairs, then triplets etc.
    std::vector<double> _clusterCutoffs;

    ///Elements considered in this clusterspace (The integers represent their order in the periodic table)
    std::vector<int> _elements;

    ///Cluster space information, the vector is over all dimension of the clusterspace. the first int is the orbit index, the vector of ints are MC vectors
    std::vector<std::pair<int, std::vector<int>>> _clusterSpaceInfo;

    ///a boolean to keep track if this is initialized or not
    bool _isClusterspaceInitialized;


    /**
    This maps a orbit to other orbits i.e. of _equalSitesList[0] = [1,2,3] then orbit zero should be seen as equal to orbit 1,2 and three.
    This will mean that when retrieving the cluster vector the zeroth element will be a combination of orbit 0,1,2 and 3.
    */
    std::map<int, std::vector<int>> _equalOrbitsList;

    ///return the default clusterfunction of the input element when this site can have Mi elements
    double defaultClusterFunction(const int Mi, const int clusterFunction, const int element) const;

    //////Returns the allowed occupations on the sites
    std::vector<int> getAllowedOccupations(const Structure &structure, const std::vector<LatticeSite> &latticeNeighbors) const;

    //Maps real elements  to a 0,1,2, ... representation
    std::map<int, int> _elementRepresentation;
};
