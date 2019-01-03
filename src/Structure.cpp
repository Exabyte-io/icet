#include "Structure.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <string>

using namespace Eigen;

/**
  @details Initializes an icet Structure instance.
  @param positions list of positions in Cartesian coordinates
  @param chemicalSymbols list of chemical symbols
  @param cell cell metric
  @param pbc periodic boundary conditions
  @param tolerance numerical tolerance imposed when testing for equality of positions and distances
**/
Structure::Structure(const Eigen::Matrix<double, Dynamic, 3, RowMajor> &positions,
                     const std::vector<std::string> &chemicalSymbols,
                     const Eigen::Matrix3d &cell,
                     const std::vector<bool> &pbc,
                     double tolerance)
{
    setPositions(positions);
    setChemicalSymbols(chemicalSymbols);
    _cell = cell;
    _pbc = pbc;
    _tolerance = tolerance;
    _uniqueSites.resize(chemicalSymbols.size());
    _numbersOfAllowedSpecies.resize(positions.rows());
}

/**
  @details This function computes the distance between two sites.
  @param index1 index of the first site
  @param index2 index of the second site
  @param offset1 offset of site 1 relative to origin in units of lattice vectors
  @param offset2 offset of site 2 relative to origin in units of lattice vectors
  @todo Add  mic functionality
*/
double Structure::getDistance(const size_t index1, const size_t index2,
                              const Vector3d offset1 = {0.0, 0.0, 0.0},
                              const Vector3d offset2 = {0.0, 0.0, 0.0}) const
{
    if (index1 >= (size_t) _positions.rows() or
        index2 >= (size_t) _positions.rows())
    {
        std::string errorMessage = "At least one site index out of bounds ";
        errorMessage += " index1: " + std::to_string(index1);
        errorMessage += " index2: " + std::to_string(index2);
        errorMessage += " npositions: " + std::to_string(_positions.rows());
        errorMessage += " (Structure::getDistance)";
        throw std::out_of_range(errorMessage);
    }
    Vector3d pos1 = _positions.row(index1) + offset1.transpose() * _cell;
    Vector3d pos2 = _positions.row(index2) + offset2.transpose() * _cell;
    return (pos1 - pos2).norm();
}

/**
  @details This function returns the position of a site.
  @param latticeNeighbor site for which to obtain the position
  @returns a 3-dimensional position vector
*/
Vector3d Structure::getPosition(const LatticeSite &latticeNeighbor) const
{
    if (latticeNeighbor.index() >= (size_t) _positions.rows())
    {
        std::string errorMessage = "Site index out of bounds";
        errorMessage += " index: " + std::to_string(latticeNeighbor.index());
        errorMessage += " number of positions: " + std::to_string(_positions.rows());
        errorMessage += " (Structure::getPosition)";
        throw std::out_of_range(errorMessage);
    }
    Vector3d position = _positions.row(latticeNeighbor.index()) + latticeNeighbor.unitcellOffset().transpose() * _cell;
    return position;
}
/**
@details This function returns the position of a specific site in Cartesian coordinates.
@param index index of the site
 **/
Vector3d Structure::getPositionByIndex(const size_t &index) const
{
    Vector3d position = _positions.row(index);
    return position;
}
/**
  @details This function returns the atomic number of a site.
  @param site index of site
  @returns atomic number
**/
int Structure::getAtomicNumber(const size_t index) const
{
    if (index >= _atomicNumbers.size())
    {
        std::string errorMessage = "Site index out of bounds";
        errorMessage += " index: " + std::to_string(index);
        errorMessage += " nsites: " + std::to_string(_atomicNumbers.size());
        errorMessage += " (Structure::getAtomicNumber)";
        throw std::out_of_range(errorMessage);
    }
    return _atomicNumbers[index];
}

/**
  @details This function sets the symmetrically distinct sites associated
  with the structure. It requires a vector as input the length of
  which  must match the number of positions.
  @param sites list of integers
**/
void Structure::setUniqueSites(const std::vector<size_t> &sites)
{
    if (sites.size() != (size_t) _positions.rows())
    {
        std::string errorMessage = "Length of input vector does not match number of sites";
        errorMessage += " nsites: " + std::to_string(sites.size());
        errorMessage += " npositions: " + std::to_string(_positions.rows());
        errorMessage += " (Structure::setUniqueSites)";
        throw std::out_of_range(errorMessage);
    }
    _uniqueSites = sites;
}

/**
  @details This function returns the index of a unique site from the list of unique sites.
  @param i index of site
  @returns index of unique site
**/
size_t Structure::getUniqueSite(const size_t i) const
{
    if (i >= _uniqueSites.size())
    {
        std::string errorMessage = "Site index out of bounds";
        errorMessage += " i: " + std::to_string(i);
        errorMessage += " nsites: " + std::to_string(_uniqueSites.size());
        errorMessage += " (Structure::getUniqueSite)";
        throw std::out_of_range(errorMessage);
    }
    return _uniqueSites[i];
}

/**
  @details This function returns the index of the site the position of
  which matches the input position to the tolerance specified for this
  structure.

  @param position position to match in Cartesian coordinates

  @returns index of site; -1 = failed to find a match.
**/
int Structure::findSiteByPosition(const Vector3d &position) const
{
    for (size_t i = 0; i < (size_t) _positions.rows(); i++)
    {
        if ((_positions.row(i).transpose() - position).norm() < _tolerance)
        {
            return i;
        }
    }

    return -1;
}

/**
  @details This function returns the LatticeSite object the position of
  which matches the input position to the tolerance specified for this
  structure.

  The algorithm commences by extracting the fractional position.
  From the fractional position the unitcell offset is taken by rounding the
  fractional coordinates to the nearest integer.
  By subtracting the unitcell offset from the fractional position and taking
  the dot product with the cell the position relative to the primitive cell is
  found.
  The index is found by searching for the remainder position in structure.
  If no index is found a runtime_error is thrown.

  @param position position to match in Cartesian coordinates

  @returns LatticeSite object
*/
LatticeSite Structure::findLatticeSiteByPosition(const Vector3d &position) const
{
    Vector3d fractional = _cell.transpose().partialPivLu().solve(position);

    Vector3d unitcellOffset;
    for (size_t i = 0; i < 3; i++)
    {
        if (hasPBC(i))
        {
            unitcellOffset[i] = floor(roundFloat((double)fractional[i]));
            if (fabs(unitcellOffset[i] - fractional[i]) > 1.0 - _tolerance)
            {
                unitcellOffset[i] = int(round(fractional[i]));
            }
        }
        else
        {
            unitcellOffset[i] = 0.0;
        }
    }
    Vector3d remainder = (fractional - unitcellOffset).transpose() * _cell;
    auto index = findSiteByPosition(remainder);
    if (index == -1)
    {
        std::string errorMessage = "Failed to find site by position (findLatticeSiteByPosition)";
        throw std::runtime_error(errorMessage);
    }

    LatticeSite ret = LatticeSite(index, unitcellOffset);
    return ret;
}

/**
  @details This function returns a list ofLatticeSite object the position
  of each matches the respective entry in the list of input positions to the
  tolerance specified for this structure. Internally this function uses
  Structure::findLatticeSiteByPosition.

  @param positions list of position to match in Cartesian coordinates

  @returns list of LatticeSite objects
*/
std::vector<LatticeSite> Structure::findLatticeSitesByPositions(const std::vector<Vector3d> &positions) const
{
    std::vector<LatticeSite> latNbrVector;
    latNbrVector.reserve(positions.size());

    for (const Vector3d position : positions)
    {
        latNbrVector.push_back(findLatticeSiteByPosition(position));
    }

    return latNbrVector;
}

/**
  @details This function allows one to specify the number of components
  that are allowed on each lattice site via a vector. This can be employed to
  construct "parallel" cluster expansions such as in (A,B) on site #1 with
  (C,D) on site #2.
  @param numbersOfAllowedSpecies list with the number of components
  allowed on each site
**/
void Structure::setNumberOfAllowedSpecies(const std::vector<int> &numbersOfAllowedSpecies)
{
    if (numbersOfAllowedSpecies.size() != size())
    {
        std::string errorMessage;
        errorMessage += "Size of input list incompatible with structure";
        errorMessage += " length: " + std::to_string(numbersOfAllowedSpecies.size());
        errorMessage += " nsites: " + std::to_string(size());
        errorMessage += " (Structure::setNumberOfAllowedSpecies)";
        throw std::out_of_range(errorMessage);
    }
    _numbersOfAllowedSpecies = numbersOfAllowedSpecies;
}

/**
  @details This function allows one to specify the number of components
  that are allowed on each lattice site via a scalar. This can be employed to
  construct "parallel" cluster expansions such as in (A,B) on site #1 with
  (C,D) on site #2.
  @param numberOfAllowedSpecies number of components allowed
**/
void Structure::setNumberOfAllowedSpecies(const int numberOfAllowedSpecies)
{
    std::vector<int> numbersOfAllowedSpecies(_atomicNumbers.size(), numberOfAllowedSpecies);
    _numbersOfAllowedSpecies = numbersOfAllowedSpecies;
}

/**
  @details This function returns the number of components allowed on a
  given site.
  @param index index of the site
  @returns the number of the allowed components
**/
int Structure::getNumberOfAllowedSpeciesBySite(const size_t index) const
{
    if (index >= _numbersOfAllowedSpecies.size())
    {
        std::string errorMessage = "Site index out of bounds";
        errorMessage += " index: " + std::to_string(index);
        errorMessage += " nsites: " + std::to_string(_numbersOfAllowedSpecies.size());
        errorMessage += " (Structure::getNumberOfAllowedSpeciesBySite)";
        throw std::out_of_range(errorMessage);
    }
    return _numbersOfAllowedSpecies[index];
}

/**
  @details This function returns the a vector with number of components allowed on each site index
  @param indices indices of sites
  @returns the list of number of allowed components for each site
**/
std::vector<int> Structure::getNumberOfAllowedSpeciesBySites(const std::vector<LatticeSite> &sites) const
{
    std::vector<int> numberOfAllowedSpecies(sites.size());
    int i = -1;
    for (const auto site : sites)
    {
        i++;
        numberOfAllowedSpecies[i] = getNumberOfAllowedSpeciesBySite(site.index());
    }
    return numberOfAllowedSpecies;
}

/**
  @details This function turns a list of chemical symbols into a list of atomic numbers.
  @param chemicalSymbols vector of chemical symbols (strings) to be converted
**/
std::vector<int> Structure::convertChemicalSymbolsToAtomicNumbers(const std::vector<std::string> &chemicalSymbols) const
{
    std::vector<int> atomicNumbers(chemicalSymbols.size());
    for (size_t i = 0; i < chemicalSymbols.size(); i++)
    {
        atomicNumbers[i] = PeriodicTable::strInt[chemicalSymbols[i]];
    }
    return atomicNumbers;
}

/**
  @details This function turns a list of atomic numbers into a list of chemical symbols.
  @param atomicNumbers vector of atomic numbers (ints) to be converted
**/
std::vector<std::string> Structure::convertAtomicNumbersToChemicalSymbols(const std::vector<int> &atomicNumbers) const
{
    std::vector<std::string> chemicalSymbols(atomicNumbers.size());
    for (size_t i = 0; i < atomicNumbers.size(); i++)
    {
        chemicalSymbols[i] = PeriodicTable::intStr[atomicNumbers[i]];
    }
    return chemicalSymbols;
}
