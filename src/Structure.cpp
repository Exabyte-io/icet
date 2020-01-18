#include "Structure.hpp"

#include "FloatType.hpp"
#include "PeriodicTable.hpp"

using namespace Eigen;

/**
  @details Initializes an icet Structure instance.
  @param positions list of positions in Cartesian coordinates
  @param chemicalSymbols list of chemical symbols
  @param cell cell metric
  @param pbc periodic boundary conditions
**/
Structure::Structure(const Matrix<double, Dynamic, 3, RowMajor> &positions,
                     const std::vector<std::string> &chemicalSymbols,
                     const Matrix3d &cell,
                     const std::vector<bool> &pbc)
                     : _cell(cell), _pbc(pbc)
{
    setPositions(positions);
    setChemicalSymbols(chemicalSymbols);
    _uniqueSites.resize(chemicalSymbols.size());
    _numbersOfAllowedSpecies.resize(positions.rows());
}

/**
  @details This function computes the distance between two sites.
  @param index1 index of the first site
  @param index2 index of the second site
  @param offset1 offset of site 1 relative to origin in units of lattice vectors
  @param offset2 offset of site 2 relative to origin in units of lattice vectors
*/
double Structure::getDistance(const size_t index1,
                              const size_t index2,
                              const Vector3d offset1 = {0.0, 0.0, 0.0},
                              const Vector3d offset2 = {0.0, 0.0, 0.0}) const
{
    if (index1 >= (size_t)_positions.rows() ||
        index2 >= (size_t)_positions.rows())
    {
        std::ostringstream msg;
        msg << "At least one site index out of bounds ";
        msg << " index1: " << index1;
        msg << " index2: " << index2;
        msg << " positions: " << _positions.rows();
        msg << " (Structure::getDistance)";
        throw std::out_of_range(msg.str());
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
    if (latticeNeighbor.index() >= (size_t)_positions.rows())
    {
        std::ostringstream msg;
        msg << "Site index out of bounds";
        msg << " index: " << latticeNeighbor.index();
        msg << " number of positions: " << _positions.rows();
        msg << " (Structure::getPosition)";
        throw std::out_of_range(msg.str());
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
  @param index index of site
  @returns atomic number
**/
int Structure::getAtomicNumber(const size_t index) const
{
    if (index >= _atomicNumbers.size())
    {
        std::ostringstream msg;
        msg << "Site index out of bounds";
        msg << " index: " << index;
        msg << " nsites: " << _atomicNumbers.size();
        msg << " (Structure::getAtomicNumber)";
        throw std::out_of_range(msg.str());
    }
    return _atomicNumbers[index];
}

/**
  @details This function sets the symmetrically distinct sites associated with
      the structure. It requires a vector as input the length of which  must
      match the number of positions.
  @param sites list of integers
**/
void Structure::setUniqueSites(const std::vector<size_t> &sites)
{
    if (sites.size() != (size_t)_positions.rows())
    {
        std::ostringstream msg;
        msg << "Length of input vector does not match number of sites";
        msg << " nsites: " << sites.size();
        msg << " positions: " << _positions.rows();
        msg << " (Structure::setUniqueSites)";
        throw std::out_of_range(msg.str());
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
        std::ostringstream msg;
        msg << "Site index out of bounds";
        msg << " i: " << i ;
        msg << " nsites: " << _uniqueSites.size();
        msg << " (Structure::getUniqueSite)";
        throw std::out_of_range(msg.str());
    }
    return _uniqueSites[i];
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
  @param fractionalPositionTolerance tolerance applied when comparing positions in fractional coordinates

  @returns LatticeSite object
*/
LatticeSite Structure::findLatticeSiteByPosition(const Vector3d &position, const double fractionalPositionTolerance) const
{
    /// Loop over all positions
    for (size_t i = 0; i < (size_t)_positions.rows(); i++)
    {
        Vector3d distanceVector = position - _positions.row(i).transpose();
        Vector3d fractionalDistanceVector = _cell.transpose().partialPivLu().solve(distanceVector);
        icet::roundVector3d(fractionalDistanceVector, FLOATTYPE_EPSILON);
        Vector3d latticeVector = {round(fractionalDistanceVector[0]),
                                  round(fractionalDistanceVector[1]),
                                  round(fractionalDistanceVector[2])};
        if ((fractionalDistanceVector - latticeVector).norm() < fractionalPositionTolerance)
        {
            return LatticeSite(i, latticeVector);
        }
    }
    Vector3d fractionalPosition = _cell.transpose().partialPivLu().solve(position);
    std::ostringstream msg;
    msg << "Failed to find site by position (findLatticeSiteByPosition)." << std::endl;
    msg << "Try increasing symprec or position_tolerance." << std::endl;
    msg << "position: " << position[0] << " " << position[1] << " " << position[2] << std::endl;
    msg << "fractional position: " << fractionalPosition[0] << " " << fractionalPosition[1] << " " << fractionalPosition[2] << std::endl;
    msg << "fractional position tolerance: " << fractionalPositionTolerance;
    throw std::runtime_error(msg.str());

}

/**
  @details This function returns a list ofLatticeSite object the position
  of each matches the respective entry in the list of input positions to the
  tolerance specified for this structure. Internally this function uses
  Structure::findLatticeSiteByPosition.

  @param positions list of position to match in Cartesian coordinates
  @param fractionalPositionTolerance tolerance applied when comparing positions in fractional coordinates

  @returns list of LatticeSite objects
*/
std::vector<LatticeSite> Structure::findLatticeSitesByPositions(const std::vector<Vector3d> &positions, const double fractionalPositionTolerance) const
{
    std::vector<LatticeSite> latticeSites;
    latticeSites.reserve(positions.size());

    for (const Vector3d position : positions)
    {
        latticeSites.push_back(findLatticeSiteByPosition(position, fractionalPositionTolerance));
    }

    return latticeSites;
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
        std::ostringstream msg;
        msg << "Size of input list incompatible with structure";
        msg << " length: " << numbersOfAllowedSpecies.size();
        msg << " nsites: " << size();
        msg << " (Structure::setNumberOfAllowedSpecies)";
        throw std::out_of_range(msg.str());
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
        std::ostringstream msg;
        msg << "Site index out of bounds";
        msg << " index: " << index;
        msg << " nsites: " << _numbersOfAllowedSpecies.size();
        msg << " (Structure::getNumberOfAllowedSpeciesBySite)";
        throw std::out_of_range(msg.str());
    }
    return _numbersOfAllowedSpecies[index];
}

/**
  @details This function returns the a vector with number of components allowed on each site index
  @param sites indices of sites
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
