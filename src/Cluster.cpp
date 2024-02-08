#include "Cluster.hpp"

/**
@details Creates an instance of a cluster.
@param structure icet structure object
@param latticeSites list of lattice sites that form the cluster
*/
Cluster::Cluster(const std::vector<LatticeSite> &latticeSites,
                 std::shared_ptr<const Structure> structure)
{
    _latticeSites = latticeSites;
    _structure = structure;

    // A sanity check: no site index can be larger than
    // the number of sites in structure
    for (auto &site : _latticeSites)
    {
        if (site.index() > structure->size())
        {
            std::ostringstream msg;
            msg << "Lattice site incompatible with size of structure,";
            msg << " lattice site: " << site;
            msg << ", number of sites in structure: " << _structure->size();
            msg << " (Cluster initialization)";
            throw std::runtime_error(msg.str());
        }
    }
}

/**
@brief Translates this cluster by an offset.
@param offset Coordinates referring to the axes of the structure in this cluster
**/
void Cluster::translate(const Eigen::Vector3i &offset)
{
    for (LatticeSite &site : _latticeSites)
    {
        site.addUnitcellOffset(offset);
    }
}

/**
@details Transforms the sites in this cluster from the current structure to a new, given supercell.
This involves finding a map from the site in the primitive cell to the supercell.
If no map is found, mapping is attempted based on the position of the site in the supercell.
(The map is important for performance.)
@param supercell supercell structure
@param primitiveToSupercellMap map from primitive to supercell
@param fractionalPositionTolerance tolerance applied when comparing positions in fractional coordinates
**/
void Cluster::transformToSupercell(std::shared_ptr<const Structure> supercell,
                                   std::unordered_map<LatticeSite, LatticeSite> &primitiveToSupercellMap,
                                   const double fractionalPositionTolerance)
{
    LatticeSite supercellSite;
    Vector3d sitePosition;
    for (LatticeSite &site : _latticeSites)
    {
        auto find = primitiveToSupercellMap.find(site);

        if (find == primitiveToSupercellMap.end())
        {
            sitePosition = _structure->getPosition(site);
            supercellSite = supercell->findLatticeSiteByPosition(sitePosition, fractionalPositionTolerance);
            primitiveToSupercellMap[site] = supercellSite;
        }
        else
        {
            supercellSite = primitiveToSupercellMap[site];
        }

        // overwrite site to match supercell index offset
        site.setIndex(supercellSite.index());
        site.setUnitcellOffset(supercellSite.unitcellOffset());
    }
    // Now all sites refer to the supercell, so we change the _structure to point to the supercell
    _structure = supercell;
}

/**
@brief Computes the geometrical radius of the cluster.
*/
double Cluster::radius() const
{
    // The radius is always zero for singlets
    if (order() <= 1)
    {
        return 0.0;
    }
    // Compute the center of the cluster.
    Vector3d centerPosition = {0.0, 0.0, 0.0};
    for (const auto &site : _latticeSites)
    {
        centerPosition += _structure->getPosition(site);
    }
    centerPosition /= _latticeSites.size();

    // Compute the average distance of the points in the cluster to its center.
    double avgDistanceToCenter = 0.0;
    for (const auto &site : _latticeSites)
    {
        avgDistanceToCenter += (centerPosition - _structure->getPosition(site)).norm();
    }
    avgDistanceToCenter /= order();
    return avgDistanceToCenter;
}

/**
@brief Returns the positions of the sites in this cluster in Cartesian coordinates.
**/
std::vector<Vector3d> Cluster::getPositions() const
{
    std::vector<Vector3d> currentPositions;
    for (const LatticeSite &site : _latticeSites)
    {
        currentPositions.push_back(_structure->getPosition(site));
    }
    return currentPositions;
}

/**
@brief Returns the distances between the points in this cluster.
**/
std::vector<float> Cluster::distances() const
{
    std::vector<Vector3d> currentPositions = getPositions();
    std::vector<float> distances = {};
    for (size_t i = 1; i < currentPositions.size(); i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            distances.push_back((currentPositions[j] - currentPositions[i]).norm());
        }
    }
    return distances;
}

/**
@brief Returns the number of allowed components on each site in this cluster.
@return std::vector<int>
*/
std::vector<int> Cluster::getNumberOfAllowedSpeciesPerSite() const
{
    const std::vector<std::vector<int>> &allowedNumbers = _structure->allowedAtomicNumbers();
    std::vector<int> numberOfAllowedSpecies;
    for (const auto &site : _latticeSites)
    {
        numberOfAllowedSpecies.push_back(allowedNumbers[site.index()].size());
    }
    return numberOfAllowedSpecies;
}

/**
@brief Checks whether a site index is included with a zero offset.
@param siteIndex Index of site to check whether it is included
*/
bool Cluster::isSiteIndexIncludedWithZeroOffset(size_t siteIndex) const
{
    return std::any_of(_latticeSites.begin(), _latticeSites.end(), [=](const LatticeSite &ls)
                       { return ls.index() == siteIndex && ls.unitcellOffset().norm() < 1e-4; });
}

/**
@brief Counts the number of occurences of a site index among the sites in this cluster
@param siteIndex Index of site to count
*/
unsigned int Cluster::getCountOfOccurencesOfSiteIndex(size_t siteIndex) const
{
    return std::count_if(_latticeSites.begin(), _latticeSites.end(), [=](const LatticeSite &ls)
                         { return ls.index() == siteIndex; });
}

/**
@brief Stream operator for a Cluster.
*/
std::ostream &operator<<(std::ostream &os, const Cluster &cluster)
{
    int width = 77;
    std::string padding((width - 9) / 2, '=');

    // Print general information
    os << padding << " Cluster " << padding << std::endl;
    os << " Order:      " << cluster.order() << std::endl;
    os << " Radius:     " << cluster.radius() << std::endl;
    if (cluster.order() > 1)
    {
        os << " Distances:";
        for (float d : cluster.distances())
        {
            os << "  " << d;
        }
        os << std::endl;
    }

    // Print lattice sites
    os << std::string(width, '-') << std::endl;
    os << " Unitcell index |   Unitcell offset   |    Position" << std::endl;
    os << std::string(width, '-') << std::endl;
    for (unsigned int site_i = 0; site_i < cluster.order(); site_i++)
    {
        LatticeSite site = cluster.latticeSites()[site_i];
        Vector3d position = cluster.getPositions()[site_i];

        // Print index
        std::string s = std::to_string(site.index());
        padding = std::string(14 - s.size(), ' ');
        os << padding << s << "  |";

        // Print offset
        Vector3i offset = site.unitcellOffset();
        s = "";
        std::string sPart;
        for (int i = 0; i < 3; i++)
        {
            sPart = std::to_string(offset[i]);
            padding = std::string(5 - sPart.size(), ' ');
            s += " " + padding + sPart;
        }
        padding = std::string(19 - s.size(), ' ');
        os << padding << s << "  |";

        // Print position
        s = "";
        for (int i = 0; i < 3; i++)
        {
            sPart = std::to_string(position[i]);
            padding = std::string(11 - sPart.size(), ' ');
            s += " " + padding + sPart;
        }
        padding = std::string(36 - s.size(), ' ');
        os << padding << s << std::endl;
    }

    os << std::string(width, '=');
    return os;
}
