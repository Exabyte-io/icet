#include "NeighborList.hpp"
#include "Structure.hpp"
#include "Vector3dCompare.hpp"

/**
@details This function returns a vector of lattice sites that identify the
neighbors of site in question.

@param index index of site in structure for which neighbor list was build

@returns vector of LatticeSite objects
*/
std::vector<LatticeSite> NeighborList::getNeighbors(size_t index) const
{
    if (index < 0 || index >= _neighbors.size())
    {
        std::string errorMessage = "Site index out of bounds";
        errorMessage += " index: " + std::to_string(index);
        errorMessage += " nnbrs: " + std::to_string(_neighbors.size());
        errorMessage += " (NeighborList::getNeighbors)";
        throw std::out_of_range(errorMessage);
    }
    return _neighbors[index];
}

/**
@details This function returns True (False) if two sites identified by
their respective indices are (not) neighbors of each other.

@param index1 first site
@param index2 second site
@param offset \
*/
bool NeighborList::isNeighbor(const size_t index1, const size_t index2, const Vector3d offset) const
{
    if (index1 < 0 || index1 >= _neighbors.size() || index2 < 0 || index2 >= _neighbors.size())
    {
        std::string errorMessage = "Site index out of bounds";
        errorMessage += " index1: " + std::to_string(index1);
        errorMessage += " index2: " + std::to_string(index2);
        errorMessage += " nnbrs: " + std::to_string(_neighbors.size());
        errorMessage += " (NeighborList::isNeighbor)";
        throw std::out_of_range(errorMessage);
    }

    for (const auto &nbr : _neighbors[index1])
    {
        if (nbr.index() == index2) // indices are the same
        {
            /// @todo testing equality between floats/doubles is dangerous
            if (nbr.unitcellOffset() == offset) // are the _offsets equal?
            {
                return true;
            }
        }
    }
    return false;
}

/**
@details This function builds a neighbor list for the given structure.

@param structure atomic configuration
**/
void NeighborList::build(const Structure &conf)
{
    size_t numberOfSites = conf.size();
    _neighbors.resize(numberOfSites);

    Matrix3d cellInverse = conf.getCell().inverse();
    std::vector<int> unitCellExpanse(3);
    for (size_t i = 0; i < 3; i++)
    {
        if (conf.hasPBC(i))
        {
            auto v = cellInverse.col(i);
            double dotProduct = v.dot(v);
            double h = 1.0 / sqrt(dotProduct);
            int n = (int)(1.0 * _cutoff / h) + 1;
            unitCellExpanse[i] = n;
        }
        else
        {
            unitCellExpanse[i] = 0;
        }
    }

    for (int n1 = 0; n1 < unitCellExpanse[0] + 1; n1++)
    {
        for (int n2 = -unitCellExpanse[1]; n2 < unitCellExpanse[1] + 1; n2++)
        {
            for (int n3 = -unitCellExpanse[2]; n3 < unitCellExpanse[2] + 1; n3++)
            {
                for (int m = -1; m < 2; m += 2)
                {
                    Vector3d extVector(n1 * m, n2 * m, n3 * m);
                    for (size_t i = 0; i < numberOfSites; i++)
                    {
                        for (size_t j = 0; j < numberOfSites; j++)
                        {
                            Vector3d noOffset(0, 0, 0);
                            double distance_ij = conf.getDistance(i, j, noOffset, extVector);
                            if (distance_ij <= _cutoff + DISTTOL && distance_ij > 2 * DISTTOL)
                            {
                                LatticeSite neighbor = LatticeSite(j, extVector);
                                auto find_neighbor = std::find(_neighbors[i].begin(),_neighbors[i].end(), neighbor);
                                if (find_neighbor == _neighbors[i].end())
                                {
                                    _neighbors[i].push_back(neighbor);
                                }
                            }
                        }
                    }
                } // end m loop
            }
        }
    } // end n loop

    for (auto &nbr : _neighbors)
    {
        std::sort(nbr.begin(), nbr.end());
    }

}
