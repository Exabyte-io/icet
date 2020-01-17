#include "NeighborList.hpp"

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
        std::ostringstream msg;
        msg << "Site index out of bounds (NeighborList::getNeighbors)" << std::endl;
        msg << " index: " << index;
        msg << " nnbrs: " << _neighbors.size();
        throw std::out_of_range(msg.str());
    }
    return _neighbors[index];
}

/**
@details This function builds a neighbor list for the given structure.

@param structure atomic configuration
@param positionTolerance tolerance applied when evaluating positions in Cartesian coordinates; acts as an effective skin thickness
**/
void NeighborList::build(const Structure &structure, const double positionTolerance)
{
    size_t numberOfSites = structure.size();
    _neighbors.resize(numberOfSites);

    Matrix3d cellInverse = structure.getCell().inverse();
    std::vector<int> unitCellExpanse(3);
    for (size_t i = 0; i < 3; i++)
    {
        if (structure.hasPBC(i))
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
                            double distance_ij = structure.getDistance(i, j, noOffset, extVector);
                            if (distance_ij <= _cutoff + positionTolerance && distance_ij > 2 * positionTolerance)
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
