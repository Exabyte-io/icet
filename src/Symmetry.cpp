#include "Symmetry.hpp"
namespace icet
{

/// Return the transformed position `position` using the input translation and rotation
Eigen::Vector3d transformPosition(const Eigen::Vector3d &position, const Eigen::Vector3d &translation, const Eigen::Matrix3d &rotation)
{
    Eigen::Vector3d transformedPosition = position; //translation.transpose() + position.transpose() * rotation.transpose();
    return transformedPosition;
}

bool nextCartesianProduct(const std::vector<std::vector<int>> &items, std::vector<int> &currentProduct)
{
    auto n = items.size();
    if (n != currentProduct.size())
    {
        throw std::runtime_error("ERROR: items and currentProduct are different sizes in nextCartesianProduct");
    }

    for (size_t i = 0; i < n; ++i)
    {
        if (++currentProduct[i] == (int) items[i].size())
        {
            currentProduct[i] = 0;
        }
        else
        {
            return true;
        }
    }
    return false;
}

/// Creates a vector of positions by offsetting the structure's positions by an offset
std::vector<Eigen::Vector3d> getOffsetPositions(const Structure &structure, const Eigen::Vector3d &offset)
{
    std::vector<Eigen::Vector3d> positions;
    positions.reserve(structure.size());
    for (size_t i=0; i<structure.size(); i++)
    {
        LatticeSite latticeSite = LatticeSite(i, offset);
        Eigen::Vector3d position = structure.getPosition(latticeSite);
        positions.push_back(position);
    }
    return positions;

}


}
