#include "Symmetry.hpp"

namespace icet
{

/**
@details Returns the next Cartesian product of currentProduct using the vector
    of vectors items.
@param items items[n] gives the possible combinations for element n.
@param currentProduct current value
**/
bool nextCartesianProduct(const std::vector<std::vector<int>> &items,
                          std::vector<int> &currentProduct)
{
    auto n = items.size();
    if (n != currentProduct.size())
    {
        throw std::runtime_error("The sizes of items and currentProduct do not match (Symmetry/nextCartesianProduct)");
    }

    for (size_t i = 0; i < n; ++i)
    {
        if (++currentProduct[i] == (int)items[i].size())
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

} // namespace icet
