#include "MatrixOfEquivalentPositions.hpp"

/**
@details Creates the symmetry equivalent positions given a set of input
    coordinates using the rotational and translational symmetries provided upon
    initialization of the object.
@param fractionalPositions positions of sites in fractional coordinates
*/
void MatrixOfEquivalentPositions::build(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> &fractionalPositions)
{
    _matrixOfEquivalentPositions.clear();
    _matrixOfEquivalentPositions.resize(fractionalPositions.rows());
    for (unsigned j = 0; j < fractionalPositions.rows(); j++) // row
    {
        for (size_t i = 0; i < _translations.size(); i++) // column
        {
            Eigen::Vector3d permutedPos;
            permutedPos = _translations[i].transpose();
            permutedPos += fractionalPositions.row(j) * _rotations[i].transpose();
            _matrixOfEquivalentPositions[j].push_back(permutedPos);
        }
    }
}
