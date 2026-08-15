#include "Mesh/QuadraticErrorFunction.h"

#include <Eigen/Cholesky> // LDLT (Eigen/Core only forward-declares it)

namespace Mesh {

    void QuadraticErrorFunction::Add(const Eigen::Vector3f &point, const Eigen::Vector3f &normal) {
        m_normalMatrix += normal * normal.transpose();
        m_normalTimesDistance += normal * normal.dot(point);
    }

    Eigen::Vector3f QuadraticErrorFunction::Solve(const Eigen::Vector3f &centroidBias, float regularization) const {
        // m_normalMatrix is a sum of normal*normal^T outer products, so it is always symmetric
        // positive SEMI-definite; adding regularization*I (regularization > 0) makes the system
        // strictly positive definite, so ldlt() is both applicable and numerically stable
        // regardless of how rank-deficient the accumulated normals are.
        const Eigen::Matrix3f regularizedMatrix = m_normalMatrix + regularization * Eigen::Matrix3f::Identity();
        const Eigen::Vector3f regularizedRightHandSide = m_normalTimesDistance + regularization * centroidBias;
        return regularizedMatrix.ldlt().solve(regularizedRightHandSide);
    }

} // namespace Mesh
