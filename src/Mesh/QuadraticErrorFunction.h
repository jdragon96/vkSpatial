#pragma once

// Shared per-cell Quadratic Error Function (QEF) solver: minimizes the total squared
// point-to-plane distance to a set of accumulated (point, unit normal) tangent-plane
// constraints, i.e. sum_i (normal_i . (x - point_i))^2 over x. This is the standard QEF from
// dual-contouring-style feature-preserving surface extraction; consumed first by the "emc"
// (Extended Marching Cubes) strategy (ExtendedMarchingCubesExtractor.cpp) to place one feature
// vertex per crease/corner cell, and reused as-is by later feature-preserving strategies added
// in subsequent tasks.
//
// Reference: T. Ju, F. Losasso, S. Schaefer, J. Warren, "Dual Contouring of Hermite Data",
// SIGGRAPH 2002.

#include <Eigen/Core>

namespace Mesh {

    class QuadraticErrorFunction {
    public:
        // Accumulates one tangent-plane constraint (point, unit normal) into the running
        // least-squares normal equations: A^T A += normal * normal^T ; A^T b += normal *
        // (normal . point).
        void Add(const Eigen::Vector3f &point, const Eigen::Vector3f &normal);

        // Solves the accumulated system for the point minimizing total squared point-to-plane
        // distance, Tikhonov-regularized and biased toward `centroidBias` (typically the
        // cell's centroid):
        //     (A^T A + regularization * I) x = A^T b + regularization * centroidBias
        // The regularization term is what makes this numerically stable for rank-deficient
        // accumulations -- a flat cell (every normal ~parallel) constrains only 1 of 3
        // dimensions, an edge/crease cell constrains 2 -- by guaranteeing the system matrix is
        // strictly positive definite (so a Cholesky-family solve always succeeds) and pulling
        // every under-constrained direction toward `centroidBias` instead of leaving it
        // arbitrary/unbounded.
        Eigen::Vector3f Solve(const Eigen::Vector3f &centroidBias, float regularization = 1e-3f) const;

    private:
        Eigen::Matrix3f m_normalMatrix = Eigen::Matrix3f::Zero();
        Eigen::Vector3f m_normalTimesDistance = Eigen::Vector3f::Zero();
    };

} // namespace Mesh
