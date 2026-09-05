#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <vector>

namespace Engine::Registration {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Umeyama's closed-form rigid fit: the SE(3) transform that best maps one ordered point set
    // onto another, in one step and with no iteration.
    //
    // This is what makes RANSAC affordable here. A hypothesis costs one call, so the loop can
    // afford to draw thousands of them; an iterative solve per hypothesis could not.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    inline Eigen::Matrix4f SolveRigidUmeyama(const std::vector<Eigen::Vector3f> &src,
                                             const std::vector<Eigen::Vector3f> &dst) {
        const std::size_t n = src.size();
        // 3 non-collinear points are the minimum for a well-posed rigid (rotation + translation)
        // fit; below that umeyama() is degenerate.
        if (n == 0 || dst.size() != n || n < 3) return Eigen::Matrix4f::Identity();

        Eigen::Matrix3Xf S(3, Eigen::Index(n)), D(3, Eigen::Index(n));
        for (std::size_t i = 0; i < n; ++i) {
            S.col(Eigen::Index(i)) = src[i];
            D.col(Eigen::Index(i)) = dst[i];
        }

        // with_scaling = false: rigid (SE(3)) fit only, no scale term. Scale would absorb a bad
        // correspondence set into a plausible-looking transform.
        return Eigen::umeyama(S, D, false);
    }

} // namespace Engine::Registration
