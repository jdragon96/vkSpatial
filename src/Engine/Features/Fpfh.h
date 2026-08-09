#pragma once

#include "Engine/Pipeline/Registration/RegistrationTypes.h"

namespace Engine::Features {

    using Engine::Registration::Fpfh33;
    using Engine::Registration::PointCloud;

    // Fast Point Feature Histogram (FPFH) descriptor, Rusu et al. 2009.
    //
    // `cloud` MUST have per-point normals (cloud.normals.size() == cloud.points.size()).
    // `normalRadius` is the radius used to gather neighbours for each point's own SPFH
    // (Simplified Point Feature Histogram, the Darboux-frame pair features). `fpfhRadius`
    // is the (typically larger) radius used to gather neighbours whose SPFHs are
    // distance-weighted into the final FPFH.
    //
    // Returns one 33-D descriptor per input point, in the same order as cloud.points.
    // Neighbour search uses a plain CPU hash grid (cell size = fpfhRadius) — this module
    // is CPU/Eigen-only and deliberately does not use Engine::Spatial (GPU BVH, has a
    // documented large-N correctness bug, see docs/KNOWN_ISSUES).
    std::vector<Fpfh33> ComputeFpfh(const PointCloud &cloud, float normalRadius, float fpfhRadius);

} // namespace Engine::Features
