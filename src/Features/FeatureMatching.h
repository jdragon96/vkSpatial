#pragma once

#include "Features/RegistrationTypes.h"

namespace Features {

    using Registration::Correspondence;
    using Registration::Fpfh33;
    using Registration::PointCloud;

    // Nearest-neighbour feature matching in 33-D FPFH descriptor space with Lowe's ratio
    // test, src -> tgt.
    //
    // For each descriptor in `srcF`, finds its 1st- and 2nd-nearest neighbour in `tgtF` by
    // L2 distance and keeps the correspondence iff (1st-NN dist / 2nd-NN dist) < ratioThr.
    // The ratio test rejects ambiguous matches (where the best and second-best target are
    // nearly equidistant, i.e. the match isn't distinctive). Surviving correspondences are
    // sorted by ratio (best/most distinctive first) and capped at `numMaxCorr`.
    //
    // If `tgtF.size() < 2` there's no second-nearest neighbour to form a ratio from, so the
    // ratio test is skipped entirely and an empty result is returned (nothing distinctive
    // can be asserted about a 0- or 1-element target set).
    //
    // Implementation is brute-force O(|srcF| * |tgtF|) over 33-D vectors, which is fine at
    // M1 scale (srcF/tgtF are downsampled/keypoint-scale point clouds, not raw scan
    // points). An M2 speedup would replace the linear scan over tgtF with a kd-tree (e.g.
    // nanoflann) built once over tgtF and reused across all src queries. Deliberately CPU/
    // Eigen/STL only -- does not use Engine::Spatial (GPU BVH, has a documented large-N
    // correctness bug, see docs/KNOWN_ISSUES).
    std::vector<Correspondence> MatchFeatures(const std::vector<Fpfh33> &srcF, const std::vector<Fpfh33> &tgtF,
                                              float ratioThr = 0.95f, int numMaxCorr = 5000);

} // namespace Features
