#pragma once

#include "Registration/RegistrationTypes.h"
#include "Common/PointCloud.h"

namespace Features {

    using Registration::Correspondence;
    using Registration::Fpfh33;
    using Common::PointCloud;

    std::vector<Correspondence> MatchFeatures(const std::vector<Fpfh33> &srcF, const std::vector<Fpfh33> &tgtF,
                                              float ratioThr = 0.95f, int numMaxCorr = 5000);

} // namespace Features
