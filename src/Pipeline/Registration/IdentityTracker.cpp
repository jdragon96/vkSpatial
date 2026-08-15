#include "Pipeline/Registration/IdentityTracker.h"

namespace Pipeline {

    TrackingResult IdentityTracker::Track(const Frame &, const ModelSnapshot *,
                                          const Eigen::Isometry3f &) {
        TrackingResult r;
        r.pose = Eigen::Isometry3f::Identity();
        r.fitness = 1.0f;
        r.valid = true;
        return r;
    }

} // namespace Pipeline
