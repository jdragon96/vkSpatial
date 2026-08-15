#pragma once

#include "Pipeline/Registration/Tracker.h"

namespace Pipeline {

    // Frames are already world-registered (e.g. object_scan_viewer output): pose = identity. Also
    // the bootstrap command when no model exists yet.
    class IdentityTracker : public Tracker {
    public:
        const char *Name() const override { return "identity"; }
        TrackingResult Track(const Frame &, const ModelSnapshot *,
                             const Eigen::Isometry3f &) override;
    };

} // namespace Pipeline
