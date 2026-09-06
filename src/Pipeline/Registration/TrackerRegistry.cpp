#include "Pipeline/Registration/Tracker.h"
#include "Pipeline/Registration/IdentityTracker.h"
#include "Pipeline/Registration/PointToPlaneIcpTracker.h"
#include "Pipeline/Registration/GpuIcpTracker.h"
#include "Pipeline/Registration/GlobalRegistrationTracker.h"
#include "Pipeline/Registration/RelocalizingIcpTracker.h"

#include <memory>

namespace Pipeline {

    TrackerRegistry TrackerRegistry::Default() {
        TrackerRegistry reg;
        reg.Register("identity", [] { return std::make_unique<IdentityTracker>(); });
        reg.Register("icp", [] { return std::make_unique<GpuIcpTracker>(); });
        reg.Register("icp-cpu", [] { return std::make_unique<PointToPlaneIcpTracker>(); });
        reg.Register("global", [] { return std::make_unique<GlobalRegistrationTracker>(); });
        reg.Register("icp+global", [] { return std::make_unique<RelocalizingIcpTracker>(); });
        return reg;
    }

} // namespace Pipeline
