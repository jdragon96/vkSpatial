#include "Engine/Pipeline/Registration/Tracker.h"
#include "Engine/Pipeline/Registration/IdentityTracker.h"
#include "Engine/Pipeline/Registration/PointToPlaneIcpTracker.h"
#include "Engine/Pipeline/Registration/GpuIcpTracker.h"
#include "Engine/Pipeline/Registration/GlobalRegistrationTracker.h"

#include <memory>

namespace Engine::Pipeline {

    TrackerRegistry TrackerRegistry::Default() {
        TrackerRegistry reg;
        reg.Register("identity", [] { return std::make_unique<IdentityTracker>(); });
        reg.Register("icp", [] { return std::make_unique<GpuIcpTracker>(); });
        reg.Register("icp-cpu", [] { return std::make_unique<PointToPlaneIcpTracker>(); });
        reg.Register("global", [] { return std::make_unique<GlobalRegistrationTracker>(); });
        return reg;
    }

} // namespace Engine::Pipeline
