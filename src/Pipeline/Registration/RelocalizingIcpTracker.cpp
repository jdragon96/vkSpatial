#include "Pipeline/Registration/RelocalizingIcpTracker.h"

namespace Pipeline {

    TrackingResult RelocalizingIcpTracker::Track(const Frame &frame,
                                                 const ModelSnapshot *model,
                                                 const Eigen::Isometry3f &priorPose) {
        const TrackingResult local = m_local.Track(frame, model, priorPose);
        if (local.valid) {
            m_consecutiveHealthyMapFailures = 0;
            return local;
        }

        // Only failures against a HEALTHY map arm the fallback. NoModel/NoLocalTarget mean there
        // is no map (yet) where the frame is — global registration has nothing to relocalize
        // against, and those frames are the ones that bootstrap/grow the map when fused.
        const bool healthyMapFailure = local.failure == ETrackFailure::TooFewInliers ||
                                       local.failure == ETrackFailure::LowOverlap ||
                                       local.failure == ETrackFailure::ImplausibleMotion;
        if (!healthyMapFailure) return local;
        if (++m_consecutiveHealthyMapFailures < m_failuresBeforeGlobal) return local;

        // Reset BEFORE the attempt, whatever its outcome: the global pipeline is CPU-expensive
        // (full-model FPFH), so a failed attempt must wait out N fresh failures before retrying
        // instead of re-running every frame while lost.
        m_consecutiveHealthyMapFailures = 0;
        ++m_relocalizationAttempts;

        const TrackingResult global = m_global.Track(frame, model, priorPose);
        if (!global.valid) return local;

        // Refine the global pose with local ICP, and adopt ONLY if that refine passes the normal
        // local gates. The maxStepMeters gate needs no special handling: it measures the step from
        // the prior of THIS solve, which is the global pose itself — the large relocalization jump
        // from the lost prior never enters the refine.
        const TrackingResult refined = m_local.Track(frame, model, global.pose);
        if (!refined.valid) {
            // Return the LOCAL failure, not the refine's: a refine that failed NoLocalTarget-style
            // (the global pose landed off the map) would be fused by ShouldFuse, and a bad
            // relocalized pose is precisely the garbage the per-cause fusion policy exists to keep
            // out of the map.
            return local;
        }

        ++m_relocalizationSuccesses;
        return refined;
    }

} // namespace Pipeline
