#pragma once

#include "Pipeline/Registration/GlobalRegistrationTracker.h"
#include "Pipeline/Registration/GpuIcpTracker.h"
#include "Pipeline/Registration/Tracker.h"

#include <atomic>
#include <cstdint>

namespace Pipeline {

    // Local point-to-plane ICP with a feature-based global relocalization fallback.
    //
    // Local ICP only converges near its prior; once the pose is lost, every subsequent frame fails
    // against a map that is actually fine, and nothing recovers. This composite delegates to the
    // GPU ICP tracker and, after N CONSECUTIVE failures against a healthy map
    // (TooFewInliers/LowOverlap/ImplausibleMotion — never NoModel/NoLocalTarget, where there is no
    // map to relocalize against), runs prior-free global registration (FPFH + RANSAC + Ceres) and
    // re-runs local ICP seeded from its pose. The relocalized pose is adopted only if that refine
    // passes the normal local gates. See docs/ICP_REGISTRATION_QUALITY.md §"Local→Global fallback".
    class RelocalizingIcpTracker : public Tracker {
    public:
        static constexpr int kDefaultFailuresBeforeGlobal = 3;

        explicit RelocalizingIcpTracker(int failuresBeforeGlobal = kDefaultFailuresBeforeGlobal)
            : m_failuresBeforeGlobal(failuresBeforeGlobal) {}

        const char *Name() const override { return "icp+global"; }

        TrackingResult Track(const Frame &frame,
                             const ModelSnapshot *model,
                             const Eigen::Isometry3f &priorPose) override;

        // Observability: the fallback is expensive and silent success/failure would be
        // undiagnosable — expose how often it fired and how often it was adopted. Atomic because
        // Stats() is read from the caller's thread while Track runs on the registration thread.
        TrackerStats Stats() const override {
            TrackerStats stats;
            stats.relocalizationAttempts = m_relocalizationAttempts.load();
            stats.relocalizationSuccesses = m_relocalizationSuccesses.load();
            return stats;
        }

    private:
        GpuIcpTracker m_local;
        GlobalRegistrationTracker m_global;
        int m_failuresBeforeGlobal;
        int m_consecutiveHealthyMapFailures = 0;
        std::atomic<std::uint64_t> m_relocalizationAttempts{0};
        std::atomic<std::uint64_t> m_relocalizationSuccesses{0};
    };

} // namespace Pipeline
