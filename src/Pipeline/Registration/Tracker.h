#pragma once

#include "Pipeline/Types.h"
#include "Registration/RegistrationParam.h"

#include <Eigen/Geometry>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <utility>

namespace Pipeline {

    // ETrackFailure now lives in Types.h: TrackedFrame carries it across the registration ->
    // integration boundary so the fusion policy can be per-cause.

    struct TrackingResult {
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity(); // sensor -> world
        float fitness = 0.0f;
        std::size_t inliers = 0;
        float rmse = 0.0f; // residual rmse (0 if the tracker doesn't compute one, e.g. identity)
        bool valid = false;
        ETrackFailure failure = ETrackFailure::NoModel;
    };

    class Tracker {
    public:
        virtual ~Tracker() = default;
        virtual const char *Name() const = 0;
        virtual TrackingResult Track(const Frame &frame,
                                     const ModelSnapshot *model,
                                     const Eigen::Isometry3f &priorPose) = 0;
        // Optional gate tuning. A tracker with no gates (identity, global) ignores it; the ICP
        // trackers apply it; a composite forwards it to whichever tracker actually solves.
        //
        // This is a virtual rather than a setter on the concrete class because callers reach a
        // tracker through TrackerRegistry, which hands back a Tracker. Configuring one meant a
        // dynamic_cast to GpuIcpTracker, and that cast MISSES on "icp+global" -- which contains a
        // GpuIcpTracker rather than deriving from one -- so every gate the caller set was silently
        // dropped for exactly the tracker whose extra machinery makes tuning matter most.
        //
        // Fields left at 0 mean "keep the tracker's own default"; what that default is belongs to
        // the tracker, not the caller. Call it before Track: there is no locking.
        virtual void Configure(const Registration::RegistrationParam &params) { (void) params; }

        // Internal counters for Pipeline::GetStats(). Called from the caller's thread while Track
        // runs on the registration thread, so an implementation must keep them atomic.
        virtual TrackerStats Stats() const { return {}; }
    };

    class TrackerRegistry {
    public:
        using Factory = std::function<std::unique_ptr<Tracker>()>;

        void Register(const std::string &name, Factory factory) {
            m_factories[name] = std::move(factory);
        }
        std::unique_ptr<Tracker> Create(const std::string &name) const {
            const auto it = m_factories.find(name);
            return it == m_factories.end() ? nullptr : it->second();
        }
        bool Has(const std::string &name) const { return m_factories.count(name) != 0; }

        // Sorted, so a UI listing them does not reorder itself between runs -- the backing map is
        // unordered, and a combo box whose entries move is worse than no combo box.
        std::vector<std::string> Names() const {
            std::vector<std::string> names;
            names.reserve(m_factories.size());
            for (const auto &entry: m_factories) names.push_back(entry.first);
            std::sort(names.begin(), names.end());
            return names;
        }

        static TrackerRegistry Default();

    private:
        std::unordered_map<std::string, Factory> m_factories;
    };

} // namespace Pipeline
