#pragma once

#include "Pipeline/Types.h"

#include <Eigen/Geometry>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace Pipeline {

    struct TrackingResult {
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity(); // sensor -> world
        float fitness = 0.0f;
        std::size_t inliers = 0;
        float rmse = 0.0f; // residual rmse (0 if the tracker doesn't compute one, e.g. identity)
        bool valid = false;
    };

    class Tracker {
    public:
        virtual ~Tracker() = default;
        virtual const char *Name() const = 0;
        virtual TrackingResult Track(const Frame &frame,
                                     const ModelSnapshot *model,
                                     const Eigen::Isometry3f &priorPose) = 0;
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

        static TrackerRegistry Default();

    private:
        std::unordered_map<std::string, Factory> m_factories;
    };

} // namespace Pipeline
