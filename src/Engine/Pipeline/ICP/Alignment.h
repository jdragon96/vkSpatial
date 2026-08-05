#pragma once

#include "Engine/Pipeline/Types.h" // Frame, ModelSnapshot

#include <Eigen/Geometry>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace Engine::Pipeline {

    struct AlignmentResult {
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity(); // sensor -> world
        float fitness = 0.0f;
        std::size_t inliers = 0;
        bool valid = false;
    };

    class AlignmentCommand {
    public:
        virtual ~AlignmentCommand() = default;
        virtual const char *Name() const = 0;
        // Align `frame` to `model` (null before the first map), seeded by `priorPose`. Pure CPU.
        virtual AlignmentResult Execute(const Frame &frame, const ModelSnapshot *model,
                                        const Eigen::Isometry3f &priorPose) = 0;
    };

    class AlignmentRegistry {
    public:
        using Factory = std::function<std::unique_ptr<AlignmentCommand>()>;

        void Register(const std::string &name, Factory factory) {
            m_factories[name] = std::move(factory);
        }
        std::unique_ptr<AlignmentCommand> Create(const std::string &name) const {
            const auto it = m_factories.find(name);
            return it == m_factories.end() ? nullptr : it->second();
        }
        bool Has(const std::string &name) const { return m_factories.count(name) != 0; }

        static AlignmentRegistry Default();

    private:
        std::unordered_map<std::string, Factory> m_factories;
    };

} // namespace Engine::Pipeline
