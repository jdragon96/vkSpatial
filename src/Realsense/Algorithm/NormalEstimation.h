#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include "Realsense/RealSenseTypes.h"

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Realsense {
    struct NormalEstimatorStrategy {
        const char *name;
        const char *macroName; // nullptr = compile the dispatcher's default fragment
    };

    inline const std::vector<NormalEstimatorStrategy> &NormalEstimatorStrategies() {
        // One forward-difference triangle. Two samples per tangent, so the gradient carries
        // sqrt(2)*sigma of the per-pixel depth noise, and the normal it produces actually belongs
        // to (u+0.5, v+0.5) while it is stored at (u,v).
        static const std::vector<NormalEstimatorStrategy> strategies = {
                {"forward", nullptr},
                // Symmetric differences. Same cost and same 3x3 footprint, but the half-pixel
                // offset cancels and the gradient noise drops to sigma/sqrt(2) -- half the angular
                // noise of "forward" for free.
                {"central", "NORMAL_CENTRAL_DIFFERENCE"},
                // Least-squares plane through every same-surface sample in a window. The slope of
                // a least-squares fit over k evenly spaced samples carries sigma*sqrt(12/(k(k^2-1)))
                // -- 0.32*sigma at k=5 against 1.41*sigma for "forward", so 4.5x less gradient
                // noise, at the price of a k*k gather and a 3x3 eigen solve per pixel.
                {"planefit", "NORMAL_PLANE_FIT"},
        };
        return strategies;
    }

    inline std::vector<std::string> NormalEstimatorNames() {
        std::vector<std::string> names;
        names.reserve(NormalEstimatorStrategies().size());
        for (const NormalEstimatorStrategy &strategy: NormalEstimatorStrategies())
            names.emplace_back(strategy.name);
        return names;
    }

    // Throws rather than falling back to the default: a misspelled name that silently ran
    // "forward" would report the accuracy of an estimator nobody selected.
    inline const NormalEstimatorStrategy &FindNormalEstimator(const std::string &name) {
        for (const NormalEstimatorStrategy &strategy: NormalEstimatorStrategies())
            if (name == strategy.name) return strategy;

        std::string known;
        for (const NormalEstimatorStrategy &strategy: NormalEstimatorStrategies()) {
            if (!known.empty()) known += ", ";
            known += strategy.name;
        }
        throw std::runtime_error("Realsense::FindNormalEstimator: unknown normal estimator '" +
                                 name + "'; registered names are " + known);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Surface normals over an organised point cloud.
    //
    // Owns the strategy axis and the kernel variants and nothing else -- the buffers and the order
    // of the passes stay with ValidationMask, which is what sequences them.
    //
    // Neighbour search is free here: (u+-1, v+-1) IS the neighbourhood, so there is no KD-tree and
    // no radius query, which is the whole reason a depth frame is worth estimating normals on
    // directly rather than after it becomes an unordered cloud.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class NormalEstimation {
    public:
        explicit NormalEstimation(Engine::Core::Context &context) : m_context(context) {
            // Build the fall-through variant now so a broken kernel, dispatcher or shared include
            // surfaces here rather than at the first frame. Deliberately NOT the option struct's
            // default estimator: this constructor is handed no options, so compiling the one the
            // defaults happen to name would tie construction to a value nobody passed it. The
            // strategy fragments are compiled the first time each is asked for.
            NormalEstimationOptions fallThrough;
            fallThrough.estimator = "forward";
            estimatorKernel(fallThrough);
        }

        void RecordEstimate(Engine::Compute::CommandBatch &batch,
                            Engine::Core::Buffer &vertices,
                            Engine::Core::Buffer &properties,
                            Engine::Core::Buffer &normals,
                            Engine::Core::Buffer &counters,
                            int width,
                            int height,
                            const ValidationScoreOptions &scoreOptions,
                            const NormalEstimationOptions &normalOptions) {
            ValidateOptions(normalOptions);

            const NormalPushConstants pushConstants{width,
                                                    height,
                                                    scoreOptions.subpixelRms,
                                                    scoreOptions.focalLengthPixels,
                                                    scoreOptions.baselineMeters,
                                                    scoreOptions.sameSurfaceSigmaMultiplier,
                                                    normalOptions.planeFitRadius,
                                                    normalOptions.minimumPlaneFitSamples};

            Engine::Core::ComputePipeline &kernel = estimatorKernel(normalOptions);
            kernel.Bind(0, vertices).Bind(1, properties).Bind(2, normals).Bind(3, counters);
            kernel.Args(pushConstants);

            const VkExtent3D localSize = kernel.GetLocalSize();
            if (localSize.width == 0 || localSize.height == 0)
                throw std::runtime_error("Realsense::NormalEstimation::RecordEstimate: the kernel "
                                         "reported a zero local size");
            batch.Dispatch(kernel,
                           (std::uint32_t(width) + localSize.width - 1) / localSize.width,
                           (std::uint32_t(height) + localSize.height - 1) / localSize.height,
                           1);
        }

        static void ValidateOptions(const NormalEstimationOptions &options) {
            FindNormalEstimator(options.estimator); // throws on an unknown name
            if (options.planeFitRadius < 1)
                throw std::runtime_error("Realsense::NormalEstimation: planeFitRadius is " +
                                         std::to_string(options.planeFitRadius) +
                                         "; the fit needs at least one ring around the centre");
            // Three non-collinear samples are the algebraic minimum for a plane; anything less
            // cannot be a fit, and the caller's threshold decides how over-determined it must be.
            if (options.minimumPlaneFitSamples < 3)
                throw std::runtime_error("Realsense::NormalEstimation: minimumPlaneFitSamples is " +
                                         std::to_string(options.minimumPlaneFitSamples) +
                                         "; a plane needs at least 3");
        }

    private:
        // One pipeline per strategy, compiled on first use and cached. Rebuilding rather than
        // caching would re-allocate descriptors on every frame of an A/B run that alternates
        // estimators.
        Engine::Core::ComputePipeline &estimatorKernel(const NormalEstimationOptions &options) {
            auto found = m_kernels.find(options.estimator);
            if (found != m_kernels.end()) return *found->second;

            const NormalEstimatorStrategy &strategy = FindNormalEstimator(options.estimator);
            auto pipeline = std::make_unique<Engine::Core::ComputePipeline>(m_context);
            if (strategy.macroName) pipeline->Define(strategy.macroName);
            pipeline->Build("Realsense/Algorithm/NormalEstimation.EstimateNormal.glsl");
            return *m_kernels.emplace(options.estimator, std::move(pipeline)).first->second;
        }

        Engine::Core::Context &m_context;
        std::map<std::string, std::unique_ptr<Engine::Core::ComputePipeline>> m_kernels;
    };

} // namespace Realsense
