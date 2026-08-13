#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/AdvancedTSDF.h"
#include "Engine/Spatial/DirectionalIntegrationQuality.h"

#include <Eigen/Core>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace TSDF {

    /// *********************************************
    /// Build / integration configuration
    /// *********************************************

    // Build-time configuration every volume accepts. A volume that cannot honour a field ignores
    // it rather than failing, so one parameter set drives an A/B run across all of them.
    struct VolumeParams {
        float voxelSize = 0.01f;
        float truncation = 0.03f;
        uint32_t hashCapacity = 1u << 20;      // slots, not bytes; per tile for tiled strategies
        uint32_t maxPointsPerFrame = 1u << 15; // upload buffer sizing hint; grown on demand
        // NaN => the memory strategy picks its own default window (a 512^3 window centred on the origin).
        Eigen::Vector3f windowMinCorner =
                Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
    };

    // Per-run integration knobs. Separated from VolumeParams because these are the quality axes an
    // experiment sweeps between runs, while VolumeParams fixes the grid the runs share.
    struct IntegrationOptions {
        bool pointToPlane = true;      // false = projective ray distance
        float confidenceWeight = 0.5f; // surface-proximity weight lambda in [0,1]; 0 = uniform
        bool hermitePosition = false;  // cubic-Hermite zero-crossing instead of linear
        int currentFrame = 0;          // stamped into a slot on its first fill
        Engine::Spatial::IntegrationQuality quality{};
    };

    /// *********************************************
    /// Measurement surface
    /// *********************************************

    // The numbers a strategy comparison is decided on. occupiedEntryCount / slotCapacity are the
    // n and m of the load factor; insertFailureCount is the silent-drop counter that tells you
    // whether a chosen load-factor ceiling is actually safe (it must stay 0).
    struct VolumeStats {
        uint64_t occupiedEntryCount = 0; // n -- distinct keys currently stored
        uint64_t slotCapacity = 0;       // m -- total slots across every table the volume owns
        uint64_t insertFailureCount = 0; // observations dropped because probing gave up
        uint64_t deviceMemoryBytes = 0;  // GPU bytes held (tables + payload + scratch)
        uint32_t tableCount = 1;         // 1 for a single window; tile count for tiled strategies
        uint32_t growCount = 0;          // rehashes performed since Build

        // n/m. Undefined-free: returns 0 for an unbuilt volume.
        double LoadFactor() const {
            return slotCapacity == 0 ? 0.0
                                     : double(occupiedEntryCount) / double(slotCapacity);
        }
    };

    /// *********************************************
    /// Swappable volume
    /// *********************************************

    // Common swappable surface every TSDF volume implements. Lets an experiment hold the scene,
    // the frames, and the metrics fixed while varying only the strategies underneath -- memory
    // organisation (flat / tiled / submap), slot addressing (linear probe / bucketed), the
    // integration kernel, and the extraction kernel.
    //
    // The batched RecordIntegrate is the primitive: a tiled strategy fuses many tiles into one
    // submit, so self-submitting per call would make tiling impossible to express. Integrate is a
    // non-virtual convenience that wraps it in a CommandBatch, mirroring how SpatialIndex::Build
    // funnels every caller into the single virtual entry point.
    class Volume {
    public:
        virtual ~Volume() = default;

        Volume(const Volume &) = delete;
        Volume &operator=(const Volume &) = delete;

        // Allocate GPU state and compile kernels. Must be called before anything else.
        virtual void Build(Engine::Core::Context &context, const VolumeParams &params) = 0;

        // Empty the volume without reallocating. Stats reset to zero except deviceMemoryBytes.
        virtual void Reset() = 0;

        // Applies from the next integration onward; strategies ignore options they do not support.
        virtual void Configure(const IntegrationOptions &options) = 0;

        // Record upload + integrate dispatches into `batch` WITHOUT submitting. `normals` drives
        // direction selection and the stored gradient, so it must be the same length as `points`.
        // Named apart from Integrate rather than overloading it: an overload pair would be hidden
        // by any derived class that overrides only one of them.
        virtual void Record(const std::vector<Eigen::Vector3f> &points,
                            const std::vector<Eigen::Vector3f> &normals,
                            const Eigen::Vector3f &cameraPosition,
                            Engine::Compute::CommandBatch &batch) = 0;

        // Compact every occupied slot into `out`. Reusing form: `out` is resized, not reallocated,
        // so a per-frame caller keeps one buffer alive. Non-directional volumes report direction 0.
        virtual void Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const = 0;

        // Occupancy and health counters for the state as of the last completed integration.
        virtual VolumeStats Stats() const = 0;

        // Benchmark label, matching the name this volume is registered under.
        virtual const char *Name() const = 0;

        // Convenience: integrate one frame in its own batch and submit it. Equivalent to
        // Record into a fresh CommandBatch followed by Submit.
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPosition = Eigen::Vector3f::Zero());

        // Convenience: allocate a vector per call. Tests and one-off callers only -- the per-frame
        // path uses the reusing form above.
        std::vector<Engine::Spatial::AdvancedEntry> Download() const {
            std::vector<Engine::Spatial::AdvancedEntry> out;
            Download(out);
            return out;
        }

    protected:
        Volume() = default;

        // The non-virtual Integrate needs a device to open a CommandBatch on, but the base class
        // owns no state -- the concrete strategy holds the context and hands it back here.
        // Null before Build.
        virtual Engine::Core::Context *Device() const = 0;
    };

    /// *********************************************
    /// Registry
    /// *********************************************

    // Name -> volume factory, so a tool can switch strategies from a command-line flag
    // (--tsdf flat, --tsdf tile-bucketed, ...) without recompiling. Mirrors
    // Engine::Spatial::Extraction::ExtractorRegistry.
    class VolumeRegistry {
    public:
        using Factory = std::function<std::unique_ptr<Volume>()>;

        void Register(const std::string &name, Factory factory) {
            m_factories[name] = std::move(factory);
        }

        // Null when `name` is unknown -- callers report the typo rather than falling back silently.
        std::unique_ptr<Volume> Create(const std::string &name) const {
            const auto it = m_factories.find(name);
            return it == m_factories.end() ? nullptr : it->second();
        }

        bool Has(const std::string &name) const { return m_factories.count(name) != 0; }

        // Registered names, sorted, for usage messages and sweep-all runs.
        std::vector<std::string> Names() const;

        // Every volume built into the repository. Defined in VolumeRegistry.cpp.
        static VolumeRegistry Default();

    private:
        std::unordered_map<std::string, Factory> m_factories;
    };

} // namespace TSDF
