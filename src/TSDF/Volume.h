#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "TSDF/Backends/AdvancedTSDF.h"
#include "TSDF/Backends/DirectionalIntegrationQuality.h"

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
        // Hash addressing variant: "linear" (default) or "bucketed". Resolved through
        // HashStrategyByName, which falls back to linear on an unknown name.
        std::string hashStrategy = "linear";
    };

    // Per-run integration knobs. Separated from VolumeParams because these are the quality axes an
    // experiment sweeps between runs, while VolumeParams fixes the grid the runs share.
    struct IntegrationOptions {
        bool pointToPlane = true;      // false = projective ray distance
        float confidenceWeight = 0.5f; // surface-proximity weight lambda in [0,1]; 0 = uniform
        bool hermitePosition = false;  // cubic-Hermite zero-crossing instead of linear
        int currentFrame = 0;          // stamped into a slot on its first fill
        TSDF::IntegrationQuality quality{};
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
        // Hash tables and their parallel first-fill stamps ONLY: slotCapacity * kBytesPerHashSlot.
        // Upload buffers and compaction/readback scratch are EXCLUDED, and that excluded scratch
        // differs wildly by strategy (a tiled level allocates ~8 M readback entries on its first
        // download, and submap owns two such levels, while flat's scratch tracks its hash size).
        // So compare this only as table cost -- it is not the strategy's total GPU footprint.
        uint64_t deviceMemoryBytes = 0;
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
    // The batched Record is the primitive: a tiled strategy fuses many tiles into one submit, so
    // self-submitting per call would make tiling impossible to express. Integrate is a non-virtual
    // convenience that wraps it in a CommandBatch, mirroring how SpatialIndex::Build funnels every
    // caller into the single virtual entry point.
    class Volume {
    public:
        virtual ~Volume() = default;

        Volume(const Volume &) = delete;
        Volume &operator=(const Volume &) = delete;

        // Allocate GPU state and compile kernels. Must be called before anything else.
        virtual void Build(Engine::Core::Context &context, const VolumeParams &params) = 0;

        // Empty the volume. What survives differs by strategy and the difference is visible in
        // Stats: a single-window strategy keeps its (possibly grown) table, so slotCapacity and
        // deviceMemoryBytes are unchanged and only occupiedEntryCount returns to zero. A lazily
        // tiled strategy drops its tiles, so slotCapacity, tableCount AND deviceMemoryBytes all
        // fall to zero and the tables are reallocated on the next integration. Only
        // occupiedEntryCount == 0 afterwards is common to all of them.
        virtual void Reset() = 0;

        // Applies from the next integration onward; strategies ignore options they do not support.
        // Set options BEFORE the first integration. Lazily tiled strategies (tile, submap) copy the
        // current options into a tile when that tile is CREATED and do not retro-apply to tiles
        // that already exist, so a mid-run Configure leaves the volume running two different
        // configurations at once -- old tiles on the old options, new tiles on the new ones.
        // Deliberately unguarded against a missing Build: every implementation only writes plain
        // fields that a later Build preserves, so Configure-then-Build is a supported order.
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
        virtual void Download(std::vector<TSDF::AdvancedEntry> &out) const = 0;

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
        std::vector<TSDF::AdvancedEntry> Download() const {
            std::vector<TSDF::AdvancedEntry> out;
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
