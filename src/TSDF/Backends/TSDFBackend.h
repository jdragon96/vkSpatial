#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "Engine/Core/OrientedPointCloud.h"

namespace Engine::Core {
    class Context;
}

// Mirror of TSDF::AdaptiveBandConfig; see AdvancedTSDF.h for what the coefficients mean and why
// they are exposed (they are fitted to a Kinect v1 and want refitting per sensor).
struct TSDFAdaptiveBand {
    float sigmaMultiplier = 0.0f; // band = multiplier * sigma_z; 0 = off
    float minimumVoxels = 2.0f;   // floor in voxels; below one voxel the band admits nothing
    float sigmaConstant = 0.0012f;
    float sigmaQuadratic = 0.0019f;
    float sigmaOffsetMeters = 0.4f;
    float sigmaAngular = 0.0001f; // 0 = axial only
};

struct TSDFBackendConfig {
    float voxelSize = 0.01f;
    float truncation = 0.03f;
    uint32_t hashCapacity = 1u << 20;
    uint32_t maxPointPerFrame = 1u << 18;
    std::string hash = "linear";

    // Window origin. Leave empty to centre the window on the world origin.
    bool hasWindowMinCorner = false;
    Eigen::Vector3f windowMinCorner = Eigen::Vector3f::Zero();

    // Mirror AdvancedTSDF's own defaults exactly: exposing a knob must not change behaviour.
    bool pointToPlane = true;
    bool hermitePosition = false;
    float confidenceWeight = 0.5f;
    uint32_t maxDirections = 1;
    uint32_t directionExponent = 4;
    bool viewAngleWeight = false;

    // Range-adaptive truncation band. sigmaMultiplier 0 (the default) disables it, leaving the
    // fixed band this backend always had. Every default mirrors TSDF::AdaptiveBandConfig exactly;
    // the struct is repeated rather than included because this header is the namespace-free
    // boundary and naming the implementation type here would drag TSDF:: into every consumer.
    TSDFAdaptiveBand adaptiveBand;

    // Counts hash probes. A development switch -- it recompiles the integrate kernel and adds
    // three atomics per lookup.
    bool probeStats = false;
};

// One filled (voxel, direction) slot, read back for debugging. Namespace-free so a translation
// unit holding the global `class TSDF` can name it.
struct TSDFVoxel {
    Eigen::Vector3f center;
    Eigen::Vector3f normal;
    uint32_t direction = 0; // which of the 6 canonical axes this slot accumulated along
    float tsdf = 0.0f;
    float weight = 0.0f;
    int32_t firstFrame = -1;
};

struct TSDFBackendStats {
    uint64_t filledCount = 0;
    uint64_t hashCapacity = 0;
    uint64_t insertFailureCount = 0;
    uint64_t growCount = 0;
    uint64_t deviceMemoryBytes = 0;
    uint32_t tableCount = 0;

    // Zero unless TSDFBackendConfig::probeStats was set.
    uint64_t probeSlotTotal = 0;
    uint64_t probeQueryCount = 0;
    uint32_t probeSlotMax = 0;

    double LoadFactor() const {
        return hashCapacity ? double(filledCount) / double(hashCapacity) : 0.0;
    }

    // Slots examined per hash lookup. Linear-probe theory says this is about
    // 0.5 * (1 + 1 / (1 - loadFactor)) for a successful search.
    double AverageProbes() const {
        return probeQueryCount ? double(probeSlotTotal) / double(probeQueryCount) : 0.0;
    }
};

// One unit's TSDF core: the storage backend behind Integrate and Extract.
//
// Implementations live entirely in TSDFBackend.cpp. They wrap types in `namespace TSDF`, which
// cannot be named here -- TSDF.h has a global `class TSDF` and includes this header.
class TSDFBackend {
public:
    virtual ~TSDFBackend() = default;

    virtual void Build(Engine::Core::Context &context, const TSDFBackendConfig &config) = 0;

    virtual void Reset() = 0;

    // Stamped onto voxels this backend fills for the first time, so a debugger can tell which
    // frame first saw a voxel. Set before Integrate.
    virtual void SetFrameIndex(int frame) = 0;

    virtual void Integrate(const std::vector<Eigen::Vector3f> &points,
                           const std::vector<Eigen::Vector3f> &normals,
                           const Eigen::Vector3f &cameraPosition) = 0;

    virtual Engine::Core::OrientedPointCloud Extract() const = 0;

    // Appends this backend's filled slots. Not a per-frame path on a large map.
    virtual void Download(std::vector<TSDFVoxel> &out) const = 0;

    virtual TSDFBackendStats Stats() const = 0;

    virtual const char *Name() const = 0;
};

std::unique_ptr<TSDFBackend> MakeTSDFBackend(const std::string &name);

std::vector<std::string> TSDFBackendNames();

// Hash addressing, selected through TSDFBackendConfig::hash. Resolve returns the strategy that
// will actually run -- an unknown name silently falls back to linear, so callers that care about
// labelling a comparison must ask.
std::vector<std::string> TSDFHashNames();

std::string ResolveHashName(const std::string &name);
