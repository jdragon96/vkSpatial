#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "Backends/TSDFBackend.h"
#include "Memory/DataSplitter.h"

#include "Engine/Core/Context.h"

// Which window a point belongs to: floor(position / windowWorld), component-wise.
struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey &o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey &k) const {
        std::size_t h = std::hash<int>()(k.x);
        h ^= std::hash<int>()(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct TSDFConfiguration {
    bool useSubmap = true;

    std::string backend = "advanced";
    std::string splitter = "dense";

    TSDFBackendConfig backendConfig{};
    DataSplitterConfig splitterConfig{};

    int windowVoxels = 512;

    int maxResidentWindow = 0;
};

class TSDF {
public:
    void Build(Engine::Core::Context &context, TSDFConfiguration config);

    void Integrate(const std::vector<Eigen::Vector3f> &points,
                   const std::vector<Eigen::Vector3f> &normals,
                   const Eigen::Vector3f &cameraPosition = Eigen::Vector3f::Zero());

    Engine::Core::OrientedPointCloud Extract() const;

    const DataSplitter::DividePointOutput &LastDivision() const { return m_divided; }

    size_t WindowCount() const { return hashTSDF.size() + hashTSDFDetail.size(); }

    size_t BaseWindowCount() const { return hashTSDF.size(); }

    size_t DetailWindowCount() const { return hashTSDFDetail.size(); }

    // World AABB of every live window, for debug overlays. min/max pairs.
    using Box = std::pair<Eigen::Vector3f, Eigen::Vector3f>;

    std::vector<Box> BaseWindowBoxes() const { return BoxesOf(hashTSDF, m_baseWindowWorld); }

    std::vector<Box> DetailWindowBoxes() const {
        return BoxesOf(hashTSDFDetail, m_detailWindowWorld);
    }

    // Points refused because maxResidentWindow was reached. Non-zero means geometry is missing.
    uint32_t WindowLimitRefusalCount() const { return m_windowLimitRefusalCount; }

    // Summed over every window of both levels. tableCount is the window count.
    TSDFBackendStats Stats() const;

    // Every window's filled slots, both levels. Clears `out` first. Not a per-frame path.
    void Download(std::vector<TSDFVoxel> &out) const;

private:
    using WindowMap = std::unordered_map<VoxelKey, std::unique_ptr<TSDFBackend>, VoxelKeyHash>;

    void IntegrateLevel(WindowMap &windows,
                        const TSDFBackendConfig &config,
                        float windowWorld,
                        const std::vector<Eigen::Vector3f> &points,
                        const std::vector<Eigen::Vector3f> &normals,
                        const Eigen::Vector3f &cameraPosition,
                        const std::vector<uint32_t> &index);

    TSDFBackend *FindOrCreateWindow(WindowMap &windows,
                                    const VoxelKey &key,
                                    const TSDFBackendConfig &config,
                                    float windowWorld);

    static std::vector<Box> BoxesOf(const WindowMap &windows, float windowWorld);

    template<typename T>
    static void Gather(const std::vector<T> &source, const std::vector<uint32_t> &index,
                       std::vector<T> &out) {
        out.clear();
        out.reserve(index.size());
        for (uint32_t i: index) out.push_back(source[i]);
    }

private:
    Engine::Core::Context *m_context = nullptr;
    TSDFConfiguration m_config;

    float m_baseWindowWorld = 0.0f;
    float m_detailWindowWorld = 0.0f;
    TSDFBackendConfig m_detailConfig;

    std::unique_ptr<DataSplitter> dataSplitter;

    DataSplitter::DividePointOutput m_divided;
    std::vector<Eigen::Vector3f> m_gatheredPoints;
    std::vector<Eigen::Vector3f> m_gatheredNormals;
    std::unordered_map<VoxelKey, std::vector<uint32_t>, VoxelKeyHash> m_perWindowIndex;
    uint32_t m_windowLimitRefusalCount = 0;
    int m_frameIndex = -1;

    WindowMap hashTSDF;
    WindowMap hashTSDFDetail;
};
