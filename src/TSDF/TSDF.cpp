#include "TSDF.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

    // Which window covers this point: floor(position / windowWorld), component-wise.
    VoxelKey WindowKeyOf(const Eigen::Vector3f &p, float windowWorld) {
        return VoxelKey{static_cast<int>(std::floor(p.x() / windowWorld)),
                        static_cast<int>(std::floor(p.y() / windowWorld)),
                        static_cast<int>(std::floor(p.z() / windowWorld))};
    }

} // namespace

void TSDF::Build(Engine::Core::Context &context, TSDFConfiguration config) {
    m_context = &context;
    m_config = config;

    dataSplitter = MakeDataSplitter(config.splitter);
    if (!dataSplitter)
        throw std::runtime_error("TSDF::Build: unknown submap strategy '" + config.splitter + "'");
    dataSplitter->Build(context, config.splitterConfig);

    if (!MakeTSDFBackend(config.backend))
        throw std::runtime_error("TSDF::Build: unknown backend '" + config.backend + "'");

    m_detailConfig = config.backendConfig;
    m_detailConfig.voxelSize = config.backendConfig.voxelSize * 0.5f;
    m_detailConfig.truncation = config.backendConfig.truncation * 0.5f;

    m_baseWindowWorld = config.backendConfig.voxelSize * float(config.windowVoxels);
    m_detailWindowWorld = m_detailConfig.voxelSize * float(config.windowVoxels);
    hashTSDF.clear();
    hashTSDFDetail.clear();
    m_windowLimitRefusalCount = 0;
    m_frameIndex = -1;
}

void TSDF::Integrate(const std::vector<Eigen::Vector3f> &points,
                     const std::vector<Eigen::Vector3f> &normals,
                     const Eigen::Vector3f &cameraPosition) {
    if (!m_context || points.empty()) return;
    ++m_frameIndex;

    // 1. Which points deserve the detail level.
    dataSplitter->DividePoint(points, normals, m_divided);

    // 2. Update Base Map
    IntegrateLevel(hashTSDF,
                   m_config.backendConfig,
                   m_baseWindowWorld,
                   points,
                   normals,
                   cameraPosition,
                   m_divided.baseIndex);

    // 3. Update Dense Map
    if (m_config.useSubmap) {
        IntegrateLevel(
                hashTSDFDetail,
                m_detailConfig,
                m_detailWindowWorld,
                points,
                normals,
                cameraPosition,
                m_divided.detailIndex);
    }
}

void TSDF::IntegrateLevel(WindowMap &windows,
                          const TSDFBackendConfig &config,
                          float windowWorld,
                          const std::vector<Eigen::Vector3f> &points,
                          const std::vector<Eigen::Vector3f> &normals,
                          const Eigen::Vector3f &cameraPosition,
                          const std::vector<uint32_t> &index) {
    if (index.empty()) return;

    for (auto &bucket: m_perWindowIndex) bucket.second.clear();
    for (uint32_t i: index) m_perWindowIndex[WindowKeyOf(points[i], windowWorld)].push_back(i);

    for (const auto &[key, bucket]: m_perWindowIndex) {
        if (bucket.empty()) continue;

        TSDFBackend *backend = FindOrCreateWindow(windows, key, config, windowWorld);
        if (!backend) {
            m_windowLimitRefusalCount += uint32_t(bucket.size());
            continue;
        }

        backend->SetFrameIndex(m_frameIndex);
        Gather(points, bucket, m_gatheredPoints);
        Gather(normals, bucket, m_gatheredNormals);
        backend->Integrate(m_gatheredPoints, m_gatheredNormals, cameraPosition);
    }
}

TSDFBackend *TSDF::FindOrCreateWindow(WindowMap &windows,
                                      const VoxelKey &key,
                                      const TSDFBackendConfig &config,
                                      float windowWorld) {
    auto it = windows.find(key);
    if (it != windows.end()) return it->second.get();

    if (m_config.maxResidentWindow > 0 && WindowCount() >= size_t(m_config.maxResidentWindow))
        return nullptr;

    TSDFBackendConfig placed = config;
    placed.hasWindowMinCorner = true;
    placed.windowMinCorner = Eigen::Vector3f(float(key.x), float(key.y), float(key.z)) * windowWorld;

    std::unique_ptr<TSDFBackend> backend = MakeTSDFBackend(m_config.backend);
    backend->Build(*m_context, placed);
    return (windows[key] = std::move(backend)).get();
}

Engine::Core::OrientedPointCloud TSDF::Extract() const {
    Engine::Core::OrientedPointCloud cloud;
    for (const WindowMap *level: {&hashTSDF, &hashTSDFDetail})
        for (const auto &[key, backend]: *level) {
            const Engine::Core::OrientedPointCloud part = backend->Extract();
            cloud.points.insert(cloud.points.end(), part.points.begin(), part.points.end());
            cloud.normals.insert(cloud.normals.end(), part.normals.begin(), part.normals.end());
        }
    return cloud;
}

TSDFBackendStats TSDF::Stats() const {
    TSDFBackendStats total;
    for (const WindowMap *level: {&hashTSDF, &hashTSDFDetail})
        for (const auto &[key, backend]: *level) {
            const TSDFBackendStats one = backend->Stats();
            total.filledCount += one.filledCount;
            total.hashCapacity += one.hashCapacity;
            total.insertFailureCount += one.insertFailureCount;
            total.growCount += one.growCount;
            total.deviceMemoryBytes += one.deviceMemoryBytes;
            total.tableCount += one.tableCount;
            total.probeSlotTotal += one.probeSlotTotal;
            total.probeQueryCount += one.probeQueryCount;
            total.probeSlotMax = std::max(total.probeSlotMax, one.probeSlotMax);
        }
    return total;
}

void TSDF::Download(std::vector<TSDFVoxel> &out) const {
    out.clear();
    for (const WindowMap *level: {&hashTSDF, &hashTSDFDetail})
        for (const auto &[key, backend]: *level) backend->Download(out);
}

std::vector<TSDF::Box> TSDF::BoxesOf(const WindowMap &windows, float windowWorld) {
    std::vector<Box> boxes;
    boxes.reserve(windows.size());
    for (const auto &[key, backend]: windows) {
        const Eigen::Vector3f corner =
                Eigen::Vector3f(float(key.x), float(key.y), float(key.z)) * windowWorld;
        boxes.emplace_back(corner, corner + Eigen::Vector3f::Constant(windowWorld));
    }
    return boxes;
}
