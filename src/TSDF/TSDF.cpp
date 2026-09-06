#include "TSDF.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

    int WindowAxis(float coordinate, float windowWorld) {
        return static_cast<int>(std::floor(coordinate / windowWorld));
    }

    template<typename Fn>
    void ForEachTouchedWindow(const Eigen::Vector3f &p, float windowWorld, float truncation, Fn &&fn) {
        const Eigen::Vector3f low = p - Eigen::Vector3f::Constant(truncation);
        const Eigen::Vector3f high = p + Eigen::Vector3f::Constant(truncation);
        const int x0 = WindowAxis(low.x(), windowWorld), x1 = WindowAxis(high.x(), windowWorld);
        const int y0 = WindowAxis(low.y(), windowWorld), y1 = WindowAxis(high.y(), windowWorld);
        const int z0 = WindowAxis(low.z(), windowWorld), z1 = WindowAxis(high.z(), windowWorld);
        for (int x = x0; x <= x1; ++x) {
            for (int y = y0; y <= y1; ++y) {
                for (int z = z0; z <= z1; ++z) {
                    fn(VoxelKey{x, y, z});
                }
            }
        }
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

    // Rebuilt rather than cleared in place: keeping the keys made every frame walk every window
    // the scan had ever touched, for both levels.
    m_perWindowIndex.clear();
    for (uint32_t i: index)
        ForEachTouchedWindow(points[i], windowWorld, config.truncation,
                             [&](const VoxelKey &key) { m_perWindowIndex[key].push_back(i); });

    // A point can now be routed to several windows (its band straddles a boundary), so a refused
    // window is NOT a lost point -- only a point that reached NO window is lost. Counting per
    // refused bucket would report one point many times.
    m_pointAccepted.assign(points.size(), 0);

    for (const auto &[key, bucket]: m_perWindowIndex) {
        if (bucket.empty()) continue;

        TSDFBackend *backend = FindOrCreateWindow(windows, key, config, windowWorld);
        if (!backend) continue;
        for (uint32_t i: bucket) m_pointAccepted[i] = 1;

        backend->SetFrameIndex(m_frameIndex);
        Gather(points, bucket, m_gatheredPoints);
        Gather(normals, bucket, m_gatheredNormals);
        backend->Integrate(m_gatheredPoints, m_gatheredNormals, cameraPosition);
    }

    for (uint32_t i: index)
        if (!m_pointAccepted[i]) ++m_windowLimitRefusalCount;
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
