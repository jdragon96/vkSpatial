#include "TSDF/Volume.h"

namespace TSDF {

    void Volume::Integrate(const std::vector<Eigen::Vector3f> &points,
                           const std::vector<Eigen::Vector3f> &normals,
                           const Eigen::Vector3f &cameraPosition) {
        if (points.empty()) return;
        Engine::Core::Context *device = Device();
        if (device == nullptr) return; // Build has not run -- nothing to record onto
        Engine::Compute::CommandBatch batch(*device);
        Record(points, normals, cameraPosition, batch);
        batch.Submit();
    }

} // namespace TSDF
