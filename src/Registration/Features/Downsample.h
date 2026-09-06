#pragma once

#include "Registration/RegistrationTypes.h"
#include "Common/PointCloud.h"

namespace Features {

    using Common::PointCloud;

    Common::PointCloud DownsampleVoxel(const Common::PointCloud &in, float voxelSize);

} // namespace Features
