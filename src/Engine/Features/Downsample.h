#pragma once

#include "Engine/Registration/RegistrationTypes.h"

namespace Engine::Features {

    using Engine::Registration::PointCloud;

    // Voxel-grid downsample: one output point per occupied cell (centroid of the points
    // that fall in that cell). If input normals are present, the output normal per cell
    // is the (renormalized) average of the input normals in that cell.
    PointCloud DownsampleVoxel(const PointCloud &in, float voxelSize);

} // namespace Engine::Features
