#pragma once

#include "AsyncTsdfMapper.h" // asyncmap::MapperFrame, asyncmap::MapSnapshot

#include <Eigen/Core>
#include <Eigen/Geometry>

// Shared data types for the Track/Map/Render reconstruction pipeline (example2). Reuses the
// AsyncTsdfMapper types: MapperFrame is the raw captured cloud, MapSnapshot is the model handoff.
namespace pipeline {

    using Frame = asyncmap::MapperFrame;      // points, normals, cam (sensor/world camera hint)
    using ModelSnapshot = asyncmap::MapSnapshot;

    // A frame plus the pose Track resolved for it.
    struct TrackedFrame {
        Frame frame;
        Eigen::Isometry3f pose = Eigen::Isometry3f::Identity(); // sensor -> world
        Eigen::Vector3f cameraWorld = Eigen::Vector3f::Zero();  // world camera position (view weight)
    };

} // namespace pipeline
