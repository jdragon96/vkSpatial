#pragma once

#include "Engine/Eval/SyntheticSurface.h"

#include <Eigen/Core>
#include <cmath>
#include <utility>
#include <vector>

namespace Engine::Eval {

    struct ScanFrame {
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> normals;
        Eigen::Vector3f cameraPos = Eigen::Vector3f::Zero();
        Eigen::Vector3f aabbCenterHint = Eigen::Vector3f::Zero();
    };

    struct OrbitParams {
        std::vector<Eigen::Vector3f> surfaceSamples; // candidate surface points to scan
        float cameraRadius = 3.0f;
        Eigen::Vector3f orbitCenter = Eigen::Vector3f::Zero();
        int numElevation = 9;        // latitude rings, −80°..+80°
        int numAzimuth = 36;         // longitudes per ring
        float cosVisibility = 0.15f; // reject grazing samples
    };

    // One ScanFrame per camera pose. A surface sample is included in a frame if its outward
    // normal faces the camera (normal · dir(sample→cam) > cosVisibility). Mirrors the
    // visibility logic in example2/voxel_tsdf_mc.cpp so multi-view running-average and
    // directional layering are exercised. Empty frames are skipped.
    inline std::vector<ScanFrame> GenerateOrbitScan(const Surface &surface,
                                                    const OrbitParams &params) {
        std::vector<ScanFrame> frames;
        for (int el = 0; el < params.numElevation; ++el) {
            const float elDeg = params.numElevation > 1
                                        ? -80.0f + 160.0f * float(el) / float(params.numElevation - 1)
                                        : 0.0f;
            const float elRad = elDeg * float(M_PI) / 180.0f;
            const float cosEl = std::cos(elRad), sinEl = std::sin(elRad);

            for (int az = 0; az < params.numAzimuth; ++az) {
                const float azRad = 2.0f * float(M_PI) * float(az) / float(params.numAzimuth);
                const Eigen::Vector3f cam =
                        params.orbitCenter +
                        params.cameraRadius * Eigen::Vector3f(cosEl * std::cos(azRad), sinEl,
                                                              cosEl * std::sin(azRad));

                ScanFrame frame;
                frame.cameraPos = cam;
                Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
                for (const auto &p : params.surfaceSamples) {
                    const Eigen::Vector3f toCam = (cam - p);
                    const float len = toCam.norm();
                    if (len < 1e-8f) continue;
                    const Eigen::Vector3f dir = toCam / len;
                    if (surface.NormalAt(p).dot(dir) > params.cosVisibility) {
                        frame.points.push_back(p);
                        frame.normals.push_back(surface.NormalAt(p));
                        centroid += p;
                    }
                }
                if (frame.points.empty()) continue;
                frame.aabbCenterHint = centroid / float(frame.points.size());
                frames.push_back(std::move(frame));
            }
        }
        return frames;
    }

} // namespace Engine::Eval
