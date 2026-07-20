#pragma once

#include "Engine/Eval/ScanSampler.h"     // ScanFrame
#include "Engine/Eval/SyntheticSurface.h" // Surface

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Synthetic scan-dataset builder for validating TSDF / DirectionalTSDF (Integrate + Extract).
// A 3D object is placed at the centre; a trackball camera orbits it while pitching downward;
// each frame renders a depth image by sphere-tracing the analytic surface, so the captured
// points are exactly what the camera sees (correct occlusion). Frames are the same
// {points, normals, cameraPos} tuples that DirectionalTSDF::Integrate consumes, and are also
// written out as a per-frame PLY sequence + ground truth + manifest.
namespace Engine::Eval {

    // ── Camera trajectory ────────────────────────────────────────────────────────
    struct CameraPose {
        Eigen::Vector3f eye = Eigen::Vector3f::Zero();
        Eigen::Vector3f target = Eigen::Vector3f::Zero();
        Eigen::Vector3f up = Eigen::Vector3f(0, 1, 0);
    };

    struct TrackballParams {
        Eigen::Vector3f orbitCenter = Eigen::Vector3f::Zero();
        float radius = 3.0f;
        int numFrames = 60;
        float startElevationDeg = 80.0f; // camera high (looking down)
        float endElevationDeg = -20.0f;  // ...rotating downward
        float azimuthTurns = 1.5f;       // spiral: full turns over the sweep (0 = pure pitch)
        float startAzimuthDeg = 0.0f;
        Eigen::Vector3f up = Eigen::Vector3f(0, 1, 0);
    };

    // Ordered poses: elevation lerps start→end (downward), azimuth spins by azimuthTurns.
    inline std::vector<CameraPose> TrackballDownTrajectory(const TrackballParams &tp) {
        std::vector<CameraPose> poses;
        poses.reserve(std::max(0, tp.numFrames));
        const float deg2rad = float(M_PI) / 180.0f;
        for (int f = 0; f < tp.numFrames; ++f) {
            const float t = tp.numFrames > 1 ? float(f) / float(tp.numFrames - 1) : 0.0f;
            const float el = (tp.startElevationDeg + (tp.endElevationDeg - tp.startElevationDeg) * t) * deg2rad;
            const float az = (tp.startAzimuthDeg) * deg2rad + tp.azimuthTurns * 2.0f * float(M_PI) * t;
            const float ce = std::cos(el), se = std::sin(el);
            CameraPose pose;
            pose.eye = tp.orbitCenter +
                       tp.radius * Eigen::Vector3f(ce * std::cos(az), se, ce * std::sin(az));
            pose.target = tp.orbitCenter;
            pose.up = tp.up;
            poses.push_back(pose);
        }
        return poses;
    }

    // ── Depth-image capture (analytic SDF sphere-tracing) ────────────────────────
    struct CaptureParams {
        int width = 160;
        int height = 120;
        float fovYDeg = 55.0f;
        float tMin = 1e-3f;
        float tMax = 100.0f;
        float hitEps = 1e-4f;
        int maxSteps = 256;

        // Point-cloud capture (util::ObjectScanner file/points path) depth-buffer occlusion —
        // ignored by the analytic ray-march path above:
        int splatRadius = -1;       // occluder footprint px; <0 = auto (point cloud→1 fills holes, dense mesh→0)
        float occlusionEps = 0.02f; // relative depth tolerance for the "nearest" test (2%)
        float cosVisibility = 0.0f; // front-facing cutoff: normal·dir(p→eye) > this
    };

    // Renders one depth image and returns the visible surface points (+ normals) in world
    // space. Rays that never converge within tMax simply produce no point (background).
    inline ScanFrame CaptureDepthImage(const Surface &surface, const CameraPose &cam,
                                       const CaptureParams &cp) {
        ScanFrame frame;
        frame.cameraPos = cam.eye;

        const Eigen::Vector3f forward = (cam.target - cam.eye).normalized();
        Eigen::Vector3f right = forward.cross(cam.up);
        if (right.norm() < 1e-6f) right = forward.cross(Eigen::Vector3f(1, 0, 0)); // up ∥ forward
        right.normalize();
        const Eigen::Vector3f trueUp = right.cross(forward);

        const float tanHalfFovY = std::tan(0.5f * cp.fovYDeg * float(M_PI) / 180.0f);
        const float aspect = float(cp.width) / float(std::max(1, cp.height));

        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (int py = 0; py < cp.height; ++py) {
            const float ndcY = (1.0f - 2.0f * (float(py) + 0.5f) / float(cp.height)) * tanHalfFovY;
            for (int px = 0; px < cp.width; ++px) {
                const float ndcX = (2.0f * (float(px) + 0.5f) / float(cp.width) - 1.0f) * aspect * tanHalfFovY;
                const Eigen::Vector3f dir = (forward + ndcX * right + ndcY * trueUp).normalized();

                // Sphere-trace the unsigned distance field; from outside a solid this stops at
                // the nearest surface crossing → correct visibility/occlusion.
                float t = cp.tMin;
                bool hit = false;
                for (int s = 0; s < cp.maxSteps; ++s) {
                    const Eigen::Vector3f p = cam.eye + t * dir;
                    const float d = surface.Distance(p);
                    if (d < cp.hitEps) { hit = true; break; }
                    t += std::max(d, cp.hitEps);
                    if (t > cp.tMax) break;
                }
                if (!hit) continue;

                const Eigen::Vector3f p = cam.eye + t * dir;
                Eigen::Vector3f n = surface.NormalAt(p);
                if (n.dot(cam.eye - p) < 0.0f) n = -n; // orient toward camera
                frame.points.push_back(p);
                frame.normals.push_back(n);
                centroid += p;
            }
        }
        if (!frame.points.empty())
            frame.aabbCenterHint = centroid / float(frame.points.size());
        return frame;
    }

    // Full trackball scan: one ScanFrame per pose (empty frames skipped).
    inline std::vector<ScanFrame> GenerateTrackballScan(const Surface &surface,
                                                        const std::vector<CameraPose> &poses,
                                                        const CaptureParams &cp) {
        std::vector<ScanFrame> frames;
        frames.reserve(poses.size());
        for (const CameraPose &pose : poses) {
            ScanFrame frame = CaptureDepthImage(surface, pose, cp);
            if (!frame.points.empty()) frames.push_back(std::move(frame));
        }
        return frames;
    }

    // ── Dataset writer ───────────────────────────────────────────────────────────
    inline void WritePlyPointNormal(const std::string &path,
                                    const std::vector<Eigen::Vector3f> &points,
                                    const std::vector<Eigen::Vector3f> &normals) {
        std::ofstream f(path);
        if (!f.is_open()) throw std::runtime_error("WritePlyPointNormal: cannot open " + path);
        const bool hasN = normals.size() == points.size();
        f << "ply\nformat ascii 1.0\n"
          << "element vertex " << points.size() << "\n"
          << "property float x\nproperty float y\nproperty float z\n";
        if (hasN) f << "property float nx\nproperty float ny\nproperty float nz\n";
        f << "end_header\n";
        for (size_t i = 0; i < points.size(); ++i) {
            const Eigen::Vector3f &p = points[i];
            f << p.x() << ' ' << p.y() << ' ' << p.z();
            if (hasN) {
                const Eigen::Vector3f &n = normals[i];
                f << ' ' << n.x() << ' ' << n.y() << ' ' << n.z();
            }
            f << '\n';
        }
    }

    // Writes: <dir>/frame_%04d.ply (per-frame visible points+normals),
    //         <dir>/ground_truth.ply (dense GT surface), <dir>/manifest.txt (poses+intrinsics).
    // Returns the number of frames written.
    inline size_t WriteScanDataset(const std::string &dir, const std::string &objectDesc,
                                   const std::vector<ScanFrame> &frames,
                                   const std::vector<CameraPose> &poses,
                                   const CaptureParams &cp,
                                   const std::vector<Eigen::Vector3f> &groundTruth,
                                   const std::vector<Eigen::Vector3f> &groundTruthNormals = {}) {
        namespace fs = std::filesystem;
        fs::create_directories(dir);

        WritePlyPointNormal(dir + "/ground_truth.ply", groundTruth, groundTruthNormals);

        std::ofstream m(dir + "/manifest.txt");
        if (!m.is_open()) throw std::runtime_error("WriteScanDataset: cannot open manifest in " + dir);
        m << "# scan dataset manifest\n";
        m << "object " << objectDesc << "\n";
        m << "intrinsics width " << cp.width << " height " << cp.height
          << " fovYDeg " << cp.fovYDeg << "\n";
        m << "ground_truth ground_truth.ply " << groundTruth.size() << "\n";
        m << "frames " << frames.size() << "\n";
        m << "# columns: file  numPoints  eyeX eyeY eyeZ  tgtX tgtY tgtZ  upX upY upZ\n";

        for (size_t i = 0; i < frames.size(); ++i) {
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%04zu.ply", i);
            WritePlyPointNormal(dir + "/" + name, frames[i].points, frames[i].normals);

            // poses[] may be longer than frames[] (empty frames were skipped). Best-effort pair
            // by index; the eye is authoritative regardless (it is stored on the frame).
            const CameraPose &pz = i < poses.size() ? poses[i] : CameraPose{};
            const Eigen::Vector3f &e = frames[i].cameraPos;
            m << name << ' ' << frames[i].points.size() << "  "
              << e.x() << ' ' << e.y() << ' ' << e.z() << "  "
              << pz.target.x() << ' ' << pz.target.y() << ' ' << pz.target.z() << "  "
              << pz.up.x() << ' ' << pz.up.y() << ' ' << pz.up.z() << "\n";
        }
        return frames.size();
    }

} // namespace Engine::Eval
