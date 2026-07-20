#pragma once

// ObjectScanner — records a synthetic multi-view scan of a 3D object (analytic Surface or a
// 3D point-cloud file) with a trackball-down camera, then Replays the recorded frames through
// a callback. Scanning is kept separate from reconstruction: the callback is where the caller
// drives its own TSDF (e.g. `scanner.Replay([&](size_t, const Frame& f){
// tsdf.Integrate(f.points, f.normals, f.cameraPos, f.aabbCenterHint); })`).
//
// Two capture backends:
//   - analytic Surface  → exact ray-marched depth images (Engine::Eval::CaptureDepthImage).
//   - point cloud/file  → depth-buffer occlusion (front-facing + nearest-per-pixel).

#include "Engine/Eval/ScanDataset.h" // TrackballParams, CaptureParams, CameraPose, ScanFrame, writers
#include "Engine/Eval/SyntheticSurface.h"
#include "utilities/PlyMesh.h" // TriMesh, LoadPlyMesh, ComputeVertexNormals

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace util {

    class ObjectScanner {
    public:
        using Frame = Engine::Eval::ScanFrame;
        using ReplayCallback = std::function<void(size_t frameIndex, const Frame &)>;

        // ── Record from an analytic object: exact ray-marched depth images ──
        void Record(const Engine::Eval::Surface &surface,
                    const Engine::Eval::TrackballParams &trajectory,
                    const Engine::Eval::CaptureParams &capture = {},
                    uint32_t groundTruthCount = 20000) {
            reset(capture);
            const auto poses = Engine::Eval::TrackballDownTrajectory(trajectory);
            for (const auto &pose : poses) {
                Frame f = Engine::Eval::CaptureDepthImage(surface, pose, capture);
                if (!f.points.empty()) { m_frames.push_back(std::move(f)); m_framePoses.push_back(pose); }
            }
            m_groundTruth = surface.SampleDense(groundTruthCount);
            m_groundTruthNormals.clear();
            m_groundTruthNormals.reserve(m_groundTruth.size());
            for (const auto &p : m_groundTruth) m_groundTruthNormals.push_back(surface.NormalAt(p));
        }

        // ── Record from an in-memory point cloud (points + matching normals) ──
        void RecordFromPoints(const std::vector<Eigen::Vector3f> &points,
                              const std::vector<Eigen::Vector3f> &normals,
                              const Engine::Eval::TrackballParams &trajectory,
                              const Engine::Eval::CaptureParams &capture = {}) {
            if (normals.size() != points.size())
                throw std::runtime_error("ObjectScanner: points and normals must be the same length");
            Engine::Eval::CaptureParams cp = capture;
            if (cp.splatRadius < 0) cp.splatRadius = 1; // sparse cloud: splat fills sampling holes
            reset(cp);
            const auto poses = Engine::Eval::TrackballDownTrajectory(trajectory);
            for (const auto &pose : poses) {
                Frame f = captureFromPoints(points, normals, pose, cp);
                if (!f.points.empty()) { m_frames.push_back(std::move(f)); m_framePoses.push_back(pose); }
            }
            m_groundTruth = points; // the input cloud is the ground truth
            m_groundTruthNormals = normals;
        }

        // ── Record from a 3D file: triangle mesh (has faces) or point cloud (has normals) ──
        // The camera trajectory is auto-fitted to the geometry's bounding box (files are at
        // arbitrary scale/position), preserving numFrames/elevations/turns.
        void RecordFromFile(const std::string &plyPath,
                            const Engine::Eval::TrackballParams &trajectory,
                            const Engine::Eval::CaptureParams &capture = {}) {
            if (plyHasFaces(plyPath)) { RecordFromMesh(plyPath, trajectory, capture); return; }
            std::vector<Eigen::Vector3f> pts, normals;
            LoadPlyPointNormal(plyPath, pts, normals);
            RecordFromPoints(pts, normals, fitTrajectory(trajectory, pts), capture);
        }

        // ── Record from a triangle-mesh file: computes per-vertex normals, scans the vertices
        // as a dense oriented point cloud via depth-buffer occlusion. Camera auto-fitted. ──
        void RecordFromMesh(const std::string &meshPath,
                            const Engine::Eval::TrackballParams &trajectory,
                            const Engine::Eval::CaptureParams &capture = {}) {
            TriMesh mesh;
            LoadPlyMesh(meshPath, mesh);
            if (mesh.vertices.empty())
                throw std::runtime_error("ObjectScanner: mesh has no vertices: " + meshPath);
            const std::vector<Eigen::Vector3f> normals = ComputeVertexNormals(mesh);
            // Winding may be inconsistent → disable the front-face cull and let depth-buffer
            // occlusion carry visibility (output normals are still oriented toward the camera).
            Engine::Eval::CaptureParams cp = capture;
            cp.cosVisibility = -1.0f;
            // Dense vertices >> pixels ⇒ no sampling holes; splatting would only over-occlude
            // silhouettes and drop visible points, so default to no splat.
            if (cp.splatRadius < 0) cp.splatRadius = 0;
            RecordFromPoints(mesh.vertices, normals, fitTrajectory(trajectory, mesh.vertices), cp);
        }

        // ── Replay: feed each recorded frame to the callback (in capture order) ──
        void Replay(const ReplayCallback &callback) const {
            for (size_t i = 0; i < m_frames.size(); ++i) callback(i, m_frames[i]);
        }

        const std::vector<Frame> &Frames() const { return m_frames; }
        const std::vector<Engine::Eval::CameraPose> &FramePoses() const { return m_framePoses; }
        const std::vector<Eigen::Vector3f> &GroundTruth() const { return m_groundTruth; }
        const std::vector<Eigen::Vector3f> &GroundTruthNormals() const { return m_groundTruthNormals; }

        // Persist the recording as a reusable dataset (frame_%04d.ply + ground_truth.ply + manifest).
        size_t WriteDataset(const std::string &dir, const std::string &objectDesc) const {
            return Engine::Eval::WriteScanDataset(dir, objectDesc, m_frames, m_framePoses,
                                                  m_capture, m_groundTruth, m_groundTruthNormals);
        }

        // Load an ASCII PLY point cloud with x y z nx ny nz. Throws if normals are absent.
        static void LoadPlyPointNormal(const std::string &path,
                                       std::vector<Eigen::Vector3f> &points,
                                       std::vector<Eigen::Vector3f> &normals) {
            std::ifstream f(path);
            if (!f.is_open()) throw std::runtime_error("ObjectScanner: cannot open " + path);
            std::string line;
            size_t count = 0;
            bool hasNormals = false, ascii = false;
            std::vector<std::string> props;
            while (std::getline(f, line)) {
                std::istringstream ss(line);
                std::string tok;
                ss >> tok;
                if (tok == "format") { std::string fmt; ss >> fmt; ascii = (fmt == "ascii"); }
                else if (tok == "element") { std::string e; ss >> e; if (e == "vertex") ss >> count; }
                else if (tok == "property") { std::string t, name; ss >> t >> name; props.push_back(name); }
                else if (tok == "end_header") break;
            }
            hasNormals = std::find(props.begin(), props.end(), "nx") != props.end();
            if (!ascii) throw std::runtime_error("ObjectScanner: only ASCII PLY is supported: " + path);
            if (!hasNormals)
                throw std::runtime_error("ObjectScanner: PLY needs per-point normals (nx ny nz): " + path);
            points.clear(); normals.clear();
            points.reserve(count); normals.reserve(count);
            for (size_t i = 0; i < count && std::getline(f, line); ++i) {
                std::istringstream ss(line);
                float x, y, z, nx, ny, nz;
                ss >> x >> y >> z >> nx >> ny >> nz;
                points.emplace_back(x, y, z);
                normals.emplace_back(nx, ny, nz);
            }
        }

    private:
        std::vector<Frame> m_frames;
        std::vector<Engine::Eval::CameraPose> m_framePoses; // aligned 1:1 with m_frames
        std::vector<Eigen::Vector3f> m_groundTruth;
        std::vector<Eigen::Vector3f> m_groundTruthNormals;
        Engine::Eval::CaptureParams m_capture;

        void reset(const Engine::Eval::CaptureParams &capture) {
            m_frames.clear();
            m_framePoses.clear();
            m_groundTruth.clear();
            m_groundTruthNormals.clear();
            m_capture = capture;
        }

        // True if the PLY header declares a face element (→ triangle mesh, not a point cloud).
        static bool plyHasFaces(const std::string &path) {
            std::ifstream f(path, std::ios::binary);
            std::string line;
            while (std::getline(f, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.rfind("element face", 0) == 0) return true;
                if (line == "end_header") break;
            }
            return false;
        }

        // Auto-fit the orbit to a loaded geometry's bounding box (keeps frames/elevations/turns).
        static Engine::Eval::TrackballParams fitTrajectory(Engine::Eval::TrackballParams traj,
                                                           const std::vector<Eigen::Vector3f> &pts) {
            if (pts.empty()) return traj;
            Eigen::Vector3f mn = pts[0], mx = pts[0];
            for (const auto &p : pts) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
            traj.orbitCenter = 0.5f * (mn + mx);
            traj.radius = std::max(1e-3f, (mx - mn).norm() * 1.2f); // camera outside the bbox
            return traj;
        }

        // Depth-buffer visibility of a point cloud from one camera pose: project all points,
        // keep the nearest per pixel (splatted to fill sampling holes), then a front-facing
        // point is visible if it is within occlusionEps of its pixel's nearest depth.
        static Frame captureFromPoints(const std::vector<Eigen::Vector3f> &pts,
                                       const std::vector<Eigen::Vector3f> &normals,
                                       const Engine::Eval::CameraPose &cam,
                                       const Engine::Eval::CaptureParams &cp) {
            Frame frame;
            frame.cameraPos = cam.eye;

            const Eigen::Vector3f forward = (cam.target - cam.eye).normalized();
            Eigen::Vector3f right = forward.cross(cam.up);
            if (right.norm() < 1e-6f) right = forward.cross(Eigen::Vector3f(1, 0, 0));
            right.normalize();
            const Eigen::Vector3f trueUp = right.cross(forward);

            const float tanHalfFovY = std::tan(0.5f * cp.fovYDeg * float(M_PI) / 180.0f);
            const float aspect = float(cp.width) / float(std::max(1, cp.height));
            const int W = cp.width, H = cp.height;

            // Project p to (camZ, px, py); returns false if outside the frustum / behind camera.
            auto project = [&](const Eigen::Vector3f &p, float &camZ, int &px, int &py) -> bool {
                const Eigen::Vector3f rel = p - cam.eye;
                camZ = rel.dot(forward);
                if (camZ <= cp.tMin) return false;
                const float u = rel.dot(right) / (camZ * aspect * tanHalfFovY);
                const float v = rel.dot(trueUp) / (camZ * tanHalfFovY);
                if (u < -1.0f || u > 1.0f || v < -1.0f || v > 1.0f) return false;
                px = int((u + 1.0f) * 0.5f * float(W));
                py = int((1.0f - v) * 0.5f * float(H));
                px = std::min(std::max(px, 0), W - 1);
                py = std::min(std::max(py, 0), H - 1);
                return true;
            };

            std::vector<float> zbuf(size_t(W) * H, std::numeric_limits<float>::infinity());
            const int r = std::max(0, cp.splatRadius);
            for (size_t i = 0; i < pts.size(); ++i) {
                float camZ; int px, py;
                if (!project(pts[i], camZ, px, py)) continue;
                for (int dy = -r; dy <= r; ++dy)
                    for (int dx = -r; dx <= r; ++dx) {
                        const int nx = px + dx, ny = py + dy;
                        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                        float &z = zbuf[size_t(ny) * W + nx];
                        if (camZ < z) z = camZ;
                    }
            }

            Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
            for (size_t i = 0; i < pts.size(); ++i) {
                float camZ; int px, py;
                if (!project(pts[i], camZ, px, py)) continue;
                const Eigen::Vector3f toEye = cam.eye - pts[i];
                const float len = toEye.norm();
                if (len < 1e-8f) continue;
                const Eigen::Vector3f dirEye = toEye / len;
                // Front-facing cull (disabled when cosVisibility <= -1, e.g. meshes with
                // uncertain winding) — depth-buffer occlusion enforces visibility regardless.
                if (cp.cosVisibility > -1.0f && normals[i].dot(dirEye) <= cp.cosVisibility) continue;
                const float nearest = zbuf[size_t(py) * W + px];
                if (camZ > nearest * (1.0f + cp.occlusionEps)) continue; // occluded
                Eigen::Vector3f nOut = normals[i];
                if (nOut.dot(dirEye) < 0.0f) nOut = -nOut; // orient toward camera
                frame.points.push_back(pts[i]);
                frame.normals.push_back(nOut);
                centroid += pts[i];
            }
            if (!frame.points.empty())
                frame.aabbCenterHint = centroid / float(frame.points.size());
            return frame;
        }
    };

} // namespace util
