#pragma once

#include "Engine/Pipeline/Types.h" // Engine::Pipeline::Frame

#include "utilities/PointCloudIO.h" // util::LoadPly

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

// Frame reading lives with the Reconstruction stage -- the pipeline's FileFrameSource and any app that
// needs the raw clouds up front (e.g. a density precompute or a display layer) load them the SAME way,
// rather than each re-implementing PLY loading + camera estimation.
namespace Engine::Pipeline {

    // A camera-position hint for a world-registered cloud with no recorded pose: place the eye off the
    // surface along the mean normal, a few diagonals back, so view-dependent weighting has a sane origin.
    inline Eigen::Vector3f EstimateCameraHint(const std::vector<Eigen::Vector3f> &pts,
                                              const std::vector<Eigen::Vector3f> &nrm) {
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero(), meanN = Eigen::Vector3f::Zero();
        for (const auto &p: pts) centroid += p;
        for (const auto &n: nrm) meanN += n;
        centroid /= float(std::max<std::size_t>(1, pts.size()));
        Eigen::Vector3f mn = pts.empty() ? Eigen::Vector3f::Zero() : pts[0], mx = mn;
        for (const auto &p: pts) {
            mn = mn.cwiseMin(p);
            mx = mx.cwiseMax(p);
        }
        const float diag = (mx - mn).norm();
        meanN = meanN.norm() > 1e-6f ? meanN.normalized() : Eigen::Vector3f(0, 0, 1);
        return centroid + std::max(1.0f, 3.0f * diag) * meanN;
    }

    // World-space bounds of a loaded frame set, plus the largest frame's point count (sizes the map's
    // per-frame upload budget).
    struct FrameBounds {
        Eigen::Vector3f min = Eigen::Vector3f::Constant(1e30f);
        Eigen::Vector3f max = Eigen::Vector3f::Constant(-1e30f);
        std::size_t maxFramePoints = 0;
        bool valid = false;
        Eigen::Vector3f Center() const { return 0.5f * (min + max); }
        float Extent() const { return valid ? (max - min).norm() : 1.0f; }
    };

    // Load every PLY in `files` (in the given order) that has per-point normals into a Frame, estimating
    // a camera hint for each. Files that fail to load or lack normals are skipped. This is the single
    // reader the density precompute and any display layer share with the streaming FileFrameSource.
    inline std::vector<Frame> LoadFrames(const std::vector<std::string> &files) {
        std::vector<Frame> frames;
        frames.reserve(files.size());
        for (const std::string &path: files) {
            Frame fr;
            if (!util::LoadPly(path, fr.pts, fr.nrm) || fr.nrm.size() != fr.pts.size() || fr.pts.empty())
                continue;
            fr.cam = EstimateCameraHint(fr.pts, fr.nrm);
            frames.push_back(std::move(fr));
        }
        return frames;
    }

    inline FrameBounds ComputeBounds(const std::vector<Frame> &frames) {
        FrameBounds b;
        for (const Frame &fr: frames) {
            for (const Eigen::Vector3f &p: fr.pts) {
                b.min = b.min.cwiseMin(p);
                b.max = b.max.cwiseMax(p);
            }
            b.maxFramePoints = std::max(b.maxFramePoints, fr.pts.size());
            b.valid = b.valid || !fr.pts.empty();
        }
        return b;
    }

} // namespace Engine::Pipeline
