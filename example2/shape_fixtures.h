#pragma once

// Analytic shape fixtures for the SimpleTSDF-vs-DirectionalTSDF feature-preservation
// comparison (Task 2: correctness milestone). RENDER-FREE by contract: only Eigen + std,
// NO Engine::Render / PointCloudPass includes — the Task-2 target links only
// Engine::Spatial/Core. (Task 3 layers render/color helpers on top of this.)
//
// Provides, in namespace fixtures:
//   - Shape / Region enums and the per-viewpoint View struct.
//   - NearestDistance(shape, p): exact unsigned distance to the true surface (the oracle).
//   - ClassifyRegion(shape, p, voxelSize): flat / curved / edge bucket for per-region error.
//   - SampleViews(shape, voxelSize): 8 virtual views, each a front-facing surface-sample set,
//     grid coarsened so every view keeps <= ~800 points (large-N GPU-integrate budget).
//
// Shapes are centred at the origin. Cube: half-extent H = 1.5. Cylinder: axis +Z,
// radius R = 1.5, half-height HZ = 1.5. (World unit == 1 mm in this codebase.)

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fixtures {

    enum class Shape { Cube, Cylinder };
    enum class Region { Flat, Curved, Edge };

    struct View {
        Eigen::Vector3f camPos = Eigen::Vector3f::Zero();
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> normals; // parallel to points, unit outward
    };

    inline constexpr float kCubeHalf = 1.5f;   // H
    inline constexpr float kCylRadius = 1.5f;  // R
    inline constexpr float kCylHalfZ = 1.5f;   // HZ
    inline constexpr float kViewRadius = 5.0f; // camera sphere radius
    inline constexpr int kPerViewBudget = 800; // <= this many samples per Integrate call
    inline constexpr float kPi = 3.14159265358979323846f;

    // -------- exact unsigned distance to the true surface (ground-truth oracle) --------

    inline float NearestDistance(Shape s, const Eigen::Vector3f &p) {
        if (s == Shape::Cube) {
            // Standard signed box SDF, then |.|.
            const float qx = std::abs(p.x()) - kCubeHalf;
            const float qy = std::abs(p.y()) - kCubeHalf;
            const float qz = std::abs(p.z()) - kCubeHalf;
            const float ox = std::max(qx, 0.0f), oy = std::max(qy, 0.0f), oz = std::max(qz, 0.0f);
            const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
            const float inside = std::min(std::max(std::max(qx, qy), qz), 0.0f);
            return std::abs(outside + inside);
        }
        // Cylinder, axis +Z: radial and axial gaps combined as a 2D box SDF in (dr, dz).
        const float dr = std::hypot(p.x(), p.y()) - kCylRadius;
        const float dz = std::abs(p.z()) - kCylHalfZ;
        const float outside = std::hypot(std::max(dr, 0.0f), std::max(dz, 0.0f));
        const float inside = std::min(std::max(dr, dz), 0.0f);
        return std::abs(outside + inside);
    }

    // -------- exact outward unit normal at the nearest surface point (oracle for normals) --------
    // At sharp features (cube edges/corners, cylinder rim) the true normal is discontinuous; this
    // returns the box / 2D-box SDF gradient there (the bisector of the adjoining faces). Callers
    // treat Edge-region normal error as a reference number only (see the feature-compare table).

    inline Eigen::Vector3f NearestNormal(Shape s, const Eigen::Vector3f &p) {
        if (s == Shape::Cube) {
            const float qx = std::abs(p.x()) - kCubeHalf;
            const float qy = std::abs(p.y()) - kCubeHalf;
            const float qz = std::abs(p.z()) - kCubeHalf;
            Eigen::Vector3f n((p.x() >= 0.0f ? 1.0f : -1.0f) * std::max(qx, 0.0f),
                              (p.y() >= 0.0f ? 1.0f : -1.0f) * std::max(qy, 0.0f),
                              (p.z() >= 0.0f ? 1.0f : -1.0f) * std::max(qz, 0.0f));
            if (n.norm() > 1e-6f) return n.normalized();
            // Inside (or exactly on a face): nearest face = axis with the largest (closest-to-0) q.
            int axis = 0;
            float best = qx;
            if (qy > best) { best = qy; axis = 1; }
            if (qz > best) { best = qz; axis = 2; }
            Eigen::Vector3f e = Eigen::Vector3f::Zero();
            e[axis] = p[axis] >= 0.0f ? 1.0f : -1.0f;
            return e;
        }
        // Cylinder, axis +Z: 2D box SDF gradient in (radial, axial).
        const float r = std::hypot(p.x(), p.y());
        const float dr = r - kCylRadius;
        const float dz = std::abs(p.z()) - kCylHalfZ;
        const float zsign = p.z() >= 0.0f ? 1.0f : -1.0f;
        Eigen::Vector3f n = Eigen::Vector3f::Zero();
        if (r > 1e-6f) {
            n.x() = (p.x() / r) * std::max(dr, 0.0f);
            n.y() = (p.y() / r) * std::max(dr, 0.0f);
        }
        n.z() = zsign * std::max(dz, 0.0f);
        if (n.norm() > 1e-6f) return n.normalized();
        // Inside: side wall vs cap, whichever gap is closer to zero (the larger negative value).
        if (dr >= dz && r > 1e-6f) return Eigen::Vector3f(p.x() / r, p.y() / r, 0.0f);
        return Eigen::Vector3f(0.0f, 0.0f, zsign);
    }

    // Angle between two vectors in degrees (unit-normalized internally; 0 for a zero input).
    inline float NormalAngleDeg(const Eigen::Vector3f &a, const Eigen::Vector3f &b) {
        const float na = a.norm(), nb = b.norm();
        if (na < 1e-12f || nb < 1e-12f) return 0.0f;
        const float c = std::clamp(a.dot(b) / (na * nb), -1.0f, 1.0f);
        return std::acos(c) * 180.0f / kPi;
    }

    // -------- per-point region classification for the per-region error table --------

    inline Region ClassifyRegion(Shape s, const Eigen::Vector3f &p, float voxelSize) {
        const float band = 2.0f * voxelSize;
        if (s == Shape::Cube) {
            int nearFaces = 0;
            if (std::abs(std::abs(p.x()) - kCubeHalf) <= band) ++nearFaces;
            if (std::abs(std::abs(p.y()) - kCubeHalf) <= band) ++nearFaces;
            if (std::abs(std::abs(p.z()) - kCubeHalf) <= band) ++nearFaces;
            return nearFaces >= 2 ? Region::Edge : Region::Flat;
        }
        // Cylinder.
        const float r = std::hypot(p.x(), p.y());
        const bool nearCapPlane = std::abs(std::abs(p.z()) - kCylHalfZ) <= band;
        const bool nearRimRadius = std::abs(r - kCylRadius) <= band;
        if (nearRimRadius && nearCapPlane) return Region::Edge; // rim (side meets cap)
        if (nearCapPlane) return Region::Flat;                  // cap disk
        return Region::Curved;                                  // side wall
    }

    // -------- render-free color helpers (Task 3) --------
    //
    // Local Rgb, deliberately NOT tsdf_fixtures.h's Rgb (which drags in PointCloudPass.h ->
    // Engine::Render/Vulkan). shape_fixtures.h stays render-free by contract (see file header)
    // so tsdf_feature_compare's headless --dump path keeps linking only Engine::Spatial/Core;
    // the windowed app converts Rgb -> PointVertex.rgba itself.
    struct Rgb {
        uint8_t r, g, b;
    };

    // Sequential blue->red error colormap: t=0 (no error) -> blue, t=1 (>= maxErr) -> red.
    inline Rgb errorColor(float err, float maxErr) {
        const float t = std::clamp(maxErr > 0.0f ? err / maxErr : 0.0f, 0.0f, 1.0f);
        return {static_cast<uint8_t>(255.0f * t), static_cast<uint8_t>(60),
                static_cast<uint8_t>(255.0f * (1.0f - t))};
    }

    // Region-bucket color: Flat=grey, Curved=blue, Edge=red (matches the per-region table).
    inline Rgb regionColor(Region r) {
        switch (r) {
            case Region::Flat: return {200, 200, 200};
            case Region::Curved: return {80, 160, 230};
            case Region::Edge: return {240, 80, 80};
        }
        return {200, 200, 200};
    }

    namespace detail {

        // 8 viewpoint directions: cube-corner directions (+-1,+-1,+-1)/sqrt(3).
        inline std::array<Eigen::Vector3f, 8> ViewDirs() {
            std::array<Eigen::Vector3f, 8> dirs;
            int i = 0;
            for (int sx = -1; sx <= 1; sx += 2)
                for (int sy = -1; sy <= 1; sy += 2)
                    for (int sz = -1; sz <= 1; sz += 2)
                        dirs[i++] = Eigen::Vector3f(float(sx), float(sy), float(sz)).normalized();
            return dirs;
        }

        // Full-surface candidate samples at a given linear step (view-independent).
        inline void CubeCandidates(float step, std::vector<Eigen::Vector3f> &pts,
                                   std::vector<Eigen::Vector3f> &nrm) {
            pts.clear();
            nrm.clear();
            const float H = kCubeHalf;
            const int N = std::max(2, int(std::lround(2.0f * H / step)) + 1);
            auto coord = [&](int i) { return -H + (2.0f * H) * float(i) / float(N - 1); };
            for (int axis = 0; axis < 3; ++axis) {
                const int u = (axis + 1) % 3, v = (axis + 2) % 3;
                for (int sign = -1; sign <= 1; sign += 2) {
                    Eigen::Vector3f n = Eigen::Vector3f::Zero();
                    n[axis] = float(sign);
                    for (int iu = 0; iu < N; ++iu)
                        for (int iv = 0; iv < N; ++iv) {
                            Eigen::Vector3f p = Eigen::Vector3f::Zero();
                            p[axis] = float(sign) * H;
                            p[u] = coord(iu);
                            p[v] = coord(iv);
                            pts.push_back(p);
                            nrm.push_back(n);
                        }
                }
            }
        }

        inline void CylinderCandidates(float step, std::vector<Eigen::Vector3f> &pts,
                                       std::vector<Eigen::Vector3f> &nrm) {
            pts.clear();
            nrm.clear();
            const float R = kCylRadius, HZ = kCylHalfZ;
            // Side wall: theta x z grid, radial outward normal.
            const int Ntheta = std::max(8, int(std::lround(2.0f * kPi * R / step)));
            const int Nz = std::max(2, int(std::lround(2.0f * HZ / step)) + 1);
            for (int it = 0; it < Ntheta; ++it) {
                const float theta = 2.0f * kPi * float(it) / float(Ntheta);
                const float c = std::cos(theta), sn = std::sin(theta);
                for (int iz = 0; iz < Nz; ++iz) {
                    const float z = -HZ + (2.0f * HZ) * float(iz) / float(Nz - 1);
                    pts.emplace_back(R * c, R * sn, z);
                    nrm.emplace_back(c, sn, 0.0f);
                }
            }
            // Two caps: x,y grid clipped to the disk, +-Z normal.
            const int Nc = std::max(2, int(std::lround(2.0f * R / step)) + 1);
            auto ccoord = [&](int i) { return -R + (2.0f * R) * float(i) / float(Nc - 1); };
            for (int sign = -1; sign <= 1; sign += 2) {
                const Eigen::Vector3f n(0.0f, 0.0f, float(sign));
                for (int ix = 0; ix < Nc; ++ix)
                    for (int iy = 0; iy < Nc; ++iy) {
                        const float x = ccoord(ix), y = ccoord(iy);
                        if (std::hypot(x, y) > R) continue;
                        pts.emplace_back(x, y, float(sign) * HZ);
                        nrm.push_back(n);
                    }
            }
        }

        inline void Candidates(Shape s, float step, std::vector<Eigen::Vector3f> &pts,
                               std::vector<Eigen::Vector3f> &nrm) {
            if (s == Shape::Cube)
                CubeCandidates(step, pts, nrm);
            else
                CylinderCandidates(step, pts, nrm);
        }

    } // namespace detail

    // 8 views on a sphere of radius kViewRadius. Each view keeps only front-facing samples
    // (normal.dot(viewDir) > 0.1). The grid step is coarsened until every view's kept count
    // is <= kPerViewBudget, honouring the large-N GPU-integrate budget. A cube edge / cylinder
    // rim is deliberately seen from views on BOTH adjoining faces (that is what makes the
    // averaged SimpleTSDF round it, while the DirectionalTSDF keeps the layers separate).
    inline std::vector<View> SampleViews(Shape s, float voxelSize) {
        (void)voxelSize; // sampling density is set by the per-view budget, not the voxel size
        const auto dirs = detail::ViewDirs();
        float step = 0.15f;
        std::vector<View> views;
        for (int attempt = 0; attempt < 32; ++attempt) {
            std::vector<Eigen::Vector3f> cpts, cnrm;
            detail::Candidates(s, step, cpts, cnrm);
            views.clear();
            views.reserve(dirs.size());
            std::size_t maxKept = 0;
            for (const auto &d : dirs) {
                View view;
                view.camPos = d * kViewRadius;
                for (std::size_t i = 0; i < cpts.size(); ++i) {
                    const Eigen::Vector3f viewDir = (view.camPos - cpts[i]).normalized();
                    if (cnrm[i].dot(viewDir) > 0.1f) {
                        view.points.push_back(cpts[i]);
                        view.normals.push_back(cnrm[i]);
                    }
                }
                maxKept = std::max(maxKept, view.points.size());
                views.push_back(std::move(view));
            }
            if (maxKept <= std::size_t(kPerViewBudget)) break;
            step *= 1.15f; // coarsen and retry
        }
        return views;
    }

} // namespace fixtures
