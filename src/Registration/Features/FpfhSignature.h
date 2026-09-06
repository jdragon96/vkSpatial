#pragma once

// Fast Point Feature Histograms (Rusu, Blodow, Beetz 2009). Per-point 33-dim descriptor
// (3 angular features x 11 bins) computed on an Engine::Core::OrientedPointCloud. Pure CPU, header-only.
//
// Decoupled by design: consumes only Engine::Core::OrientedPointCloud + a Engine::Spatial::NeighborQuery, so the same code
// serves surfaces extracted from SimpleTSDF or DirectionalTSDF (or any other source). FPFH is a
// post-extraction descriptor — it is NOT updated at TSDF integrate time.
//
// The descriptor is translation/rotation invariant (features are angles between relative
// vectors and normals) and PCL-compatible (each 11-bin sub-block normalised to sum 100), so it
// can be cross-checked against Open3D/PCL.

#include "BVH/NeighborQuery.h"
#include "Engine/Core/OrientedPointCloud.h"

#include <Eigen/Core>
#include <Eigen/Geometry> // Vector3f::cross
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace Features {

    static constexpr int FPFH_BINS = 11;              // bins per angular feature
    static constexpr int FPFH_DIM = 3 * FPFH_BINS;    // 33

    struct FpfhConfig {
        // Neighbourhood radius in world units. Must exceed the surface sampling spacing; a good
        // default for TSDF surfaces is ~2.5 x voxelSize. The caller sets this per data scale.
        float radius = 0.25f;
    };

    struct FpfhSignature {
        std::array<float, FPFH_DIM> hist{}; // [0..10]=alpha, [11..21]=phi, [22..32]=theta
    };

    namespace fpfh_detail {

        // Darboux-frame angular features of the ordered pair (source s → target t).
        // alpha = v·n_t, phi = u·(p_t−p_s)/d, theta = atan2(w·n_t, u·n_t).
        inline void pairFeatures(const Eigen::Vector3f &ps, const Eigen::Vector3f &ns,
                                 const Eigen::Vector3f &pt, const Eigen::Vector3f &nt,
                                 float &alpha, float &phi, float &theta) {
            const Eigen::Vector3f dp = pt - ps;
            const float d = dp.norm();
            alpha = phi = theta = 0.0f;
            if (d < 1e-12f) return;
            const Eigen::Vector3f u = ns;
            Eigen::Vector3f v = dp.cross(u);
            const float vn = v.norm();
            if (vn < 1e-12f) return; // dp ∥ ns → degenerate frame
            v /= vn;
            const Eigen::Vector3f w = u.cross(v);
            alpha = v.dot(nt);
            phi = u.dot(dp) / d;
            theta = std::atan2(w.dot(nt), u.dot(nt));
        }

        inline int binOf(float x, float lo, float hi) {
            const int b = int((x - lo) / (hi - lo) * float(FPFH_BINS));
            return b < 0 ? 0 : (b >= FPFH_BINS ? FPFH_BINS - 1 : b);
        }

        // Normalise each 11-bin sub-block so it sums to 100 (PCL convention).
        inline void normalizeBlocks(std::array<float, FPFH_DIM> &h) {
            for (int b = 0; b < 3; ++b) {
                float s = 0.0f;
                for (int k = 0; k < FPFH_BINS; ++k) s += h[b * FPFH_BINS + k];
                if (s > 1e-12f) {
                    const float inv = 100.0f / s;
                    for (int k = 0; k < FPFH_BINS; ++k) h[b * FPFH_BINS + k] *= inv;
                }
            }
        }

    } // namespace fpfh_detail

    // Core: per-point FPFH using a caller-provided neighbourhood (built over cloud.points).
    inline std::vector<FpfhSignature> ComputeFPFH(const Engine::Core::OrientedPointCloud &cloud,
                                                  const FpfhConfig &cfg,
                                                  const Engine::Spatial::NeighborQuery &nn) {
        const size_t n = cloud.size();
        const float radius = cfg.radius;

        // ── Stage 1: SPFH (Simplified PFH) per point ──
        std::vector<std::array<float, FPFH_DIM>> spfh(n, std::array<float, FPFH_DIM>{});
        std::vector<uint32_t> idx;
        std::vector<float> dist;
        for (size_t i = 0; i < n; ++i) {
            nn.Radius(cloud.points[i], radius, idx, dist);
            auto &h = spfh[i];
            for (size_t m = 0; m < idx.size(); ++m) {
                const uint32_t j = idx[m];
                if (j == i || dist[m] < 1e-9f) continue; // skip self
                const Eigen::Vector3f dhat = (cloud.points[j] - cloud.points[i]) / dist[m];
                // Pick the source as the point whose normal aligns better with the connecting
                // line → symmetric feature for the pair (i,j).
                float alpha, phi, theta;
                if (std::fabs(cloud.normals[i].dot(dhat)) >= std::fabs(cloud.normals[j].dot(dhat)))
                    fpfh_detail::pairFeatures(cloud.points[i], cloud.normals[i],
                                              cloud.points[j], cloud.normals[j], alpha, phi, theta);
                else
                    fpfh_detail::pairFeatures(cloud.points[j], cloud.normals[j],
                                              cloud.points[i], cloud.normals[i], alpha, phi, theta);
                h[fpfh_detail::binOf(alpha, -1.0f, 1.0f)] += 1.0f;
                h[FPFH_BINS + fpfh_detail::binOf(phi, -1.0f, 1.0f)] += 1.0f;
                h[2 * FPFH_BINS + fpfh_detail::binOf(theta, -float(M_PI), float(M_PI))] += 1.0f;
            }
            fpfh_detail::normalizeBlocks(h);
        }

        // ── Stage 2: FPFH = SPFH_i + (1/k) Σ_j (1/d_ij) SPFH_j, then renormalise ──
        std::vector<FpfhSignature> out(n);
        for (size_t i = 0; i < n; ++i) {
            nn.Radius(cloud.points[i], radius, idx, dist);
            std::array<float, FPFH_DIM> acc{};
            int k = 0;
            for (size_t m = 0; m < idx.size(); ++m) {
                const uint32_t j = idx[m];
                if (j == i || dist[m] < 1e-9f) continue;
                const float w = 1.0f / dist[m];
                for (int c = 0; c < FPFH_DIM; ++c) acc[c] += w * spfh[j][c];
                ++k;
            }
            std::array<float, FPFH_DIM> f = spfh[i];
            if (k > 0)
                for (int c = 0; c < FPFH_DIM; ++c) f[c] += acc[c] / float(k);
            fpfh_detail::normalizeBlocks(f);
            out[i].hist = f;
        }
        return out;
    }

    // Convenience: builds a CPU uniform-grid neighbourhood internally (cell size = radius).
    inline std::vector<FpfhSignature> ComputeFPFH(const Engine::Core::OrientedPointCloud &cloud,
                                                  const FpfhConfig &cfg) {
        Engine::Spatial::CpuGridNeighborhood nn(cloud.points, cfg.radius);
        return ComputeFPFH(cloud, cfg, nn);
    }

} // namespace Features
