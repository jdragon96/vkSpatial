#include "Engine/Spatial/AdaptiveVoxelGrid.h"
#include "Engine/Spatial/MarchingCubesTables.h"

#include <Eigen/Geometry> // Vector3f::cross
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace Engine::Spatial {

    namespace {

        // Hashes an integer voxel coordinate for the CPU Marching Cubes value map.
        struct IVec3Hash {
            size_t operator()(const std::array<int, 3> &v) const noexcept {
                size_t h = std::hash<int>()(v[0]);
                h = h * 31u + std::hash<int>()(v[1]);
                h = h * 31u + std::hash<int>()(v[2]);
                return h;
            }
        };

        using VoxelValueMap = std::unordered_map<std::array<int, 3>, float, IVec3Hash>;

        // Recovers the integer fine-voxel coordinate from a world-space centre; identical to
        // buildMixed()'s vcoord lambda (v = lround(center/h - 0.5) per axis).
        std::array<int, 3> fineCoordOf(const Eigen::Vector3f &center, float h) {
            return {
                    static_cast<int>(std::lround(center.x() / h - 0.5f)),
                    static_cast<int>(std::lround(center.y() / h - 0.5f)),
                    static_cast<int>(std::lround(center.z() / h - 0.5f))};
        }

        // Mirrors voxel_common.glsl's vertInterp: linear interpolation to the TSDF zero-crossing.
        Eigen::Vector3f vertInterp(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, float v1, float v2) {
            const float dv = v2 - v1;
            if (std::abs(dv) < 1e-6f) return (p1 + p2) * 0.5f;
            const float t = -v1 / dv;
            return p1 + t * (p2 - p1);
        }

        // Runs single-resolution Marching Cubes over every cube whose 8 corners are all present
        // in `values` (keyed by integer voxel coordinate), mirroring voxel_tsdf_mc.comp's
        // processCube/edgeTable/triTable logic (tables transcribed in MarchingCubesTables.h).
        // `cellSize` is the world-space size of one voxel at this level. Emitted vertices are
        // welded to unique positions and given per-vertex area-weighted normals (mirrors
        // SimpleTSDF::ExtractPointCloud's welding+normal step); the volume itself stores no
        // normals.
        AdaptiveMesh cpuMarchingCubes(const VoxelValueMap &values, float cellSize) {
            AdaptiveMesh mesh;
            if (values.empty()) return mesh;

            // MC edge index -> corner-index pair (edge e connects CORNER[a] and CORNER[b]);
            // matches voxel_tsdf_mc.comp's explicit ev[0..11] = vertInterp(p[a],p[b],...) list.
            static constexpr int kEdgeCorners[12][2] = {
                    {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

            // Candidate cube-base coordinates: for each occupied voxel, the 8 cubes that could
            // have it as one of their corners (mirrors voxel_tsdf_mc.comp main()'s dx,dy,dz in
            // {-1,0} sweep over each sufficiently-observed voxel).
            std::unordered_set<std::array<int, 3>, IVec3Hash> bases;
            bases.reserve(values.size() * 8u);
            for (const auto &kv : values) {
                const auto &coord = kv.first;
                for (int dx = -1; dx <= 0; ++dx)
                    for (int dy = -1; dy <= 0; ++dy)
                        for (int dz = -1; dz <= 0; ++dz)
                            bases.insert({coord[0] + dx, coord[1] + dy, coord[2] + dz});
            }

            struct RawTri {
                Eigen::Vector3f a, b, c;
            };
            std::vector<RawTri> tris;

            for (const auto &base : bases) {
                float sdf[8];
                bool complete = true;
                for (int c = 0; c < 8 && complete; ++c) {
                    const std::array<int, 3> k{base[0] + mc::CORNER[c][0], base[1] + mc::CORNER[c][1],
                                                base[2] + mc::CORNER[c][2]};
                    const auto it = values.find(k);
                    if (it == values.end()) {
                        complete = false;
                        break;
                    }
                    sdf[c] = it->second;
                }
                if (!complete) continue; // a missing corner: skip this cell (spec Task 3 rule)

                int cubeIndex = 0;
                for (int c = 0; c < 8; ++c)
                    if (sdf[c] < 0.0f) cubeIndex |= (1 << c);

                const int et = mc::edgeTable[cubeIndex];
                if (et == 0) continue;

                Eigen::Vector3f p[8];
                for (int c = 0; c < 8; ++c)
                    p[c] = Eigen::Vector3f(float(base[0] + mc::CORNER[c][0]), float(base[1] + mc::CORNER[c][1]),
                                           float(base[2] + mc::CORNER[c][2])) *
                           cellSize;

                Eigen::Vector3f ev[12];
                for (int e = 0; e < 12; ++e)
                    if (et & (1 << e)) {
                        const int a = kEdgeCorners[e][0], b = kEdgeCorners[e][1];
                        ev[e] = vertInterp(p[a], p[b], sdf[a], sdf[b]);
                    }

                const int triBase = cubeIndex * 16;
                for (int i = 0; i < 15; i += 3) {
                    const int ei0 = mc::triTable[triBase + i];
                    if (ei0 == -1) break;
                    const int ei1 = mc::triTable[triBase + i + 1];
                    const int ei2 = mc::triTable[triBase + i + 2];
                    // Swap ei1/ei2 for outward-facing winding (mirrors voxel_tsdf_mc.comp).
                    tris.push_back({ev[ei0], ev[ei2], ev[ei1]});
                }
            }

            if (tris.empty()) return mesh;

            // Weld coincident MC vertices onto a fine tolerance grid, accumulating area-weighted
            // triangle normals -- mirrors SimpleTSDF::ExtractPointCloud.
            const float weld = std::max(cellSize * 1e-3f, 1e-6f);
            auto key = [weld](const Eigen::Vector3f &p) -> uint64_t {
                constexpr int64_t kBias = 1 << 20;
                constexpr uint64_t kMask = (1ull << 21) - 1;
                const int64_t qx = int64_t(std::llround(p.x() / weld)) + kBias;
                const int64_t qy = int64_t(std::llround(p.y() / weld)) + kBias;
                const int64_t qz = int64_t(std::llround(p.z() / weld)) + kBias;
                return (uint64_t(qx) & kMask) | ((uint64_t(qy) & kMask) << 21) | ((uint64_t(qz) & kMask) << 42);
            };

            std::unordered_map<uint64_t, uint32_t> lut;
            std::vector<Eigen::Vector3f> nAccum;
            for (const auto &t : tris) {
                const Eigen::Vector3f fn = (t.b - t.a).cross(t.c - t.a); // area-weighted (|fn|=2*area)
                std::array<uint32_t, 3> vi{};
                int j = 0;
                for (const Eigen::Vector3f &p : {t.a, t.b, t.c}) {
                    const uint64_t k = key(p);
                    const auto it = lut.find(k);
                    uint32_t v;
                    if (it == lut.end()) {
                        v = static_cast<uint32_t>(mesh.vertices.size());
                        lut.emplace(k, v);
                        mesh.vertices.push_back(p);
                        nAccum.push_back(Eigen::Vector3f::Zero());
                    } else {
                        v = it->second;
                    }
                    nAccum[v] += fn;
                    vi[j++] = v;
                }
                mesh.triangles.emplace_back(int(vi[0]), int(vi[1]), int(vi[2]));
            }

            mesh.normals.resize(mesh.vertices.size());
            for (size_t i = 0; i < mesh.vertices.size(); ++i) {
                const float len = nAccum[i].norm();
                mesh.normals[i] = len > 1e-12f ? Eigen::Vector3f(nAccum[i] / len) : Eigen::Vector3f(0, 0, 1);
            }
            return mesh;
        }

    } // namespace

    void AdaptiveVoxelGrid::Build(Engine::Core::Context &ctx, float fineVoxelSize, float truncation,
                                  uint32_t hashCapacity, uint32_t maxPoints) {
        m_h = fineVoxelSize;
        m_trunc = truncation;
        m_fine.Build(ctx, fineVoxelSize, truncation, hashCapacity, maxPoints);
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::Integrate(const std::vector<Eigen::Vector3f> &points,
                                      const Eigen::Vector3f &cameraPos) {
        m_fine.Integrate(points, cameraPos);
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::Reset() {
        m_fine.Reset();
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::SetVarianceThreshold(float sigma2) {
        m_threshold = sigma2;
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::SetVariancePercentile(float p) {
        m_threshold = -1.0f; // re-enable percentile mode
        m_percentile = p;
        m_mixedDirty = true;
    }

    void AdaptiveVoxelGrid::SetMinOccupancy(uint32_t n) {
        m_minOcc = n;
        m_mixedDirty = true;
    }

    std::vector<MixedVoxel> AdaptiveVoxelGrid::DownloadMixedVoxels() {
        if (m_mixedDirty) buildMixed();
        return m_mixed;
    }

    size_t AdaptiveVoxelGrid::FineCount() const {
        if (m_mixedDirty) const_cast<AdaptiveVoxelGrid *>(this)->buildMixed();
        return m_fineCount;
    }

    size_t AdaptiveVoxelGrid::CoarseCount() const {
        if (m_mixedDirty) const_cast<AdaptiveVoxelGrid *>(this)->buildMixed();
        return m_coarseCount;
    }

    AdaptiveMesh AdaptiveVoxelGrid::ExtractMesh() {
        if (m_mixedDirty) buildMixed();

        // Task 3: single-resolution CPU MC over FINE (level==0) voxels only; coarse cells
        // (level==1) are handled in Task 4's multi-resolution extension.
        VoxelValueMap values;
        values.reserve(m_mixed.size());
        for (const auto &mv : m_mixed) {
            if (mv.level != 0) continue;
            values[fineCoordOf(mv.center, m_h)] = mv.tsdf;
        }
        return cpuMarchingCubes(values, m_h);
    }

    // Recovers each fine voxel's integer coordinate from its world-space centre, buckets
    // fine voxels into 2x2x2 coarse blocks (floor-divided coord, correct for negatives), and
    // coarsens a block to a single weight-averaged voxel iff it is sufficiently observed
    // (count >= m_minOcc) AND sufficiently flat (mean per-voxel variance < theta). theta is
    // either the fixed m_threshold (>=0) or, in percentile mode (m_threshold < 0), the
    // m_percentile-th quantile of the observed per-voxel variances.
    void AdaptiveVoxelGrid::buildMixed() {
        const std::vector<VoxelStat> vox = m_fine.DownloadVoxels();

        auto vcoord = [&](const Eigen::Vector3f &c) {
            return Eigen::Vector3i(
                    static_cast<int>(std::lround(c.x() / m_h - 0.5f)),
                    static_cast<int>(std::lround(c.y() / m_h - 0.5f)),
                    static_cast<int>(std::lround(c.z() / m_h - 0.5f)));
        };

        float theta = m_threshold;
        if (theta < 0.0f) {
            std::vector<float> s;
            s.reserve(vox.size());
            for (const auto &v : vox) s.push_back(v.variance);
            std::sort(s.begin(), s.end());
            if (s.empty()) {
                theta = 0.0f;
            } else {
                const size_t idx = std::min(s.size() - 1, static_cast<size_t>(m_percentile * s.size()));
                const float qv = s[idx];
                // Escape any exact-tie plateau at the p-th quantile value: real fixture data
                // has a large mass of EXACTLY-zero-variance fine voxels (voxels touched by a
                // single observation have zero sample variance by definition), so for common
                // percentiles qv==0 and a literal "meanVar < qv" would coarsen nothing --
                // every tied (zero-variance) block fails a strict "< 0" test. Percentile mode
                // means "coarsen the low-variance p-fraction, ties included", so theta is
                // redefined as the smallest observed variance strictly greater than qv; the
                // comparison below stays a single, uniform "meanVar < theta" (ties at/below qv
                // now satisfy it), matching fixed-threshold mode's semantics exactly when there
                // are no ties.
                const auto it = std::upper_bound(s.begin(), s.end(), qv);
                theta = (it != s.end()) ? *it : std::nextafter(qv, std::numeric_limits<float>::infinity());
            }
        }

        // Floor division (a / b rounded toward -infinity), correct for negative coords --
        // plain C++ integer division truncates toward zero, which would bucket e.g. -1 and 0
        // into different-signed halves of the same coarse block.
        auto fdiv = [](int a, int b) {
            const int q = a / b, r = a % b;
            return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
        };

        struct Blk {
            std::vector<size_t> idx;
        };
        std::map<std::array<int, 3>, Blk> blocks;
        for (size_t i = 0; i < vox.size(); ++i) {
            const Eigen::Vector3i vc = vcoord(vox[i].center);
            blocks[{fdiv(vc.x(), 2), fdiv(vc.y(), 2), fdiv(vc.z(), 2)}].idx.push_back(i);
        }

        m_mixed.clear();
        m_fineCount = 0;
        m_coarseCount = 0;
        for (const auto &[ck, blk] : blocks) {
            double vs = 0.0;
            float wsum = 0.0f, dwsum = 0.0f;
            for (size_t i : blk.idx) {
                vs += double(vox[i].variance);
                wsum += vox[i].weight;
                dwsum += vox[i].weight * vox[i].tsdf;
            }
            const float meanVar = static_cast<float>(vs / double(blk.idx.size()));
            if (blk.idx.size() >= m_minOcc && meanVar < theta) {
                // Coarse voxel: centre of the 2x2x2 fine block, weight-averaged value.
                const Eigen::Vector3f cc(
                        (ck[0] * 2 + 1) * m_h, (ck[1] * 2 + 1) * m_h, (ck[2] * 2 + 1) * m_h);
                m_mixed.push_back({cc, dwsum / std::max(wsum, 1e-6f), wsum, 2.0f * m_h, 1});
                ++m_coarseCount;
            } else {
                for (size_t i : blk.idx) {
                    m_mixed.push_back({vox[i].center, vox[i].tsdf, vox[i].weight, m_h, 0});
                    ++m_fineCount;
                }
            }
        }
        m_mixedDirty = false;
    }

} // namespace Engine::Spatial
