#include "Engine/Spatial/AdaptiveVoxelGrid.h"
#include "Engine/Spatial/Extraction/MarchingCubesCore.h"
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

        using Engine::Spatial::Extraction::core::IVec3Hash;
        using VoxelValueMap = Engine::Spatial::Extraction::core::VoxelValueMap;

        // Recovers the integer voxel coordinate (in units of `cellSize`) from a world-space
        // centre; identical to buildMixed()'s vcoord lambda (v = lround(center/cellSize - 0.5)
        // per axis). Used for fine voxels (cellSize=h) and coarse voxels (cellSize=2h) alike --
        // both levels share the same "index -> (index+0.5)*cellSize" centring convention, so a
        // coarse block's centre recovers its block index exactly like a fine voxel's centre
        // recovers its fine index.
        std::array<int, 3> latticeCoordOf(const Eigen::Vector3f &center, float cellSize) {
            return {
                    static_cast<int>(std::lround(center.x() / cellSize - 0.5f)),
                    static_cast<int>(std::lround(center.y() / cellSize - 0.5f)),
                    static_cast<int>(std::lround(center.z() / cellSize - 0.5f))};
        }

        // Floor division (a/b rounded toward -infinity), correct for negative coords -- plain
        // C++ integer division truncates toward zero, which would bucket e.g. -1 and 0 into
        // different-signed halves of the same coarse block.
        int floorDiv(int a, int b) {
            const int q = a / b, r = a % b;
            return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
        }

        std::array<int, 3> floorDiv2(const std::array<int, 3> &v) {
            return {floorDiv(v[0], 2), floorDiv(v[1], 2), floorDiv(v[2], 2)};
        }

        // Cross-resolution, finer-favoring corner sampler (spec Sec.5 / plan Task 4 Step 3):
        // `sampleAtFine(v)` looks up a corner expressed in FINE-lattice integer units, trying
        // the fine map first and falling back to the coarse map (via floor-div-by-2) if the
        // fine sample is absent (e.g. that corner sits inside a coarsened block).
        // `sampleAtCoarse(ck)` looks up a corner expressed in COARSE-lattice integer units by
        // converting to the equivalent fine coordinate (ck*2) and reusing sampleAtFine -- this
        // is what lets a coarse cube's corner pick up an actual finer value when its neighbour
        // block was kept fine rather than coarsened (finer-favoring).
        struct CornerSampler {
            const VoxelValueMap *fine;
            const VoxelValueMap *coarse;

            bool sampleAtFine(const std::array<int, 3> &v, float &out) const {
                if (const auto it = fine->find(v); it != fine->end()) {
                    out = it->second;
                    return true;
                }
                if (const auto jt = coarse->find(floorDiv2(v)); jt != coarse->end()) {
                    out = jt->second;
                    return true;
                }
                return false;
            }

            bool sampleAtCoarse(const std::array<int, 3> &ck, float &out) const {
                return sampleAtFine({ck[0] * 2, ck[1] * 2, ck[2] * 2}, out);
            }
        };

        // Collects a VoxelValueMap's keys into a vector, the input core::CandidateBases expects
        // (raw triangle generation and welding themselves now live in the shared core --
        // MarchingCubesCore.h/.cpp -- reused verbatim here).
        std::vector<std::array<int, 3>> occupiedCoordsOf(const VoxelValueMap &values) {
            std::vector<std::array<int, 3>> coords;
            coords.reserve(values.size());
            for (const auto &kv : values) coords.push_back(kv.first);
            return coords;
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

    // Task 4: multi-resolution CPU MC over the FULL mixed grid (fine level==0 at h, coarse
    // level==1 at 2h), with transitional-boundary handling per the design doc (Sec.5) / plan
    // (Task 4 Step 3):
    //
    //  1. Any coarse block face-adjacent to an ORIGINALLY-fine block is uniformly refined into
    //     8 fine sub-cells that all sample the coarse block's own averaged value (spec Sec.8
    //     fallback). This refinement is then propagated transitively (a coarse block adjacent
    //     to an already-refined block is refined too, via BFS/fixed-point) until no coarse
    //     block borders a fine/refined one. After this closure, the surviving ("deep-interior")
    //     coarse blocks are guaranteed to be surrounded only by other surviving coarse blocks
    //     (or the edge of the observed domain) -- see the proof in the Task 4 report: this is
    //     what makes the fine and coarse MC passes below operate on disjoint, non-adjacent
    //     domains, which is the mechanism that avoids BOTH overlapping (duplicate) geometry and
    //     cracks/gaps at the shared lattice, rather than a per-face partial clip of a single
    //     boundary layer (found analytically to still leave sub-voxel gaps one layer further
    //     in -- see the Task 4 report for the derivation).
    //  2. Fine-pass cube corners and coarse-pass cube corners both resolve through the same
    //     finer-favoring CornerSampler: a fine-lattice corner prefers an actual fine value and
    //     falls back to the owning coarse block's value; a coarse-lattice corner converts to its
    //     equivalent fine coordinate and reuses that same finer-favoring lookup (so a coarse
    //     cube touching an actual fine voxel picks up the finer sample instead of a coarse
    //     average) -- this is the "finer-favoring" rule from the design doc.
    //  3. Fine-pass and coarse-pass raw triangles are combined and welded ONCE at 0.25*h
    //     (plan's vertex-collapse epsilon), with area-weighted normals recomputed from the
    //     welded positions; triangles that become degenerate after welding are dropped.
    AdaptiveMesh AdaptiveVoxelGrid::ExtractMesh() {
        if (m_mixedDirty) buildMixed();

        VoxelValueMap fineMap, coarseMap;
        fineMap.reserve(m_mixed.size());
        for (const auto &mv : m_mixed) {
            if (mv.level == 0)
                fineMap[latticeCoordOf(mv.center, m_h)] = mv.tsdf;
            else
                coarseMap[latticeCoordOf(mv.center, 2.0f * m_h)] = mv.tsdf;
        }

        // Coarse-index blocks that buildMixed() kept fine (the refinement seed set).
        std::unordered_set<std::array<int, 3>, IVec3Hash> fineBlocks;
        fineBlocks.reserve(fineMap.size());
        for (const auto &kv : fineMap) fineBlocks.insert(floorDiv2(kv.first));

        static constexpr int kFaceNb[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        auto neighborsOf = [](const std::array<int, 3> &ck, int i) {
            return std::array<int, 3>{ck[0] + kFaceNb[i][0], ck[1] + kFaceNb[i][1], ck[2] + kFaceNb[i][2]};
        };

        // BFS/fixed-point boundary refinement: seed with coarse blocks face-adjacent to an
        // originally-fine block, then flood outward through coarse-coarse adjacency.
        std::unordered_set<std::array<int, 3>, IVec3Hash> refined;
        std::vector<std::array<int, 3>> queue;
        for (const auto &kv : coarseMap) {
            const auto &ck = kv.first;
            for (int i = 0; i < 6; ++i)
                if (fineBlocks.count(neighborsOf(ck, i))) {
                    if (refined.insert(ck).second) queue.push_back(ck);
                    break;
                }
        }
        for (size_t qi = 0; qi < queue.size(); ++qi) {
            const auto ck = queue[qi];
            for (int i = 0; i < 6; ++i) {
                const auto nb = neighborsOf(ck, i);
                if (coarseMap.count(nb) && refined.insert(nb).second) queue.push_back(nb);
            }
        }

        VoxelValueMap workFine = fineMap;
        VoxelValueMap workCoarse = coarseMap;
        for (const auto &ck : refined) {
            const float v = coarseMap.at(ck);
            for (int ox = 0; ox <= 1; ++ox)
                for (int oy = 0; oy <= 1; ++oy)
                    for (int oz = 0; oz <= 1; ++oz)
                        workFine[{ck[0] * 2 + ox, ck[1] * 2 + oy, ck[2] * 2 + oz}] = v;
            workCoarse.erase(ck);
        }

        const CornerSampler sampler{&workFine, &workCoarse};

        std::vector<Extraction::core::RawTriangle> tris;
        Extraction::core::GenerateRawTriangles(
                Extraction::core::CandidateBases(occupiedCoordsOf(workFine)),
                [&](const std::array<int, 3> &v, float &out) { return sampler.sampleAtFine(v, out); }, m_h, tris);
        Extraction::core::GenerateRawTriangles(
                Extraction::core::CandidateBases(occupiedCoordsOf(workCoarse)),
                [&](const std::array<int, 3> &ck, float &out) { return sampler.sampleAtCoarse(ck, out); },
                2.0f * m_h, tris);

        return Extraction::core::WeldAndComputeNormals(tris, 0.25f * m_h);
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

        // Floor division (a / b rounded toward -infinity, see file-local floorDiv above),
        // correct for negative coords -- plain C++ integer division truncates toward zero,
        // which would bucket e.g. -1 and 0 into different-signed halves of the same coarse
        // block.
        struct Blk {
            std::vector<size_t> idx;
        };
        std::map<std::array<int, 3>, Blk> blocks;
        for (size_t i = 0; i < vox.size(); ++i) {
            const Eigen::Vector3i vc = vcoord(vox[i].center);
            blocks[{floorDiv(vc.x(), 2), floorDiv(vc.y(), 2), floorDiv(vc.z(), 2)}].idx.push_back(i);
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
