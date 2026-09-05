#include "Features/Fpfh.h"

#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace Features {

    namespace {

        constexpr int kBins = 11;
        constexpr float kPi = 3.14159265358979323846f;

        // Hash-grid cell key, mirroring the Downsample.cpp VoxelKey style (own
        // translation-unit-local copy — this module is CPU/Eigen-only and does not
        // share state with Downsample.cpp or Engine::Spatial).
        struct GridKey {
            int32_t x = 0, y = 0, z = 0;

            bool operator==(const GridKey &o) const {
                return x == o.x && y == o.y && z == o.z;
            }
        };

        struct GridKeyHash {
            size_t operator()(const GridKey &k) const {
                uint64_t h = uint64_t(uint32_t(k.x)) * 73856093ull;
                h ^= uint64_t(uint32_t(k.y)) * 19349663ull;
                h ^= uint64_t(uint32_t(k.z)) * 83492791ull;
                h ^= h >> 33;
                return size_t(h);
            }
        };

        using Grid = std::unordered_map<GridKey, std::vector<int>, GridKeyHash>;

        inline GridKey KeyOf(const Eigen::Vector3f &p, float cellSize) {
            return GridKey{
                    int32_t(std::floor(p.x() / cellSize)),
                    int32_t(std::floor(p.y() / cellSize)),
                    int32_t(std::floor(p.z() / cellSize))};
        }

        Grid BuildGrid(const PointCloud &cloud, float cellSize) {
            Grid grid;
            grid.reserve(cloud.points.size());
            for (size_t i = 0; i < cloud.points.size(); ++i) {
                grid[KeyOf(cloud.points[i], cellSize)].push_back(int(i));
            }
            return grid;
        }

        // Scans the 3x3x3 block of cells (cell size == cellSize, the same size the grid
        // was built with) around `center` and keeps candidates within `radius`. Exact ONLY
        // when radius <= cellSize: a query at a cell edge with a neighbour `radius` away can
        // land two cells out (unscanned) once radius > cellSize. Holds here because pass 2
        // uses fpfhRadius == cellSize and pass 1 uses normalRadius < fpfhRadius; a caller
        // passing radius > cellSize would silently get truncated neighbourhoods.
        void QueryRadius(const PointCloud &cloud, const Grid &grid, float cellSize,
                          const Eigen::Vector3f &center, float radius, std::vector<int> &out) {
            out.clear();
            const float r2 = radius * radius;
            const GridKey c = KeyOf(center, cellSize);
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        const GridKey k{c.x + dx, c.y + dy, c.z + dz};
                        const auto it = grid.find(k);
                        if (it == grid.end()) continue;
                        for (int idx: it->second) {
                            const float d2 = (cloud.points[idx] - center).squaredNorm();
                            if (d2 <= r2) out.push_back(idx);
                        }
                    }
        }

        inline int BinOf(float val, float lo, float hi, int bins) {
            const float t = (val - lo) / (hi - lo);
            int b = int(std::floor(t * float(bins)));
            if (b < 0) b = 0;
            if (b >= bins) b = bins - 1;
            return b;
        }

        // SPFH(p): Darboux-frame pair features between p and each neighbour q (q != p),
        // binned into 3 histograms of kBins each, individually normalized to sum 1, and
        // concatenated into the 33-D descriptor (f1 | f2 | f3).
        Fpfh33 ComputeSpfh(const PointCloud &cloud, int p, const std::vector<int> &neighbors) {
            Fpfh33 h = Fpfh33::Zero();

            Eigen::Matrix<float, kBins, 1> hf1 = Eigen::Matrix<float, kBins, 1>::Zero();
            Eigen::Matrix<float, kBins, 1> hf2 = Eigen::Matrix<float, kBins, 1>::Zero();
            Eigen::Matrix<float, kBins, 1> hf3 = Eigen::Matrix<float, kBins, 1>::Zero();

            const Eigen::Vector3f &pp = cloud.points[p];
            const Eigen::Vector3f &np = cloud.normals[p];

            int count = 0;
            for (int qi: neighbors) {
                if (qi == p) continue;

                const Eigen::Vector3f d = cloud.points[qi] - pp;
                const float dist = d.norm();
                if (dist <= 1e-12f) continue;

                const Eigen::Vector3f dhat = d / dist;
                const Eigen::Vector3f u = np;
                const Eigen::Vector3f vRaw = dhat.cross(u);
                const float vNorm = vRaw.norm();
                if (vNorm <= 1e-12f) continue; // d parallel to n_p: Darboux frame undefined, skip pair

                const Eigen::Vector3f v = vRaw / vNorm;
                const Eigen::Vector3f w = u.cross(v);
                const Eigen::Vector3f &nq = cloud.normals[qi];

                const float f1 = v.dot(nq);
                const float f2 = u.dot(dhat);
                const float f3 = std::atan2(w.dot(nq), u.dot(nq));

                hf1(BinOf(f1, -1.0f, 1.0f, kBins)) += 1.0f;
                hf2(BinOf(f2, -1.0f, 1.0f, kBins)) += 1.0f;
                hf3(BinOf(f3, -kPi, kPi, kBins)) += 1.0f;
                ++count;
            }

            if (count > 0) {
                hf1 /= float(count);
                hf2 /= float(count);
                hf3 /= float(count);
            }

            h.segment<kBins>(0) = hf1;
            h.segment<kBins>(kBins) = hf2;
            h.segment<kBins>(2 * kBins) = hf3;
            return h;
        }

        // Renormalizes each of the 3 kBins-wide sub-histograms of `h` to sum to 1
        // independently (mirrors the PCL/KISS-Matcher convention of normalizing the f1,
        // f2, f3 blocks separately rather than the whole 33-vector at once).
        void RenormalizeBlocks(Fpfh33 &h) {
            for (int block = 0; block < 3; ++block) {
                const float sum = h.segment<kBins>(block * kBins).sum();
                if (sum > 1e-12f) h.segment<kBins>(block * kBins) /= sum;
            }
        }

    } // namespace

    std::vector<Fpfh33> ComputeFpfh(const PointCloud &cloud, float normalRadius, float fpfhRadius) {
        const size_t n = cloud.points.size();
        std::vector<Fpfh33> result;
        if (n == 0 || cloud.normals.size() != n || normalRadius <= 0.0f || fpfhRadius <= 0.0f) return result;

        const Grid grid = BuildGrid(cloud, fpfhRadius);

        // Pass 1: SPFH per point (each point's own Darboux-frame histogram).
        std::vector<Fpfh33> spfh(n);
        std::vector<int> neighborBuf;
        for (size_t i = 0; i < n; ++i) {
            QueryRadius(cloud, grid, fpfhRadius, cloud.points[i], normalRadius, neighborBuf);
            spfh[i] = ComputeSpfh(cloud, int(i), neighborBuf);
        }

        // Pass 2: FPFH = SPFH(p) + (1/k) * sum_{q in N(p,fpfhRadius)} (1/|q-p|) * SPFH(q).
        result.resize(n);
        for (size_t i = 0; i < n; ++i) {
            QueryRadius(cloud, grid, fpfhRadius, cloud.points[i], fpfhRadius, neighborBuf);

            Fpfh33 weighted = Fpfh33::Zero();
            int k = 0;
            for (int qi: neighborBuf) {
                if (qi == int(i)) continue;
                const float dist = (cloud.points[qi] - cloud.points[i]).norm();
                if (dist <= 1e-12f) continue;
                weighted += (1.0f / dist) * spfh[qi];
                ++k;
            }

            Fpfh33 out = spfh[i];
            if (k > 0) out += weighted / float(k);
            RenormalizeBlocks(out);
            result[i] = out;
        }

        return result;
    }

} // namespace Features
