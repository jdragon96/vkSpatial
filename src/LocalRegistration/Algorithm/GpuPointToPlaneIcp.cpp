#include "LocalRegistration/Algorithm/GpuPointToPlaneIcp.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace Pipeline {

    LocalGrid::LocalGrid(const std::vector<Eigen::Vector3f> &points, float cell) : m_pts(points) {
        m_cell = std::max(cell, 1e-8f);
        if (points.empty()) {
            m_bucketStart.assign(2, 0);
            return;
        }

        Eigen::Vector3f minBound = points[0], maxBound = points[0];
        for (const auto &point: points) {
            minBound = minBound.cwiseMin(point);
            maxBound = maxBound.cwiseMax(point);
        }
        m_origin = minBound;

        const auto updateDimsFromCellSize = [&]() {
            for (int axis = 0; axis < 3; ++axis)
                m_dims[axis] = std::max(1, int(std::floor((maxBound[axis] - minBound[axis]) / m_cell)) + 1);
        };
        updateDimsFromCellSize();

        constexpr uint64_t kMaxCellCount = 32ull * 1024 * 1024;
        constexpr float kOneStepConvergenceMargin = 1.02f;
        for (int attempt = 0; attempt < 64; ++attempt) {
            const uint64_t currentCellCount = uint64_t(m_dims.x()) * uint64_t(m_dims.y()) * uint64_t(m_dims.z());
            if (currentCellCount <= kMaxCellCount) break;
            const double overflowRatio = double(currentCellCount) / double(kMaxCellCount);
            m_cell *= float(std::cbrt(overflowRatio)) * kOneStepConvergenceMargin;
            updateDimsFromCellSize();
        }

        const int cellCount = m_dims.x() * m_dims.y() * m_dims.z();
        m_bucketStart.assign(cellCount + 1, 0);
        for (const auto &point: points) ++m_bucketStart[cellIndex(cellOf(point)) + 1];
        for (int i = 0; i < cellCount; ++i) m_bucketStart[i + 1] += m_bucketStart[i];

        m_bucketIdx.resize(points.size());
        std::vector<uint32_t> nextFreeSlot(m_bucketStart.begin(), m_bucketStart.end() - 1);
        for (int i = 0; i < int(points.size()); ++i)
            m_bucketIdx[nextFreeSlot[cellIndex(cellOf(points[i]))]++] = uint32_t(i);
    }

    int LocalGrid::Nearest(const Eigen::Vector3f &query, float radius) const {
        if (m_pts.empty()) return -1;
        const Eigen::Vector3i centerCell = cellOf(query);
        int bestPointIndex = -1;
        float bestSquaredDistance = radius * radius;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const Eigen::Vector3i neighborCell(centerCell.x() + dx, centerCell.y() + dy, centerCell.z() + dz);
                    if ((neighborCell.array() < 0).any() || (neighborCell.array() >= m_dims.array()).any()) continue;
                    const int neighborCellIndex = cellIndex(neighborCell);
                    for (uint32_t slot = m_bucketStart[neighborCellIndex]; slot < m_bucketStart[neighborCellIndex + 1]; ++slot) {
                        const int pointIndex = int(m_bucketIdx[slot]);
                        const float squaredDistance = (query - m_pts[pointIndex]).squaredNorm();
                        if (squaredDistance < bestSquaredDistance) {
                            bestSquaredDistance = squaredDistance;
                            bestPointIndex = pointIndex;
                        }
                    }
                }
        return bestPointIndex;
    }

    namespace {
        struct IcpPC {
            float T[16]; // column-major mat4
            float originX, originY, originZ, cell;
            int32_t dimsX, dimsY, dimsZ;
            float maxCorr;
            float huberScale, normalCompatibilityCosine; // Tier 2: robust weighting + normal rejection
            uint32_t numSrc, numCells;
        };
        void writeVec3Buf(Engine::Core::Buffer &buf, const std::vector<Eigen::Vector3f> &v) {
            std::vector<float> pad(v.size() * 4u, 0.0f); // vec4 std430 stride
            for (size_t i = 0; i < v.size(); ++i) {
                pad[i * 4] = v[i].x();
                pad[i * 4 + 1] = v[i].y();
                pad[i * 4 + 2] = v[i].z();
            }
            buf.Allocate(uint32_t(pad.size() * sizeof(float)));
            buf.Upload(pad.data(), uint32_t(pad.size() * sizeof(float)));
        }
    } // namespace

    GpuPointToPlaneIcp::GpuPointToPlaneIcp(Engine::Core::Context &ctx) : m_ctx(&ctx) {
        m_src = std::make_unique<Engine::Core::Buffer>(ctx);
        m_tgtPts = std::make_unique<Engine::Core::Buffer>(ctx);
        m_tgtNrm = std::make_unique<Engine::Core::Buffer>(ctx);
        m_bucketStart = std::make_unique<Engine::Core::Buffer>(ctx);
        m_bucketIdx = std::make_unique<Engine::Core::Buffer>(ctx);
        m_partials = std::make_unique<Engine::Core::Buffer>(ctx);
        m_sourceNormals = std::make_unique<Engine::Core::Buffer>(ctx);
        m_kernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_kernel->Build("LocalRegistration/Algorithm/GpuPointToPlaneIcp.Iterate.glsl");
    }

    GpuPointToPlaneIcp::IterOut GpuPointToPlaneIcp::Accumulate(
            const std::vector<Eigen::Vector3f> &src, const Registration::PointCloud &tgt,
            const Eigen::Matrix4f &T, float maxCorrDist) {
        if (tgt.points.empty()) return AccumulateCentred(src, tgt, Eigen::Vector3f::Zero(), T, maxCorrDist);
        Eigen::Vector3f c = Eigen::Vector3f::Zero();
        for (const auto &q: tgt.points) c += q;
        c /= float(tgt.points.size());
        return AccumulateCentred(src, tgt, c, T, maxCorrDist);
    }

    GpuPointToPlaneIcp::IterOut GpuPointToPlaneIcp::AccumulateCentred(
            const std::vector<Eigen::Vector3f> &src, const Registration::PointCloud &tgt,
            const Eigen::Vector3f &c, const Eigen::Matrix4f &T, float maxCorrDist) {
        IterOut out;
        out.H.setZero();
        out.b.setZero();
        out.inliers = 0;
        if (!prepareCentred(src, {}, tgt, c, maxCorrDist)) return out;
        return dispatchCentred(T, kNoRobustWeightingHuberScale, kNoNormalRejectionCosine, maxCorrDist);
    }

    bool GpuPointToPlaneIcp::prepareCentred(const std::vector<Eigen::Vector3f> &src,
                                            const std::vector<Eigen::Vector3f> &sourceNormals,
                                            const Registration::PointCloud &tgt,
                                            const Eigen::Vector3f &surfaceCenter,
                                            float maxCorrDist) {
        if (src.empty() || tgt.points.size() < 3 || tgt.normals.size() != tgt.points.size()) return false;
        if (!sourceNormals.empty() && sourceNormals.size() != src.size()) return false;

        std::vector<Eigen::Vector3f> sc(src.size()), tc(tgt.points.size());
        for (size_t i = 0; i < src.size(); ++i) sc[i] = src[i] - surfaceCenter;
        for (size_t i = 0; i < tc.size(); ++i) tc[i] = tgt.points[i] - surfaceCenter;

        LocalGrid grid(tc, maxCorrDist);

        // Upload the target PERMUTED into bucket order, with an identity bucket index.
        //
        // The kernel's inner loop reads g_tgtPts[g_bidx[k]]: a second load that depends on the
        // first, landing anywhere in a multi-megabyte array. That is the whole cost. Measured on a
        // live D435 capture, one iteration over 6,618 source points against a 245k-point target
        // takes 3.1 ms, and eight iterations recorded into one command buffer take 8x that -- so
        // it is the kernel, not submission overhead (0.24 ms per dispatch), and it barely scales
        // with the source count because 26 workgroups cannot hide the latency of ~66 dependent
        // scattered loads per thread.
        //
        // Permuting here makes both loads sequential within a cell. The candidate sequence per
        // cell is unchanged -- same points, same order -- so `best` resolves to the same target
        // and the solve is bit-identical; only the addresses move.
        std::vector<Eigen::Vector3f> bucketOrderedPoints(tc.size()), bucketOrderedNormals(tc.size());
        std::vector<uint32_t> identityBucketIndex(grid.m_bucketIdx.size());
        for (size_t slot = 0; slot < grid.m_bucketIdx.size(); ++slot) {
            const uint32_t source = grid.m_bucketIdx[slot];
            bucketOrderedPoints[slot] = tc[source];
            bucketOrderedNormals[slot] = tgt.normals[source];
            identityBucketIndex[slot] = uint32_t(slot);
        }

        writeVec3Buf(*m_src, sc);
        writeVec3Buf(*m_tgtPts, bucketOrderedPoints);
        writeVec3Buf(*m_tgtNrm, bucketOrderedNormals);
        writeVec3Buf(*m_sourceNormals,
                     sourceNormals.empty() ? std::vector<Eigen::Vector3f>(src.size(), Eigen::Vector3f::Zero())
                                           : sourceNormals);
        m_bucketStart->Allocate(uint32_t(grid.m_bucketStart.size() * sizeof(uint32_t)));
        m_bucketStart->Upload(grid.m_bucketStart.data(), uint32_t(grid.m_bucketStart.size() * sizeof(uint32_t)));
        m_bucketIdx->Allocate(uint32_t(std::max<size_t>(1, identityBucketIndex.size()) * sizeof(uint32_t)));
        if (!identityBucketIndex.empty())
            m_bucketIdx->Upload(
                    identityBucketIndex.data(),
                    uint32_t(identityBucketIndex.size() * sizeof(uint32_t)));

        m_pNumSrc = uint32_t(src.size());
        m_pNumWorkerGroup = (m_pNumSrc + kLocal - 1) / kLocal;
        m_partials->AllocateHostVisibleReadback(m_pNumWorkerGroup * 29u * sizeof(int32_t));

        m_pOrigin = grid.m_origin;
        m_pDims = grid.m_dims;
        m_pCell = grid.m_cell;
        m_pMaxCorr = maxCorrDist;

        m_kernel->Bind(0, *m_src)
                .Bind(1, *m_tgtPts)
                .Bind(2, *m_tgtNrm)
                .Bind(3, *m_bucketStart)
                .Bind(4, *m_bucketIdx)
                .Bind(5, *m_partials)
                .Bind(6, *m_sourceNormals);
        return true;
    }

    GpuPointToPlaneIcp::IterOut GpuPointToPlaneIcp::dispatchCentred(const Eigen::Matrix4f &T,
                                                                    float huberScale,
                                                                    float normalCompatibilityCosine,
                                                                    float currentMaxCorrespondenceDistance) {
        IterOut out;
        out.H.setZero();
        out.b.setZero();
        out.inliers = 0;

        IcpPC pc{};
        for (int i = 0; i < 16; ++i) pc.T[i] = T.data()[i]; // Eigen is column-major -> matches std430 mat4
        pc.originX = m_pOrigin.x();
        pc.originY = m_pOrigin.y();
        pc.originZ = m_pOrigin.z();
        pc.cell = m_pCell;
        pc.dimsX = m_pDims.x();
        pc.dimsY = m_pDims.y();
        pc.dimsZ = m_pDims.z();
        pc.maxCorr = currentMaxCorrespondenceDistance; // per-iteration distance FILTER (<= pc.cell)
        pc.huberScale = huberScale;
        pc.normalCompatibilityCosine = normalCompatibilityCosine;
        pc.numSrc = m_pNumSrc;
        pc.numCells = uint32_t(m_pDims.x() * m_pDims.y() * m_pDims.z());

        m_kernel->Args(pc);
        m_kernel->DispatchElements(m_pNumSrc);

        m_partials->MakeVisibleToCPU(m_pNumWorkerGroup * 29u * sizeof(int32_t));
        const int32_t *part = static_cast<const int32_t *>(m_partials->MappedPtr());
        // Slots 0..27 are fixed-point ints; slot 28 carries the workgroup's squared-residual sum as
        // FLOAT BITS (the fixed-point path zeroed every |e| under ~7 mm, flooring the rmse).
        double acc[28] = {0};
        double sumOfSquaredResiduals = 0.0;
        for (uint32_t w = 0; w < m_pNumWorkerGroup; ++w) {
            for (int k = 0; k < 28; ++k) acc[k] += part[w * 29u + k];
            float workgroupSquaredResidualSum;
            std::memcpy(&workgroupSquaredResidualSum, &part[w * 29u + 28u], sizeof(workgroupSquaredResidualSum));
            sumOfSquaredResiduals += double(workgroupSquaredResidualSum);
        }
        int k = 0;
        for (int r = 0; r < 6; ++r)
            for (int col = r; col < 6; ++col) {
                const double v = acc[k++] / double(kScale);
                out.H(r, col) = v;
                out.H(col, r) = v;
            }
        for (int r = 0; r < 6; ++r) out.b(r) = acc[21 + r] / double(kScale);
        out.inliers = int(std::llround(acc[27])); // inlier count stored x1 (SCALE not applied to it)
        out.sumOfSquaredResiduals = sumOfSquaredResiduals;
        return out;
    }

    Registration::RegistrationResult GpuPointToPlaneIcp::Solve(
            const std::vector<Eigen::Vector3f> &src,
            const std::vector<Eigen::Vector3f> &sourceNormals,
            const Registration::PointCloud &tgt,
            const Eigen::Matrix4f &priorT,
            const Registration::RegistrationParam &params) {
        Registration::RegistrationResult res;
        res.T = priorT;
        if (src.empty() || tgt.points.size() < 3 || tgt.normals.size() != tgt.points.size()) return res;
        if (!sourceNormals.empty() && sourceNormals.size() != src.size()) return res;

        Eigen::Vector3d centroidSum = Eigen::Vector3d::Zero();
        for (const auto &q: tgt.points) {
            centroidSum += q.cast<double>();
        }
        const Eigen::Vector3f surfaceCenter = (centroidSum / double(tgt.points.size())).cast<float>();

        Eigen::Matrix4f Tc = Eigen::Matrix4f::Identity();
        Tc.block<3, 1>(0, 3) = -surfaceCenter;
        Eigen::Matrix4f TcInv = Eigen::Matrix4f::Identity();
        TcInv.block<3, 1>(0, 3) = surfaceCenter;
        Eigen::Matrix4f T = (Tc * (priorT * TcInv));

        if (!prepareCentred(src, sourceNormals, tgt, surfaceCenter, params.maxCorrDist)) {
            return res;
        }

        const float normalCompatibilityCosine =
                sourceNormals.empty() ? kNoNormalRejectionCosine : params.normalCompatibilityCosine;
        double lastSumOfSquaredResiduals = 0.0;

        for (int iter = 0; iter < params.maxIters; ++iter) {
            const Registration::AnnealedIcpIterationParams annealed = Registration::AnnealIcpIteration(params, iter);
            const IterOut a = dispatchCentred(T,
                                              annealed.huberScale,
                                              normalCompatibilityCosine,
                                              annealed.maxCorrespondenceDistance);
            if (a.inliers < params.minInliers) break;
            const Eigen::Matrix<double, 6, 1> x = a.H.ldlt().solve(a.b);
            const Eigen::Matrix3d Rd = (Eigen::AngleAxisd(x[2], Eigen::Vector3d::UnitZ()) *
                                        Eigen::AngleAxisd(x[1], Eigen::Vector3d::UnitY()) *
                                        Eigen::AngleAxisd(x[0], Eigen::Vector3d::UnitX()))
                                               .toRotationMatrix();
            Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
            delta.block<3, 3>(0, 0) = Rd.cast<float>();
            delta.block<3, 1>(0, 3) = x.tail<3>().cast<float>();
            T = delta * T;
            res.numInliers = size_t(a.inliers);
            res.fitness = float(a.inliers) / float(src.size());
            lastSumOfSquaredResiduals = a.sumOfSquaredResiduals;
            if (x.norm() < params.convEps) break;
        }

        res.T = (TcInv * (T * Tc));
        if (res.numInliers > 0) {
            res.rmse = float(std::sqrt(lastSumOfSquaredResiduals / double(res.numInliers)));
        }
        res.valid = res.numInliers >= size_t(params.minInliers) && res.fitness >= params.minFitness;
        return res;
    }

} // namespace Pipeline
