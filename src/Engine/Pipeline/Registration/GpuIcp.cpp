#include "Engine/Pipeline/Registration/GpuIcp.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace Engine::Pipeline {

    LocalGrid::LocalGrid(const std::vector<Eigen::Vector3f> &pts, float cell) : m_pts(pts) {
        m_cell = cell > 1e-8f ? cell : 1e-8f;
        if (pts.empty()) { m_bucketStart.assign(2, 0); return; }
        Eigen::Vector3f mn = pts[0], mx = pts[0];
        for (const auto &p : pts) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
        m_origin = mn;
        for (int a = 0; a < 3; ++a)
            m_dims[a] = std::max(1, int(std::floor((mx[a] - mn[a]) / m_cell)) + 1);
        const int nCells = m_dims.x() * m_dims.y() * m_dims.z();

        // Counting sort of point indices by cell -> CSR (bucketStart prefix sum, bucketIdx grouped).
        m_bucketStart.assign(nCells + 1, 0);
        for (const auto &p : pts) ++m_bucketStart[cellIndex(cellOf(p)) + 1];
        for (int i = 0; i < nCells; ++i) m_bucketStart[i + 1] += m_bucketStart[i];
        m_bucketIdx.resize(pts.size());
        std::vector<uint32_t> cursor(m_bucketStart.begin(), m_bucketStart.end() - 1);
        for (int i = 0; i < (int)pts.size(); ++i)
            m_bucketIdx[cursor[cellIndex(cellOf(pts[i]))]++] = uint32_t(i);
    }

    int LocalGrid::Nearest(const Eigen::Vector3f &q, float radius) const {
        if (m_pts.empty()) return -1;
        const Eigen::Vector3i c = cellOf(q);
        const float r2 = radius * radius;
        int best = -1; float bestD2 = r2;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const Eigen::Vector3i cc(c.x() + dx, c.y() + dy, c.z() + dz);
                    if ((cc.array() < 0).any() || (cc.array() >= m_dims.array()).any()) continue;
                    const int ci = cellIndex(cc);
                    for (uint32_t k = m_bucketStart[ci]; k < m_bucketStart[ci + 1]; ++k) {
                        const int idx = int(m_bucketIdx[k]);
                        const float d2 = (q - m_pts[idx]).squaredNorm();
                        if (d2 < bestD2) { bestD2 = d2; best = idx; }
                    }
                }
        return best;
    }

    namespace {
        struct IcpPC {
            float T[16];               // column-major mat4
            float originX, originY, originZ, cell;
            int32_t dimsX, dimsY, dimsZ; float maxCorr;
            uint32_t numSrc, numCells;
        };
        void writeVec3Buf(Engine::Core::Buffer &buf, const std::vector<Eigen::Vector3f> &v) {
            std::vector<float> pad(v.size() * 4u, 0.0f);           // vec4 std430 stride
            for (size_t i = 0; i < v.size(); ++i) { pad[i*4]=v[i].x(); pad[i*4+1]=v[i].y(); pad[i*4+2]=v[i].z(); }
            buf.Allocate(uint32_t(pad.size() * sizeof(float)));
            buf.Upload(pad.data(), uint32_t(pad.size() * sizeof(float)));
        }
    }

    GpuPointToPlaneIcp::GpuPointToPlaneIcp(Engine::Core::Context &ctx) : m_ctx(&ctx) {
        m_src = std::make_unique<Engine::Core::Buffer>(ctx);
        m_tgtPts = std::make_unique<Engine::Core::Buffer>(ctx);
        m_tgtNrm = std::make_unique<Engine::Core::Buffer>(ctx);
        m_bucketStart = std::make_unique<Engine::Core::Buffer>(ctx);
        m_bucketIdx = std::make_unique<Engine::Core::Buffer>(ctx);
        m_partials = std::make_unique<Engine::Core::Buffer>(ctx);
        m_kernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_kernel->Build("icp_iterate.comp.glsl");
    }

    GpuPointToPlaneIcp::IterOut GpuPointToPlaneIcp::Accumulate(
            const std::vector<Eigen::Vector3f> &src, const Engine::Registration::PointCloud &tgt,
            const Eigen::Matrix4f &T, float maxCorrDist) {
        if (tgt.points.empty()) return AccumulateCentred(src, tgt, Eigen::Vector3f::Zero(), T, maxCorrDist);
        Eigen::Vector3f c = Eigen::Vector3f::Zero();
        for (const auto &q : tgt.points) c += q; c /= float(tgt.points.size());
        return AccumulateCentred(src, tgt, c, T, maxCorrDist);
    }

    GpuPointToPlaneIcp::IterOut GpuPointToPlaneIcp::AccumulateCentred(
            const std::vector<Eigen::Vector3f> &src, const Engine::Registration::PointCloud &tgt,
            const Eigen::Vector3f &c, const Eigen::Matrix4f &T, float maxCorrDist) {
        IterOut out; out.H.setZero(); out.b.setZero(); out.inliers = 0;
        if (src.empty() || tgt.points.size() < 3 || tgt.normals.size() != tgt.points.size()) return out;

        // Centre on the (caller-supplied) target centroid `c` -- numerical conditioning + fixed-point
        // safety. `T` is used AS-IS: it is the caller's responsibility to already be expressed in this
        // centred frame (see icp_iterate.comp.glsl header + this class's doc comment: the returned H,b
        // are the CENTRED-frame normal equations, not the un-centred/world ones).
        std::vector<Eigen::Vector3f> sc(src.size()), tc(tgt.points.size());
        for (size_t i = 0; i < src.size(); ++i) sc[i] = src[i] - c;
        for (size_t i = 0; i < tc.size(); ++i) tc[i] = tgt.points[i] - c;

        LocalGrid grid(tc, maxCorrDist);
        writeVec3Buf(*m_src, sc);
        writeVec3Buf(*m_tgtPts, tc);
        writeVec3Buf(*m_tgtNrm, tgt.normals);
        m_bucketStart->Allocate(uint32_t(grid.m_bucketStart.size() * sizeof(uint32_t)));
        m_bucketStart->Upload(grid.m_bucketStart.data(), uint32_t(grid.m_bucketStart.size() * sizeof(uint32_t)));
        m_bucketIdx->Allocate(uint32_t(std::max<size_t>(1, grid.m_bucketIdx.size()) * sizeof(uint32_t)));
        if (!grid.m_bucketIdx.empty())
            m_bucketIdx->Upload(grid.m_bucketIdx.data(), uint32_t(grid.m_bucketIdx.size() * sizeof(uint32_t)));

        const uint32_t numWG = (uint32_t(src.size()) + kLocal - 1) / kLocal;
        m_partials->AllocateHostVisibleReadback(numWG * 28u * sizeof(int32_t));
        std::memset(m_partials->MappedPtr(), 0, numWG * 28u * sizeof(int32_t));
        m_partials->FlushMapped(numWG * 28u * sizeof(int32_t));

        IcpPC pc{};
        for (int i = 0; i < 16; ++i) pc.T[i] = T.data()[i]; // Eigen is column-major -> matches std430 mat4
        pc.originX = grid.m_origin.x(); pc.originY = grid.m_origin.y(); pc.originZ = grid.m_origin.z();
        pc.cell = grid.m_cell; pc.dimsX = grid.m_dims.x(); pc.dimsY = grid.m_dims.y(); pc.dimsZ = grid.m_dims.z();
        pc.maxCorr = maxCorrDist; pc.numSrc = uint32_t(src.size());
        pc.numCells = uint32_t(grid.m_dims.x() * grid.m_dims.y() * grid.m_dims.z());

        m_kernel->Bind(0, *m_src).Bind(1, *m_tgtPts).Bind(2, *m_tgtNrm)
                 .Bind(3, *m_bucketStart).Bind(4, *m_bucketIdx).Bind(5, *m_partials);
        m_kernel->Args(pc);
        m_kernel->DispatchElements(uint32_t(src.size())); // synchronous

        m_partials->InvalidateMapped(numWG * 28u * sizeof(int32_t));
        const int32_t *part = static_cast<const int32_t *>(m_partials->MappedPtr());
        double acc[28] = {0};
        for (uint32_t w = 0; w < numWG; ++w) for (int k = 0; k < 28; ++k) acc[k] += part[w * 28u + k];
        int k = 0;
        for (int r = 0; r < 6; ++r) for (int col = r; col < 6; ++col) {
            const double v = acc[k++] / double(kScale);
            out.H(r, col) = v; out.H(col, r) = v;
        }
        for (int r = 0; r < 6; ++r) out.b(r) = acc[21 + r] / double(kScale);
        out.inliers = int(std::llround(acc[27])); // inlier count stored x1 (SCALE not applied to it)
        return out;
    }

    Engine::Registration::RegistrationResult GpuPointToPlaneIcp::Solve(
            const std::vector<Eigen::Vector3f> &src, const Engine::Registration::PointCloud &tgt,
            const Eigen::Matrix4f &priorT, const Engine::Registration::RegistrationParam &params) {
        Engine::Registration::RegistrationResult res; res.T = priorT;
        if (src.empty() || tgt.points.size() < 3 || tgt.normals.size() != tgt.points.size()) return res;

        Eigen::Vector3f c = Eigen::Vector3f::Zero();
        for (const auto &q : tgt.points) c += q; c /= float(tgt.points.size());
        Eigen::Matrix4f Tc = Eigen::Matrix4f::Identity(); Tc.block<3,1>(0,3) = -c;  // shift world->centred
        Eigen::Matrix4f TcInv = Eigen::Matrix4f::Identity(); TcInv.block<3,1>(0,3) = c;
        Eigen::Matrix4f T = Tc * priorT * TcInv; // work in the centred frame

        for (int iter = 0; iter < params.maxIters; ++iter) {
            const IterOut a = AccumulateCentred(src, tgt, c, T, params.maxCorrDist);
            if (a.inliers < params.minInliers) break;
            const Eigen::Matrix<double,6,1> x = a.H.ldlt().solve(a.b);
            const Eigen::Matrix3d Rd = (Eigen::AngleAxisd(x[2], Eigen::Vector3d::UnitZ()) *
                                        Eigen::AngleAxisd(x[1], Eigen::Vector3d::UnitY()) *
                                        Eigen::AngleAxisd(x[0], Eigen::Vector3d::UnitX())).toRotationMatrix();
            Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
            delta.block<3,3>(0,0) = Rd.cast<float>(); delta.block<3,1>(0,3) = x.tail<3>().cast<float>();
            T = delta * T;
            res.numInliers = size_t(a.inliers);
            res.fitness = float(a.inliers) / float(src.size());
            if (x.norm() < params.convEps) break;
        }
        res.T = TcInv * T * Tc;                    // un-centre back to world
        res.valid = res.numInliers >= size_t(params.minInliers);
        return res;
    }

} // namespace Engine::Pipeline
