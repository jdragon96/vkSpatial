#include "Engine/Spatial/IResidencyBackend.h"
#include <gtest/gtest.h>

using namespace Engine::Spatial;

namespace {
// Minimal in-memory fake proving the interface is implementable without Vulkan.
class FakeBackend : public IResidencyBackend {
public:
    void BeginFrame(const Eigen::Vector3i &base) override { m_base = base; }
    void EnsureResident(const std::vector<DirectionalGroupKey> &r) override { m_last = r.size(); }
    void EndFrame() override {}
    // Interface expanded in Task 4 (Part A); this fake never had a batched residency path
    // of its own, so RecordResidency just mirrors EnsureResident's bookkeeping and records
    // nothing into `batch` (nothing in this test exercises the batch's contents).
    uint32_t RecordResidency(const std::vector<DirectionalGroupKey> &required,
                             Engine::Compute::CommandBatch &batch) override {
        (void)batch;
        m_last = required.size();
        return uint32_t(required.size());
    }
    const std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> &
    ResidentIndex() const override { return m_residentIndex; }
    VkBuffer IndexGridBuffer() const override { return VK_NULL_HANDLE; }
    VkBuffer PoolVoxelBuffer() const override { return VK_NULL_HANDLE; }
    VkBuffer MetaBuffer() const override { return VK_NULL_HANDLE; }
    uint32_t PoolCapacity() const override { return 0; }
    Eigen::Vector3i LocalBase() const override { return m_base; }
    bool IsUnified() const override { return true; }
    bool SupportsZeroCopyCpuAccess() const override { return true; }
    DirectionalHostStore &HostStore() override { return m_store; }
    ResidencyStats FrameStats() const override { return {}; }
    std::vector<uint32_t> DebugDownloadIndexGrid() override { return {}; }
    uint32_t DebugQueryPoolIndex(const DirectionalGroupKey &) override { return kInvalidPoolIndex; }
    DirectionalHostStore::Group DebugDownloadGroupVoxels(const DirectionalGroupKey &) override { return {}; }
private:
    Eigen::Vector3i m_base = Eigen::Vector3i::Zero();
    size_t m_last = 0;
    DirectionalHostStore m_store;
    std::unordered_map<DirectionalGroupKey, uint32_t, DirectionalGroupKeyHash> m_residentIndex;
};
} // namespace

TEST(ResidencyBackend, InterfaceIsImplementable) {
    FakeBackend b;
    IResidencyBackend &iface = b;
    iface.BeginFrame(Eigen::Vector3i(1, 2, 3));
    iface.EnsureResident({});
    EXPECT_EQ(iface.LocalBase(), Eigen::Vector3i(1, 2, 3));
    EXPECT_TRUE(iface.IsUnified());
}

#include "Engine/Core/Context.h"
#include "Engine/Spatial/StreamingResidencyBackend.h"

TEST(ResidencyBackend, StreamingBuildsAndClassifiesEmpty) {
    Engine::Core::Context ctx;
    StreamingResidencyBackend be;
    be.Build(ctx, /*poolCapacity=*/1024);
    EXPECT_FALSE(be.IsUnified());
    EXPECT_EQ(be.PoolCapacity(), 1024u);
    be.BeginFrame(Eigen::Vector3i(0, 0, 0));
    be.EnsureResident({}); // nothing required → no missing
    EXPECT_EQ(be.FrameStats().missingCount, 0u);
}

#include "Engine/Spatial/UnifiedResidencyBackend.h"

TEST(ResidencyBackend, UnifiedKeepsGroupsResidentZeroCopy) {
    Engine::Core::Context ctx;
    UnifiedResidencyBackend be;
    be.Build(ctx, /*initialCapacity=*/4096);
    ASSERT_TRUE(be.IsUnified());
    ASSERT_TRUE(be.SupportsZeroCopyCpuAccess());

    be.BeginFrame(Eigen::Vector3i(0, 0, 0));
    DirectionalGroupKey k{1, 2, 3, /*direction=*/0};
    be.EnsureResident({k});
    EXPECT_NE(be.DebugQueryPoolIndex(k), kInvalidPoolIndex);
    EXPECT_EQ(be.FrameStats().missingCount, 0u); // nothing "missing" — everything is resident
    EXPECT_EQ(be.FrameStats().h2dBytes, 0u);     // zero-copy: no upload bytes

    // Same key next frame → still resident, same slot (reuse), no growth.
    uint32_t slot = be.DebugQueryPoolIndex(k);
    be.BeginFrame(Eigen::Vector3i(0, 0, 0));
    be.EnsureResident({k});
    EXPECT_EQ(be.DebugQueryPoolIndex(k), slot);
}

#include "Engine/Spatial/IResidencyBackend.h"
#include <cstdlib>

TEST(ResidencyBackend, FactoryHonorsOverride) {
    Engine::Core::Context ctx;
#ifdef _WIN32
    _putenv_s("VKLBVH_RESIDENCY", "streaming");
#else
    setenv("VKLBVH_RESIDENCY", "streaming", 1);
#endif
    auto be = MakeResidencyBackend(ctx, 1024);
    ASSERT_NE(be, nullptr);
    EXPECT_FALSE(be->IsUnified());
#ifndef _WIN32
    unsetenv("VKLBVH_RESIDENCY");
#endif
}

#include "Engine/Spatial/DirectionalTSDF.h"
#include <algorithm>

namespace {
// Small synthetic frame: a planar patch of samples with +Z normals near the origin.
void makePlane(std::vector<Eigen::Vector3f> &pts, std::vector<Eigen::Vector3f> &nrm) {
    pts.clear(); nrm.clear();
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) {
            pts.emplace_back(i * 0.05f, j * 0.05f, 0.0f);
            nrm.emplace_back(0.0f, 0.0f, 1.0f);
        }
}

std::vector<Engine::Spatial::ExtractedPoint> runWith(Engine::Spatial::ResidencyMode mode) {
    Engine::Core::Context ctx;
    Engine::Spatial::DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 32768, 1u << 15, 1u << 16, mode);
    std::vector<Eigen::Vector3f> pts, nrm;
    makePlane(pts, nrm);
    tsdf.Integrate(pts, nrm, Eigen::Vector3f(0, 0, 1), Eigen::Vector3f::Zero());
    auto pc = tsdf.PointCloud();
    std::sort(pc.begin(), pc.end(), [](auto &a, auto &b) {
        if (a.position.x() != b.position.x()) return a.position.x() < b.position.x();
        if (a.position.y() != b.position.y()) return a.position.y() < b.position.y();
        if (a.position.z() != b.position.z()) return a.position.z() < b.position.z();
        if (a.normal.x() != b.normal.x()) return a.normal.x() < b.normal.x();
        if (a.normal.y() != b.normal.y()) return a.normal.y() < b.normal.y();
        return a.normal.z() < b.normal.z();
    });
    return pc;
}

// Build a DirectionalTSDF forcing Unified; false if the device has no UMA heap.
bool unifiedAvailable() {
    Engine::Core::Context ctx;
    try {
        Engine::Spatial::DirectionalTSDF t;
        t.Build(ctx, 0.1f, 0.3f, 32768, 1u << 15, 1u << 16,
                Engine::Spatial::ResidencyMode::Unified);
    } catch (...) {
        return false;
    }
    return true;
}
} // namespace

TEST(ResidencyBackend, CrossBackendReconstructionMatches) {
    if (!unifiedAvailable())
        GTEST_SKIP() << "no UMA heap on this device; cross-backend test needs unified support";

    // Explicit modes — no env. (If VKLBVH_RESIDENCY is set it would override; the
    // FactoryHonorsOverride test clears it, so nothing leaks into this run.)
    auto a = runWith(Engine::Spatial::ResidencyMode::Streaming);
    auto b = runWith(Engine::Spatial::ResidencyMode::Unified);
    ASSERT_GT(a.size(), 0u) << "extraction produced no points — cross-backend test would false-green";
    ASSERT_EQ(a.size(), b.size());
    const float eps = 1e-3f; // ε: positions match to 1 micron at 0.1mm voxel scale
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_LT((a[i].position - b[i].position).norm(), eps) << "point " << i;
        EXPECT_LT((a[i].normal - b[i].normal).norm(), eps) << "point " << i;
    }
}

#include "Engine/Spatial/DirectionalVoxelConvert.h"

TEST(DirectionalVoxelConvert, RoundTripPreservesValueWeightNormal) {
    using namespace Engine::Spatial;
    HostTsdfVoxel h{};
    h.value = 0.5f; h.weight = 3.0f;
    Eigen::Vector3f n = Eigen::Vector3f(0.2f, -0.3f, 0.9f).normalized();
    h.nx = n.x(); h.ny = n.y(); h.nz = n.z();

    const GpuTsdfVoxel g = HostVoxelToGpu(h);
    EXPECT_EQ(g.sumW, uint32_t(std::lround(3.0 * kTsdfFixedScale)));
    EXPECT_EQ(g.sumDW, int32_t(std::lround(0.5 * 3.0 * kTsdfFixedScale)));

    const HostTsdfVoxel back = GpuVoxelToHost(g);
    EXPECT_NEAR(back.value, 0.5f, 1e-3f);
    EXPECT_NEAR(back.weight, 3.0f, 1e-3f);
    EXPECT_NEAR(back.nx, n.x(), 2e-3f);
    EXPECT_NEAR(back.ny, n.y(), 2e-3f);
    EXPECT_NEAR(back.nz, n.z(), 2e-3f);
}

TEST(DirectionalVoxelConvert, DegenerateNormalStaysZero) {
    using namespace Engine::Spatial;
    GpuTsdfVoxel g{}; g.sumW = 10000; g.sumDW = 0; // sumN all zero
    const HostTsdfVoxel back = GpuVoxelToHost(g);
    EXPECT_FLOAT_EQ(back.nx, 0.0f);
    EXPECT_FLOAT_EQ(back.ny, 0.0f);
    EXPECT_FLOAT_EQ(back.nz, 0.0f);
}

TEST(ResidencyBackend, StreamingUploadRehydratesNormal) {
    using namespace Engine::Spatial;
    Engine::Core::Context ctx;
    StreamingResidencyBackend be;
    be.Build(ctx, /*poolCapacity=*/1024);

    DirectionalGroupKey key{0, 0, 0, /*direction=*/4}; // +Z layer
    DirectionalHostStore::Group g{};
    const Eigen::Vector3f n = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    for (auto &v : g) { v.value = 0.5f; v.weight = 2.0f; v.nx = n.x(); v.ny = n.y(); v.nz = n.z(); }
    be.HostStore().Put(key, g);

    be.BeginFrame(Eigen::Vector3i(0, 0, 0));
    be.EnsureResident({key});
    const auto back = be.DebugDownloadGroupVoxels(key); // uploaded sumN -> download normal

    EXPECT_NEAR(back[0].nz, 1.0f, 2e-3f);
    EXPECT_NEAR(back[0].nx, 0.0f, 2e-3f);
    EXPECT_NEAR(back[0].value, 0.5f, 2e-3f);
}
