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
