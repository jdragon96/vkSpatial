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
