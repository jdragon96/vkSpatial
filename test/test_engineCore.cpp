#include <gtest/gtest.h>

#include "Engine/Core/Context.h"
#include "Engine/Core/OneShotCommands.h"

using namespace Engine::Core;

TEST(ContextTest, ComputeOnlyConstructionProducesValidHandles) {
    Context ctx;  // enablePresent=false by default

    EXPECT_NE(ctx.instance, VK_NULL_HANDLE);
    EXPECT_NE(ctx.physicalDevice, VK_NULL_HANDLE);
    EXPECT_NE(ctx.device, VK_NULL_HANDLE);
    EXPECT_NE(ctx.computeQueue, VK_NULL_HANDLE);
    EXPECT_NE(ctx.cmdPool, VK_NULL_HANDLE);
    EXPECT_NE(ctx.allocator, VK_NULL_HANDLE);

    EXPECT_EQ(ctx.graphicsQueue, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.graphicsCmdPool, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.surface, VK_NULL_HANDLE);
}

TEST(OneShotCommandsTest, FillBufferRoundTripsThroughSubmit) {
    Context ctx;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = 256;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo{};
    ASSERT_EQ(vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, &allocationInfo),
              VK_SUCCESS);

    // Zero the buffer up front so the post-fill check below is meaningful.
    std::memset(allocationInfo.pMappedData, 0xFF, 256);

    SubmitOneShot(ctx, QueueRole::Compute, [&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, buffer, 0, 256, 0x2A2A2A2Au);
    });

    const uint32_t *data = static_cast<const uint32_t *>(allocationInfo.pMappedData);
    for (int i = 0; i < 64; ++i)
        EXPECT_EQ(data[i], 0x2A2A2A2Au) << "word " << i;

    vmaDestroyBuffer(ctx.allocator, buffer, allocation);
}
